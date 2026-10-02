"""Stage 4 — Run planning.

Code enumerates candidate commands from the RepoCard (README commands, entry-point
defaults, repo config files). The LLM only *chooses* a candidate per claim and may add
experiment-selecting arguments. Deterministic validation then checks every script,
flag, value and source, and computes paper-vs-repo hyperparameter differences for later
hypothesis reruns. Commands are argv lists; nothing is ever passed through a shell.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Optional

from sqlmodel import delete, select

from .. import llm, storage
from ..db import Claim, RunPlan, session
from ..grounding import normalise
from ..params import (CONFIG_FLAG_RE, OUTPUT_FLAG_RE, TRAINING_PARAMS, canonical, compare_hyperparameters,
                      effective_values)
from .ingest import InputError

SHELL_META = re.compile(r"[|;&<>`$]|\$\(|&&|\|\|")
DANGEROUS_SH = re.compile(r"\b(curl|wget|sudo|ssh|scp|nc|netcat|rm\s+-rf\s+/|pip3?\s+install|conda\s+install|"
                          r"apt(-get)?\s+install|git\s+clone|chmod\s+777|mkfs|dd\s+if=)\b", re.I)
ENV_PREFIX = re.compile(r"^((?:[A-Z_][A-Z0-9_]*=\S+\s+)+)")

SYSTEM = """You map each experimental claim from an ML paper to ONE command that reproduces it,
choosing ONLY from the numbered candidate commands provided (they come from the repository).
Rules:
- Pick the candidate that the authors' README or configs use for that experiment. Prefer README commands.
- Do NOT change training hyperparameters (learning rate, epochs, batch size, seed, optimizer, ...)
  even if the paper states different values: the primary run must use the authors' documented
  procedure. Paper/repo differences are investigated separately.
- `extra_args` may only select WHICH experiment runs (e.g. model size, dataset name, model variant)
  and only with values that literally appear in the claim's text. Use [] when not needed.
- Only use flags listed for the chosen candidate's script.
- If no candidate can produce the claim (e.g. the code for it is missing), set candidate_id to null,
  runnable=false and explain in reason.
- metric_location: where the metric is likely written ("stdout" or a file name) and the key/pattern."""

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["plans"],
    "properties": {"plans": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["claim_id", "candidate_id", "extra_args", "runnable", "reason", "metric_location",
                     "metric_key", "estimated_minutes", "confidence"],
        "properties": {
            "claim_id": {"type": "string"},
            "candidate_id": {"type": ["string", "null"]},
            "extra_args": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["flag", "value"],
                "properties": {"flag": {"type": "string"}, "value": {"type": "string"}}}},
            "runnable": {"type": "boolean"},
            "reason": {"type": "string"},
            "metric_location": {"type": "string"},
            "metric_key": {"type": ["string", "null"]},
            "estimated_minutes": {"type": "number"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        }}}},
}


# ------------------------------------------------------------ candidates

def build_candidates(card: dict[str, Any], repo: Path) -> list[dict[str, Any]]:
    facts = card["facts"]
    entry_files = {e["file"] for e in card["entrypoints"]}
    cands: list[dict[str, Any]] = []
    covered: set[str] = set()

    def add(script: str, args: list[str], source: str, ref: Optional[str], note: str = "",
            env_dropped: Optional[str] = None) -> None:
        argv = ["python", script, *args] if script.endswith(".py") else ["bash", script, *args]
        key = shlex.join(argv)
        if any(c["command"] == key for c in cands):
            return
        cands.append({"id": f"K{len(cands) + 1}", "script": script, "args": args, "argv": argv, "command": key,
                      "source": source, "source_ref": ref, "note": note, "env_dropped": env_dropped})

    for rc in card["readme_commands"]:
        cmd = rc["command"]
        env = None
        if m := ENV_PREFIX.match(cmd):
            env, cmd = m.group(1).strip(), cmd[m.end():]
        if SHELL_META.search(cmd):
            continue                      # pipes/redirects/chains are not runnable as argv
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            continue
        script, args = _script_and_args(tokens)
        if not script or not (repo / script).is_file():
            continue
        add(script, args, "readme", f"{rc['file']}:{rc['line']}", env_dropped=env)
        covered.add(script)

    config_files = sorted({f["file"] for f in facts if f["kind"] == "config_value"})
    for e in card["entrypoints"]:
        flags = [a["name"] for a in e["args"]]
        if e["file"] not in covered and not any(a["required"] for a in e["args"]):
            add(e["file"], [], "entrypoint_defaults", f"{e['file']}:1", "argparse defaults")
        cfg_flag = next((f for f in flags if CONFIG_FLAG_RE.match(f)), None)
        if cfg_flag:
            base = next((c for c in cands if c["script"] == e["file"] and c["source"] == "readme"), None)
            for cfg in config_files:
                base_args = [a for a in (base["args"] if base else []) if a != cfg_flag]
                add(e["file"], [*base_args, cfg_flag, cfg], "repo_config", cfg,
                    f"README command with {cfg}" if base else "defaults + config file")
    return [c for c in cands if c["script"] in entry_files or not c["script"].endswith(".py")
            or c["source"] == "readme"]


def _script_and_args(tokens: list[str]) -> tuple[Optional[str], list[str]]:
    for i, tok in enumerate(tokens):
        if re.fullmatch(r"(?:\S*/)?python3?(?:\.\d+)?", tok):
            rest = tokens[i + 1:]
            if rest[:1] == ["-m"] and len(rest) > 1:
                return rest[1].replace(".", "/") + ".py", rest[2:]
            if rest and rest[0].endswith(".py"):
                return rest[0].lstrip("./"), rest[1:]
            return None, []
        if tok in ("bash", "sh") and i + 1 < len(tokens) and tokens[i + 1].endswith(".sh"):
            return tokens[i + 1].lstrip("./"), tokens[i + 2:]
    return None, []


# ------------------------------------------------------------ validation

def validate_plan(raw: dict[str, Any], claim: dict[str, Any], cands: dict[str, dict[str, Any]],
                  card: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Return a plan dict with validation {ok, errors, warnings}. Never trusts the LLM's text."""
    errors: list[str] = []
    warnings: list[str] = []
    cand = cands.get(raw.get("candidate_id") or "")
    if raw.get("candidate_id") and cand is None:
        errors.append(f"candidate {raw['candidate_id']!r} does not exist")
    if cand is None:
        return {"runnable": False, "candidate": None, "argv": None, "extra_args": [],
                "reason": raw.get("reason") or "no candidate command reproduces this claim",
                "validation": {"ok": not errors, "errors": errors, "warnings": warnings}}

    script = cand["script"]
    try:
        target = storage.safe_join(repo, script)
    except ValueError:
        errors.append(f"script path escapes the repository: {script}")
        target = None
    if target is not None and not target.is_file():
        errors.append(f"script {script} does not exist")

    entry = next((e for e in card["entrypoints"] if e["file"] == script), None)
    arg_specs = {flag: a for a in (entry["args"] if entry else []) for flag in a["flags"]}
    extra_argv: list[str] = []
    claim_text = normalise(" ".join(filter(None, [claim["experiment"], claim.get("dataset"),
                                                  claim.get("split"), claim["quote"]])))
    passed_flags = {a for a in cand["args"] if a.startswith("-")}
    for item in raw.get("extra_args") or []:
        flag, value = item["flag"].strip(), item["value"].strip()
        spec = arg_specs.get(flag)
        if spec is None:
            errors.append(f"{flag} is not a CLI flag of {script}")
            continue
        if flag in passed_flags:
            errors.append(f"{flag} is already set by the documented command")
            continue
        canon = canonical(spec["name"])
        if canon in TRAINING_PARAMS:
            errors.append(f"{flag} is a training hyperparameter; the primary run must keep the repository's value")
            continue
        if CONFIG_FLAG_RE.match(flag) or OUTPUT_FLAG_RE.match(flag):
            errors.append(f"{flag} must come from a repository candidate, not be added")
            continue
        if not value or "\n" in value or value.startswith(("/", "~")) or ".." in value:
            errors.append(f"unsafe value for {flag}: {value!r}")
            continue
        if normalise(value) not in claim_text:
            errors.append(f"value {value!r} for {flag} does not appear in the claim text")
            continue
        if spec.get("choices") and value not in [str(c) for c in spec["choices"]]:
            errors.append(f"{value!r} is not an allowed choice for {flag}")
            continue
        if spec.get("type") in ("int", "float"):
            try:
                (int if spec["type"] == "int" else float)(value)
            except ValueError:
                errors.append(f"{flag} expects {spec['type']}, got {value!r}")
                continue
        if spec.get("action") in ("store_true", "store_false"):
            extra_argv.append(flag)
        else:
            extra_argv += [flag, value]

    if script.endswith(".sh") and target is not None and target.is_file():
        body = target.read_text(encoding="utf-8", errors="replace")
        if m := DANGEROUS_SH.search(body):
            errors.append(f"{script} contains a disallowed command: {m.group(0)!r}")
        warnings.append("shell script: its internal commands are not flag-validated")
    for a in (entry["args"] if entry else []):
        if a["required"] and not any(f in cand["args"] + extra_argv for f in a["flags"]):
            errors.append(f"required argument {a['name']} is missing")
    if cand.get("env_dropped"):
        warnings.append(f"environment prefix dropped from README command: {cand['env_dropped']}")

    gpu_only = script in card["summary"]["gpu_only_files"]
    runnable = bool(raw.get("runnable", True)) and not gpu_only and not errors
    reason = raw.get("reason") or ""
    if gpu_only:
        reason = f"{script} requires a GPU (CUDA calls with no CPU fallback); the sandbox is CPU-only"
    argv = [*cand["argv"], *extra_argv]
    return {"runnable": runnable, "candidate": cand, "argv": argv, "extra_args": extra_argv, "reason": reason,
            "validation": {"ok": not errors, "errors": errors, "warnings": warnings}}


# ------------------------------------------------------------ dev planner

def dev_choose(claims: list[dict[str, Any]], cands: list[dict[str, Any]]) -> dict[str, Any]:
    """No-LLM fallback: pick the candidate whose script/config/args share the most words
    with the claim (README commands first on ties). Labelled dev_sample downstream."""
    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]{3,}|\d+", text.lower()) if w not in {"the", "and", "with", "test"}}

    plans = []
    for c in claims:
        cw = words(" ".join(filter(None, [c["experiment"], c.get("dataset")])))
        ranked = sorted(cands, key=lambda k: (-len(cw & words(k["command"])), k["source"] != "readme", k["id"]))
        best = ranked[0] if ranked else None
        plans.append({"claim_id": c["id"], "candidate_id": best["id"] if best else None, "extra_args": [],
                      "runnable": best is not None, "reason": "" if best else "no candidate commands",
                      "metric_location": "stdout", "metric_key": c["metric"], "estimated_minutes": 5,
                      "confidence": "low"})
    return {"plans": plans}


# ------------------------------------------------------------------ stage

def _prompt(claims: list[dict[str, Any]], cands: list[dict[str, Any]], card: dict[str, Any],
            feedback: Optional[dict[str, list[str]]] = None) -> str:
    entry_args = {e["file"]: [{"flag": a["name"], "default": a["default"], "help": a["help"]} for a in e["args"]]
                  for e in card["entrypoints"]}
    payload = {
        "claims": [{k: c[k] for k in ("id", "experiment", "dataset", "split", "metric", "reported_value", "quote")}
                   for c in claims],
        "candidates": [{"id": k["id"], "command": k["command"], "source": k["source"],
                        "source_ref": k["source_ref"], "note": k["note"]} for k in cands],
        "script_flags": entry_args,
        "gpu_only_scripts": card["summary"]["gpu_only_files"],
    }
    text = json.dumps(payload, indent=1, default=str)
    if feedback:
        text += "\n\nYour previous plans for these claims FAILED validation. Fix them:\n" + json.dumps(feedback, indent=1)
    return text


def run(job_id: str, job: dict[str, Any], log: Callable[..., None]) -> dict[str, Any]:
    card = storage.read_stage(job_id, "scan")
    repo = storage.repo_dir(job_id)
    with session() as s:
        rows = s.exec(select(Claim).where(Claim.job_id == job_id).order_by(Claim.id)).all()
    claims = [{**r.model_dump(), "hyperparameters": json.loads(r.hyperparameters)} for r in rows]
    grounded = [c for c in claims if c["grounded"]]
    skipped = [c["id"] for c in claims if not c["grounded"]]
    if skipped:
        log(f"Not planning unverified claim(s) {', '.join(skipped)}: they are excluded from scoring")

    cands = build_candidates(card, repo)
    cand_by_id = {c["id"]: c for c in cands}
    log(f"{len(cands)} candidate command(s) from the repository",
        data={"candidates": [{"id": c["id"], "command": c["command"], "source": c["source"]} for c in cands]})
    if not grounded or not cands:
        if not cands:
            log("No runnable candidate command found in the repository", level="warning")
        plans_out = [_not_runnable(c, "no candidate command in the repository" if not cands else "") for c in grounded]
        _persist(job_id, plans_out)
        return {"origin": "none", "candidates": cands, "plans": plans_out, "skipped_unverified": skipped}

    def ask(subset: list[dict[str, Any]], feedback: Optional[dict[str, list[str]]] = None) -> tuple[dict, str]:
        try:
            res = llm.complete_json(name="run_plans", schema=PLAN_SCHEMA, system=SYSTEM,
                                    user=_prompt(subset, cands, card, feedback),
                                    dev_sample=lambda: dev_choose(subset, cands), max_tokens=8000)
        except llm.LLMError as exc:
            raise InputError(str(exc)) from exc
        return res.data, res.origin

    log("Asking the LLM to match claims to candidate commands" if job["llm_mode"] == "live"
        else "Dev mode: matching claims to commands by word overlap (no LLM)")
    data, origin = ask(grounded)
    raw_by_claim = {p["claim_id"]: p for p in data.get("plans", []) if isinstance(p, dict)}

    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, list[str]] = {}
    for c in grounded:
        raw = raw_by_claim.get(c["id"]) or {"candidate_id": None, "runnable": False,
                                           "reason": "the LLM returned no plan for this claim"}
        v = validate_plan(raw, c, cand_by_id, card, repo)
        results[c["id"]] = {"raw": raw, **v}
        if not v["validation"]["ok"]:
            failures[c["id"]] = v["validation"]["errors"]

    if failures and origin == "llm":
        log(f"{len(failures)} plan(s) failed validation; asking the LLM once more with the errors",
            level="warning", data={"errors": failures})
        retry_claims = [c for c in grounded if c["id"] in failures]
        data2, _ = ask(retry_claims, failures)
        for p in data2.get("plans", []):
            c = next((x for x in retry_claims if x["id"] == p.get("claim_id")), None)
            if c is None:
                continue
            v = validate_plan(p, c, cand_by_id, card, repo)
            results[c["id"]] = {"raw": p, **v, "retried": True}

    plans_out = []
    for i, c in enumerate(grounded, 1):
        r = results[c["id"]]
        cand = r["candidate"]
        plan: dict[str, Any] = {
            "id": f"P{i}", "claim_id": c["id"], "origin": origin, "runnable": r["runnable"],
            "reason": r["reason"] if not r["validation"]["ok"] or not r["runnable"] else "",
            "validation": r["validation"], "retried": r.get("retried", False),
            "argv": r["argv"], "command": shlex.join(r["argv"]) if r["argv"] else None,
            "script": cand["script"] if cand else None,
            "source": cand["source"] if cand else None, "source_ref": cand["source_ref"] if cand else None,
            "extra_args": r["extra_args"],
            "metric_location": r["raw"].get("metric_location"), "metric_key": r["raw"].get("metric_key"),
            "estimated_minutes": r["raw"].get("estimated_minutes"), "confidence": r["raw"].get("confidence"),
            "overrides": [],
        }
        if not r["validation"]["ok"]:
            plan["runnable"] = False
            plan["reason"] = "no valid plan after retry: " + "; ".join(r["validation"]["errors"])
        if cand:
            eff = effective_values(cand["script"], plan["argv"][2:], card["facts"], cand["source_ref"])
            plan["overrides"] = compare_hyperparameters(c["hyperparameters"], eff)
        plans_out.append(plan)

    # identical argv -> one execution shared by several claims (e.g. accuracy and F1 of one run)
    groups: dict[str, list[str]] = {}
    for p in plans_out:
        if p["runnable"]:
            groups.setdefault(p["command"], []).append(p["id"])
    for p in plans_out:
        p["run_group"] = next((k for k, (cmd, ids) in enumerate(groups.items(), 1) if p["id"] in ids), None)

    _persist(job_id, plans_out)
    n_run = sum(p["runnable"] for p in plans_out)
    mism = [f"{p['claim_id']}:{o['param']}" for p in plans_out for o in p["overrides"] if o["status"] == "mismatch"]
    log(f"{n_run} of {len(plans_out)} claim(s) runnable as {len(groups)} distinct command(s)",
        data={"plans": [{"id": p["id"], "claim": p["claim_id"], "command": p["command"], "runnable": p["runnable"]}
                        for p in plans_out]})
    for p in plans_out:
        if not p["runnable"]:
            log(f"{p['claim_id']} will not run: {p['reason']}", level="warning")
    if mism:
        log(f"Paper vs repository differences found: {', '.join(mism)} (kept out of the primary run)",
            level="warning")
    return {"origin": origin, "candidates": cands, "plans": plans_out, "skipped_unverified": skipped,
            "groups": [{"command": cmd, "plans": ids} for cmd, ids in groups.items()]}


def _not_runnable(claim: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"id": f"P-{claim['id']}", "claim_id": claim["id"], "origin": "none", "runnable": False,
            "reason": reason, "validation": {"ok": True, "errors": [], "warnings": []}, "argv": None,
            "command": None, "script": None, "source": None, "source_ref": None, "extra_args": [],
            "metric_location": None, "metric_key": None, "estimated_minutes": None, "confidence": None,
            "overrides": [], "run_group": None}


def _persist(job_id: str, plans: list[dict[str, Any]]) -> None:
    with session() as s:
        s.exec(delete(RunPlan).where(RunPlan.job_id == job_id))
        for p in plans:
            s.add(RunPlan(
                job_id=job_id, id=p["id"], claim_id=p["claim_id"], script=p["script"] or "",
                command=p["command"] or "", source_of_command=p["source_ref"],
                expected_metric_location=p["metric_location"],
                paper_overrides=json.dumps(p["overrides"]), estimated_minutes=p["estimated_minutes"],
                confidence=p["confidence"],
                validation=json.dumps({**p["validation"], "runnable": p["runnable"], "reason": p["reason"],
                                       "argv": p["argv"], "origin": p["origin"]})))
        s.commit()
