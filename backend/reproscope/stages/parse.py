"""Stage 6 — Result parsing.

For every planned claim, find the metric value the run actually produced:
  1. structured output files written by the run (JSON / CSV)
  2. recognised metric patterns in the run log (prefers test/eval lines, last occurrence)
  3. LLM fallback over the log tail, accepted ONLY if the quoted line exists verbatim in the
     log and contains the value
Values are normalised to the claim's unit; every value carries file/line evidence.
Nothing is guessed: no evidence -> NO_METRIC.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from sqlmodel import delete, select

from .. import llm, storage
from ..db import Claim, Result, Run, session
from ..grounding import number_forms
from .ingest import InputError

# metric family -> regex matching how code names / prints it
FAMILIES: dict[str, str] = {
    "accuracy": r"(?:top-?1[ _-]?)?acc(?:uracy)?",
    "balanced_accuracy": r"balanced[ _-]?acc(?:uracy)?",
    "f1": r"(?:macro|micro|weighted)?[ _-]?f1(?:[ _-]?score)?|f-?measure",
    "precision": r"precision",
    "recall": r"recall|sensitivity",
    "auc": r"(?:roc[ _-]?)?auc(?:[ _-]?roc)?",
    "rmse": r"rmse|root[ _-]mean[ _-]squared[ _-]error",
    "mae": r"mae|mean[ _-]absolute[ _-]error",
    "mse": r"(?<!r)mse|mean[ _-]squared[ _-]error",
    "r2": r"r2|r\^2|r²|r[ _-]?squared",
    "loss": r"loss",
    "bleu": r"bleu",
    "rouge": r"rouge[ _-]?(?:l|1|2)?",
    "perplexity": r"perplexity|ppl",
    "error": r"error(?:[ _-]?rate)?|err",
}
VARIANTS = ("macro", "micro", "weighted", "top-5", "top5")
NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

LLM_SYSTEM = """You read the end of an experiment's log and find ONE metric value.
Return found=false unless a line in the log explicitly reports this metric for the evaluation the
claim describes. `line` must be copied EXACTLY from the log (one full line). Never compute or infer values."""
LLM_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["found", "value", "line", "reason"],
    "properties": {"found": {"type": "boolean"}, "value": {"type": ["number", "null"]},
                   "line": {"type": ["string", "null"]}, "reason": {"type": "string"}},
}


def family_of(metric: str) -> Optional[str]:
    m = metric.lower()
    checks = [("balanced_accuracy", r"balanced"), ("f1", r"f1|f-?measure|f-?score"), ("auc", r"auc"),
              ("rmse", r"rmse|root mean"), ("mae", r"\bmae\b|mean absolute"), ("r2", r"\br2\b|r\^2|r²|r-squared"),
              ("mse", r"\bmse\b|mean squared"), ("perplexity", r"perplexity|ppl"), ("bleu", r"bleu"),
              ("rouge", r"rouge"), ("precision", r"precision"), ("recall", r"recall|sensitivity"),
              ("error", r"error"), ("loss", r"loss"), ("accuracy", r"acc")]
    return next((fam for fam, pat in checks if re.search(pat, m)), None)


def variant_of(text: str) -> Optional[str]:
    t = text.lower()
    return next((v for v in VARIANTS if v in t), None)


def normalise_value(raw: float, claim_unit: str, printed_percent: bool) -> tuple[float, Optional[str]]:
    if claim_unit == "percent" and not printed_percent and 0 <= raw <= 1:
        return round(raw * 100, 6), "converted from a fraction to percent (×100)"
    if claim_unit == "fraction" and (printed_percent or raw > 1) and raw <= 100:
        return round(raw / 100, 8), "converted from percent to a fraction (÷100)"
    return raw, None


# ------------------------------------------------------------- 1. files

def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return _flatten(obj[-1], f"{prefix}[-1]")         # last record (e.g. final epoch)
    return [(prefix, obj)]


def _key_score(key: str, family: str, variant: Optional[str], hint: Optional[str]) -> Optional[int]:
    k = key.lower()
    leaf = re.split(r"[.\[\]]", k)[-1] or k
    if not re.search(rf"(?:^|[^a-z]){FAMILIES[family]}(?:$|[^a-z])", leaf):
        return None
    score = 10
    if hint and hint.lower() == leaf:
        score += 20
    if re.search(r"test|eval", k):
        score += 6
    if re.search(r"train", k):
        score -= 8
    if re.search(r"\bval|valid", k):
        score -= 3
    if variant and variant in k:
        score += 5
    elif variant is None and any(v in k for v in VARIANTS):
        score -= 2
    if family == "accuracy" and re.search(r"balanced|top-?5", k):
        return None
    return score


def from_files(outputs_dir: Path, outputs: list[dict[str, Any]], family: str, variant: Optional[str],
               hint: Optional[str]) -> Optional[dict[str, Any]]:
    best: Optional[tuple[int, dict[str, Any]]] = None
    for o in outputs:
        if o.get("skipped"):
            continue
        path = outputs_dir / o["path"]
        suffix = path.suffix.lower()
        if suffix not in (".json", ".csv", ".jsonl") or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            pairs: list[tuple[str, Any]] = []
            if suffix == ".json":
                pairs = _flatten(json.loads(text))
            elif suffix == ".jsonl":
                last = [ln for ln in text.splitlines() if ln.strip()][-1]
                pairs = _flatten(json.loads(last))
            else:
                rows = list(csv.DictReader(io.StringIO(text)))
                if rows:
                    pairs = list(rows[-1].items())
        except (ValueError, IndexError):
            continue
        for key, value in pairs:
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            score = _key_score(key, family, variant, hint)
            if score is None:
                continue
            cand = {"raw_value": num, "source": "file", "printed_percent": False,
                    "evidence": {"file": o["path"], "key": key}, "key_text": key}
            if best is None or score > best[0]:
                best = (score, cand)
    return best[1] if best else None


# --------------------------------------------------------------- 2. log

def from_log(log_text: str, family: str, variant: Optional[str]) -> tuple[Optional[dict[str, Any]], int]:
    pattern = re.compile(rf"(?<![a-z])({FAMILIES[family]})(?![a-z])[^0-9\n-]{{0,25}}?{NUM}\s*(%)?", re.I)
    hits = []
    for no, line in enumerate(log_text.splitlines(), 1):
        for m in pattern.finditer(line):
            name = m.group(1).lower()
            if family == "accuracy" and re.search(r"balanced|top-?5", line[max(0, m.start() - 12):m.end()], re.I):
                continue
            score = 0
            if re.search(r"\b(test|eval|evaluation|final)\b", line, re.I):
                score += 6
            if re.search(r"\btrain(ing)?\b", line, re.I):
                score -= 8
            if re.search(r"\b(val|valid|validation|dev)\b", line, re.I):
                score -= 3
            if variant and variant in line.lower():
                score += 5
            hits.append((score, no, {"raw_value": float(m.group(2)), "source": "regex",
                                     "printed_percent": bool(m.group(3)),
                                     "evidence": {"line_no": no, "line_text": line.strip()[:300]},
                                     "key_text": name + " " + line.lower()}))
    if not hits:
        return None, 0
    top = max(h[0] for h in hits)
    best = [h for h in hits if h[0] == top][-1]          # last line among the best-scoring
    return best[2], len(hits)


# ---------------------------------------------------------- 3. LLM

def from_llm(log_text: str, claim: dict[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
    lines = log_text.splitlines()
    tail = lines[-200:]
    offset = len(lines) - len(tail)
    prompt = json.dumps({"metric": claim["metric"], "experiment": claim["experiment"],
                         "dataset": claim.get("dataset"), "split": claim.get("split"),
                         "log_tail": "\n".join(tail)}, indent=1)
    try:
        res = llm.complete_json(name="metric_from_log", schema=LLM_SCHEMA, system=LLM_SYSTEM, user=prompt,
                                dev_sample=lambda: {"found": False, "value": None, "line": None,
                                                    "reason": "dev mode: no LLM fallback"}, max_tokens=3000)
    except llm.LLMError as exc:
        raise InputError(str(exc)) from exc
    d = res.data
    if not d.get("found") or d.get("value") is None or not d.get("line"):
        return None, d.get("reason") or "LLM found no matching line"
    quoted = d["line"].strip()
    for i, line in enumerate(tail):
        if line.strip() == quoted:
            line_norm = line.lower()
            if any(re.search(rf"(?<![\d.]){re.escape(f)}(?!\d)", line_norm) for f in number_forms(float(d["value"]))):
                return {"raw_value": float(d["value"]), "source": "llm", "printed_percent": "%" in line,
                        "evidence": {"line_no": offset + i + 1, "line_text": line.strip()[:300],
                                     "llm_reason": d.get("reason")}, "key_text": line.lower()}, ""
            return None, "LLM-quoted line does not contain the value it reported (rejected)"
    return None, "LLM-quoted line does not appear in the log (rejected)"


# -------------------------------------------------------------- measure

def measure(log_text: str, outputs: list[dict[str, Any]], outputs_dir: Path, claim: dict[str, Any],
            metric_key: Optional[str] = None, use_llm: bool = True) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Find `claim`'s metric in one run's outputs/log. Returns (normalised result or None, attempts).
    Used for primary, repeat and hypothesis runs alike so every value is measured the same way."""
    family = family_of(claim["metric"])
    variant = variant_of(claim["metric"])
    if family is None:
        return None, [f"metric '{claim['metric']}' is not a recognised metric family"]
    found = from_files(outputs_dir, outputs, family, variant, metric_key)
    attempts = ["files: " + ("found" if found else f"no {family} value in {len(outputs)} output file(s)")]
    n_hits = 0
    if found is None:
        found, n_hits = from_log(log_text, family, variant)
        attempts.append("log patterns: " + ("found" if found else "no matching line"))
    if found is None and use_llm and log_text.strip():
        found, why = from_llm(log_text, claim)
        attempts.append("LLM fallback: " + ("verified line found" if found else why))
    if found is None:
        return None, attempts
    value, note = normalise_value(found["raw_value"], claim["unit"], found["printed_percent"])
    found_variant = variant_of(found.get("key_text", ""))
    variant_note = None
    if variant and found_variant != variant:
        variant_note = (f"paper reports {variant} {family}; the run reports "
                        f"{(found_variant + ' ') if found_variant else 'an unspecified '}{family}")
    return {"value": value, "raw_value": found["raw_value"], "source": found["source"],
            "evidence": found["evidence"], "normalisation": note, "variant_note": variant_note,
            "log_matches": n_hits}, attempts


# -------------------------------------------------------------- stage

def run(job_id: str, job: dict[str, Any], log: Callable[..., None]) -> dict[str, Any]:
    plan_out = storage.read_stage(job_id, "plan") or {"plans": []}
    exec_out = storage.read_stage(job_id, "execute") or {"runs": [], "plan_runs": {}}
    with session() as s:
        claims = {c.id: c.model_dump() for c in s.exec(select(Claim).where(Claim.job_id == job_id)).all()}
        runs = {r.id: r for r in s.exec(select(Run).where(Run.job_id == job_id, Run.kind == "primary")).all()}
        s.exec(delete(Result).where(Result.job_id == job_id))
        s.commit()
    run_summaries = {r["id"]: r for r in exec_out.get("runs", [])}

    results: list[dict[str, Any]] = []
    for plan in plan_out["plans"]:
        claim = claims.get(plan["claim_id"])
        if claim is None:
            continue
        base = {"claim_id": claim["id"], "plan_id": plan["id"], "metric": claim["metric"], "unit": claim["unit"]}
        rid = exec_out.get("plan_runs", {}).get(plan["id"])
        if not plan["runnable"] or rid is None:
            results.append({**base, "status": "not_run", "run_id": None,
                            "reason": plan.get("reason") or "no runnable plan"})
            continue
        run_row, summary = runs.get(rid), run_summaries.get(rid, {})
        if run_row is None or run_row.status != "ok":
            status = run_row.status if run_row else "missing"
            results.append({**base, "status": "not_run", "run_id": rid, "run_status": status,
                            "reason": summary.get("reason") or f"run {rid} ended with status '{status}'"})
            continue

        log_text = (storage.job_dir(job_id) / run_row.log_path).read_text(encoding="utf-8", errors="replace") \
            if run_row.log_path else ""
        outputs = json.loads(run_row.environment).get("outputs", [])
        found, attempts = measure(log_text, outputs, storage.runs_dir(job_id) / rid / "outputs", claim,
                                  plan.get("metric_key"))
        if found is None:
            results.append({**base, "status": "no_metric", "run_id": rid, "attempts": attempts,
                            "reason": attempts[0] if family_of(claim["metric"]) is None
                            else f"the run did not report {claim['metric']}"})
            continue
        results.append({**base, "status": "found", "run_id": rid, **found, "attempts": attempts})

    with session() as s:
        for r in results:
            if r["status"] != "found":
                continue
            s.merge(Result(job_id=job_id, run_id=r["run_id"], metric=f"{r['metric']}#{r['claim_id']}",
                           claim_id=r["claim_id"], value=r["value"], raw_value=r["raw_value"], source=r["source"],
                           evidence=json.dumps({**r["evidence"], "normalisation": r["normalisation"],
                                                "variant_note": r["variant_note"]})))
        s.commit()

    for r in results:
        if r["status"] == "found":
            ev = r["evidence"]
            where = f"{ev['file']} [{ev['key']}]" if "file" in ev else f"log line {ev['line_no']}: {ev['line_text']}"
            log(f"{r['claim_id']}: {r['metric']} = {r['value']:g} from {r['source']} ({where})"
                + (f"; {r['normalisation']}" if r["normalisation"] else ""))
            if r["variant_note"]:
                log(f"{r['claim_id']}: {r['variant_note']}", level="warning")
        elif r["status"] == "no_metric":
            log(f"{r['claim_id']}: no {r['metric']} value found ({'; '.join(r.get('attempts', [])) or r['reason']})",
                level="warning")
        else:
            log(f"{r['claim_id']}: not run — {r['reason']}", level="warning")
    found_n = sum(r["status"] == "found" for r in results)
    log(f"Obtained values for {found_n} of {len(results)} planned claim(s)")
    return {"results": results}
