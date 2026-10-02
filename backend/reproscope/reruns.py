"""User-approved hypothesis reruns (POST /api/jobs/{id}/rerun).

A person picks a claim and up to three CLI overrides. The overrides are validated against
the script's real argparse/click flags (same rules as the planner), the claim's command is
re-run in the sandbox with ONLY those changes, as many times as the primary measurement,
and the gap is compared using the claim's existing tolerance. Results are stored as JSON
under artifacts/{job}/reruns/ and appear in the report.
"""
from __future__ import annotations

import json
import re
import statistics
import threading
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from sqlmodel import select

from . import storage
from .db import Claim, Comparison, session, utcnow
from .events import bus
from .params import CONFIG_FLAG_RE, OUTPUT_FLAG_RE
from .verdicts import classify_hypothesis

MAX_OVERRIDES = 3
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class RerunRequest(BaseModel):
    claim_id: str = Field(min_length=1, max_length=20)
    overrides: dict[str, str] = Field(min_length=1)
    note: Optional[str] = Field(default=None, max_length=300)

    @field_validator("overrides")
    @classmethod
    def _limit(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > MAX_OVERRIDES:
            raise ValueError(f"at most {MAX_OVERRIDES} overrides per rerun (change one thing at a time)")
        return {k.strip(): str(val).strip() for k, val in v.items()}


class RerunError(ValueError):
    """Invalid rerun request (HTTP 400/409)."""


def _rerun_dir(job_id: str):
    return storage.job_dir(job_id) / "reruns"


def list_reruns(job_id: str) -> list[dict[str, Any]]:
    d = _rerun_dir(job_id)
    if not d.is_dir():
        return []
    items = [json.loads(p.read_text(encoding="utf-8")) for p in d.glob("U*.json")]
    return sorted(items, key=lambda r: int(r["id"][1:]))


def _save(job_id: str, rec: dict[str, Any]) -> None:
    d = _rerun_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rec['id']}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


def validate(job_id: str, req: RerunRequest) -> dict[str, Any]:
    """Check the request against the plan and the script's flags. Returns the prepared rerun."""
    plan_out = storage.read_stage(job_id, "plan") or {"plans": []}
    exec_out = storage.read_stage(job_id, "execute") or {"plan_runs": {}}
    card = storage.read_stage(job_id, "scan") or {"entrypoints": []}
    plan = next((p for p in plan_out["plans"] if p["claim_id"] == req.claim_id), None)
    if plan is None or not plan.get("runnable") or not plan.get("argv"):
        raise RerunError(f"Claim {req.claim_id} has no runnable command to rerun.")
    with session() as s:
        comp = s.get(Comparison, (job_id, req.claim_id))
    if comp is None or comp.obtained is None:
        raise RerunError(f"Claim {req.claim_id} has no measured result to compare a rerun against.")
    entry = next((e for e in card["entrypoints"] if e["file"] == plan["script"]), None)
    specs = {flag: a for a in (entry["args"] if entry else []) for flag in a["flags"]}
    argv = list(plan["argv"])
    for flag, value in req.overrides.items():
        spec = specs.get(flag)
        if spec is None:
            raise RerunError(f"{flag} is not a CLI flag of {plan['script']}.")
        if CONFIG_FLAG_RE.match(flag) or OUTPUT_FLAG_RE.match(flag):
            raise RerunError(f"{flag} selects a config or output file and cannot be overridden here.")
        if not value or len(value) > 100 or "\n" in value or value.startswith(("/", "~")) or ".." in value \
                or not re.fullmatch(r"[\w.+\-:,=@%]+", value):
            raise RerunError(f"Unsafe or empty value for {flag}: {value!r}.")
        if spec.get("type") in ("int", "float"):
            try:
                (int if spec["type"] == "int" else float)(value)
            except ValueError as exc:
                raise RerunError(f"{flag} expects {spec['type']}, got {value!r}.") from exc
        if spec.get("choices") and value not in [str(c) for c in spec["choices"]]:
            raise RerunError(f"{value!r} is not an allowed choice for {flag}.")
        if spec.get("action") in ("store_true", "store_false"):
            raise RerunError(f"{flag} is a switch; switches cannot be set to a value here.")
        argv = _set_flag(argv, flag, value)
    primary_run = exec_out.get("plan_runs", {}).get(plan["id"])
    group = [p["claim_id"] for p in plan_out["plans"]
             if exec_out.get("plan_runs", {}).get(p["id"]) == primary_run and primary_run]
    return {"plan": plan, "argv": argv, "claim_ids": group or [req.claim_id],
            "repeats": max(1, len(json.loads(comp.seed_values or "[]")))}


def _set_flag(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    for i, tok in enumerate(out):
        if tok == flag and i + 1 < len(out):
            out[i + 1] = value
            return out
        if tok.startswith(flag + "="):
            out[i] = f"{flag}={value}"
            return out
    return out + [flag, value]


def start(job_id: str, req: RerunRequest) -> dict[str, Any]:
    with _locks_guard:
        lock = _locks.setdefault(job_id, threading.Lock())
    if lock.locked():
        raise RerunError("Another rerun is already running for this analysis; wait for it to finish.")
    prepared = validate(job_id, req)
    existing = list_reruns(job_id)
    # number after the highest existing id (not the count), so a deleted rerun never causes a collision
    rid = f"U{max((int(r['id'][1:]) for r in existing), default=0) + 1}"
    rec = {"id": rid, "claim_id": req.claim_id, "claim_ids": prepared["claim_ids"], "overrides": req.overrides,
           "note": req.note, "argv": prepared["argv"], "repeats": prepared["repeats"], "status": "queued",
           "run_ids": [], "results": {}, "error": None, "created_at": utcnow(), "finished_at": None}
    _save(job_id, rec)
    threading.Thread(target=_execute, args=(job_id, rec, lock), daemon=True, name=f"rerun-{job_id}-{rid}").start()
    return rec


def _execute(job_id: str, rec: dict[str, Any], lock: threading.Lock) -> None:
    from .stages.compare import _run_and_measure          # shared sandbox + measurement path
    with lock:
        rec["status"] = "running"
        _save(job_id, rec)
        change = ", ".join(f"{k} {v}" for k, v in rec["overrides"].items())
        bus.emit(job_id, f"User-approved rerun {rec['id']}: {' '.join(rec['argv'])} ({rec['repeats']} run(s))",
                 stage="compare", level="info", data={"rerun": rec["id"]})
        try:
            exec_out = storage.read_stage(job_id, "execute") or {}
            plan_out = storage.read_stage(job_id, "plan") or {"plans": []}
            image = exec_out.get("image")
            if not image:
                raise RuntimeError("no environment image from the original analysis")
            options = json.loads(_job_options(job_id))
            with session() as s:
                claims = {c.id: c.model_dump() for c in s.exec(select(Claim).where(Claim.job_id == job_id)).all()}
                comps = {c.claim_id: c for c in s.exec(select(Comparison).where(Comparison.job_id == job_id)).all()}
            plans = {p["claim_id"]: p for p in plan_out["plans"]}
            values: dict[str, list[float]] = {cid: [] for cid in rec["claim_ids"]}
            log = lambda msg, **kw: bus.emit(job_id, msg, stage="compare", **kw)  # noqa: E731
            for k in range(1, rec["repeats"] + 1):
                run_id = rec["id"] if k == 1 else f"{rec['id']}.{k}"
                rec["run_ids"].append(run_id)
                got = _run_and_measure(job_id, run_id, "user_rerun", rec["argv"], rec["overrides"],
                                       rec["claim_ids"], claims, plans, image,
                                       int(options.get("timeout_s", 600)), False, log)
                for cid, v in got.items():
                    if v is not None:
                        values[cid].append(v)
                _save(job_id, rec)
            for cid in rec["claim_ids"]:
                comp = comps.get(cid)
                if comp is None or comp.obtained is None:
                    continue
                if not values[cid]:
                    rec["results"][cid] = {"values": [], "outcome": "inconclusive",
                                           "statement": "The rerun produced no value for this claim."}
                    continue
                after = statistics.fmean(values[cid])
                gap_after = after - comp.reported
                outcome, text = classify_hypothesis(change, comp.abs_delta, gap_after, comp.tolerance)
                rec["results"][cid] = {"values": [round(v, 6) for v in values[cid]], "obtained_after": round(after, 6),
                                       "obtained_before": comp.obtained, "gap_before": comp.abs_delta,
                                       "gap_after": round(gap_after, 6), "tolerance": comp.tolerance,
                                       "outcome": outcome, "statement": text}
            rec["status"] = "done"
        except Exception as exc:  # recorded, never hidden
            rec["status"], rec["error"] = "failed", f"{type(exc).__name__}: {exc}"
        rec["finished_at"] = utcnow()
        _save(job_id, rec)
        bus.emit(job_id, f"Rerun {rec['id']} {rec['status']}", stage="compare",
                 level="info" if rec["status"] == "done" else "warning", data={"rerun": rec["id"]})


def _job_options(job_id: str) -> str:
    from .db import Job
    with session() as s:
        job = s.get(Job, job_id)
    return job.options if job else "{}"
