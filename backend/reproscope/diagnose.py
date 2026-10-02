"""Evidence-constrained explanations for claims that did not reproduce.

The LLM receives, per claim, the comparison and a numbered list of evidence (findings,
hypothesis-rerun outcomes, deviations). Every explanation must cite evidence ids from
that list; uncited or mis-cited explanations are discarded. "Unexplained" is valid.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from . import llm

SYSTEM = """You explain why reproduced ML results differ from a paper, using ONLY the evidence given.
Rules:
- Each hypothesis must cite one or more evidence ids from that claim's list. Never cite anything else.
- Do not claim causation beyond the evidence. A hypothesis-rerun result that moved the gap within
  tolerance SUPPORTS a cause; it does not prove it. Say "may", "is consistent with", "supports".
- If the evidence does not explain the gap, set unexplained=true and give no hypotheses.
- likelihood: high only when a rerun supports it; medium for a direct configuration/metric/data
  difference; low for general reproducibility issues (seeds, unpinned dependencies).
- suggested_check: one concrete next step, or null."""

SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["diagnoses"],
    "properties": {"diagnoses": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["claim_id", "unexplained", "hypotheses"],
        "properties": {
            "claim_id": {"type": "string"},
            "unexplained": {"type": "boolean"},
            "hypotheses": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["text", "evidence_ids", "likelihood", "suggested_check"],
                "properties": {"text": {"type": "string"},
                               "evidence_ids": {"type": "array", "items": {"type": "string"}},
                               "likelihood": {"type": "string", "enum": ["high", "medium", "low"]},
                               "suggested_check": {"type": ["string", "null"]}}}}}}}},
}

RULE_LIKELIHOOD = {"PARAM_MISMATCH": "medium", "SPLIT_MISMATCH": "medium", "METRIC_DEFINITION": "medium",
                   "GPU_ONLY": "high", "REDUCED_BUDGET": "high", "MISSING_ARTEFACT": "high",
                   "NONDETERMINISM": "low", "SEED_NOT_SET": "low", "DEPS_UNPINNED": "low",
                   "MISSING_DEPENDENCY": "low", "EVAL_LEAKAGE": "medium", "PARAM_UNSTATED": "low"}
MAX_HYPOTHESES = 4


def dev_sample(targets: list[dict[str, Any]]) -> dict[str, Any]:
    """No-LLM fallback: list the claim's own evidence in a fixed order (labelled dev_sample)."""
    out = []
    for t in targets:
        hyps = []
        for e in t["evidence"]:
            if e["kind"] != "finding":
                continue
            supported = (e.get("test") or {}).get("supports_hypothesis") == "supports"
            hyps.append({"text": e["summary"] + (f" {e['test']['statement']}" if e.get("test") else ""),
                         "evidence_ids": [e["id"]],
                         "likelihood": "high" if supported else RULE_LIKELIHOOD.get(e["rule"], "low"),
                         "suggested_check": None})
        order = {"high": 0, "medium": 1, "low": 2}
        hyps.sort(key=lambda h: order[h["likelihood"]])
        out.append({"claim_id": t["claim_id"], "unexplained": not hyps, "hypotheses": hyps[:MAX_HYPOTHESES]})
    return {"diagnoses": out}


def diagnose(targets: list[dict[str, Any]], log: Callable[..., None]) -> dict[str, dict[str, Any]]:
    """targets: [{claim_id, claim, comparison, evidence: [{id, kind, rule?, summary, test?}]}]."""
    if not targets:
        return {}
    payload = [{"claim_id": t["claim_id"],
                "claim": {k: t["claim"].get(k) for k in ("experiment", "dataset", "metric", "reported_value",
                                                          "reported_std", "unit")},
                "comparison": {k: t["comparison"].get(k) for k in ("verdict", "obtained", "abs_delta",
                                                                   "tolerance", "tolerance_basis", "note")},
                "evidence": t["evidence"]} for t in targets]
    try:
        res = llm.complete_json(name="diagnoses", schema=SCHEMA, system=SYSTEM,
                                user=json.dumps(payload, indent=1, default=str),
                                dev_sample=lambda: dev_sample(targets), max_tokens=8000)
    except llm.LLMError as exc:
        log(f"Explanations unavailable: {exc}", level="warning")
        return {t["claim_id"]: {"hypotheses": [], "origin": "error", "unexplained": True,
                                "dropped": 0, "error": str(exc)} for t in targets}

    allowed = {t["claim_id"]: {e["id"] for e in t["evidence"]} for t in targets}
    out: dict[str, dict[str, Any]] = {}
    dropped_total = 0
    for d in res.data.get("diagnoses", []):
        cid = d.get("claim_id")
        if cid not in allowed:
            continue
        kept, dropped = [], 0
        for h in d.get("hypotheses", []):
            ids = [i for i in h.get("evidence_ids", []) if i in allowed[cid]]
            if not ids or len(ids) != len(h.get("evidence_ids", [])) or not h.get("text", "").strip():
                dropped += 1
                continue
            kept.append({**h, "evidence_ids": ids})
        dropped_total += dropped
        out[cid] = {"hypotheses": kept[:MAX_HYPOTHESES], "origin": res.origin,
                    "unexplained": bool(d.get("unexplained")) or not kept, "dropped": dropped}
    for t in targets:
        out.setdefault(t["claim_id"], {"hypotheses": [], "origin": res.origin, "unexplained": True, "dropped": 0})
    if dropped_total:
        log(f"Discarded {dropped_total} explanation(s) that cited no or unknown evidence", level="warning")
    return out
