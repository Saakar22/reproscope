"""Stage 2 — Claim extraction.

The LLM reads the paper's page text and proposes quantitative claims as schema-locked
JSON. Deterministic code then validates every claim (pydantic), verifies its quote and
number against the PDF text (grounding), keeps only hyperparameters whose wording is
actually in the paper, and persists the result. Ungrounded claims are kept but marked
unverified and are never scored.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlmodel import delete

from .. import llm, storage
from ..db import Claim, session
from ..grounding import ground_claim, ground_text, normalise
from .ingest import InputError

MAX_PROMPT_CHARS = 120_000   # ~30k tokens: well inside a 131k context, gentle on free-tier TPM

SYSTEM = """You extract quantitative experimental results from machine-learning papers.
Rules:
- Only results the authors report for their own experiments (skip related-work numbers).
- Copy every number exactly as printed. Never compute, round or convert numbers.
- `quote` must be verbatim text from the page: the sentence or the table row containing the number
  (for a table row, copy the row's label and the cell value as they appear).
- `page` is the PAGE number shown in the input where the quote appears.
- `unit`: "percent" if the number is a percentage (e.g. 91.2 meaning 91.2%), "fraction" if it is in
  [0,1] (e.g. 0.912), otherwise "raw".
- `hyperparameters`: only settings the paper explicitly states for this experiment, each with a
  verbatim `quote`. Never guess defaults. Use an empty list if none are stated.
- `reported_std`: only if the paper prints a ± / standard deviation for this value, else null.
- Use null when a field is not stated. Do not invent datasets, splits or settings."""

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["paper_title", "claims"],
    "properties": {
        "paper_title": {"type": ["string", "null"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["experiment", "dataset", "split", "metric", "higher_is_better",
                             "reported_value", "reported_std", "unit", "source", "page", "quote",
                             "hyperparameters", "preprocessing", "is_main_result"],
                "properties": {
                    "experiment": {"type": "string", "description": "method/model and setting"},
                    "dataset": {"type": ["string", "null"]},
                    "split": {"type": ["string", "null"]},
                    "metric": {"type": "string", "description": "e.g. accuracy, macro-F1, RMSE"},
                    "higher_is_better": {"type": "boolean"},
                    "reported_value": {"type": "number"},
                    "reported_std": {"type": ["number", "null"]},
                    "unit": {"type": "string", "enum": ["percent", "fraction", "raw"]},
                    "source": {"type": "string", "description": "e.g. 'Table 2' or 'Section 4 text'"},
                    "page": {"type": "integer"},
                    "quote": {"type": "string"},
                    "hyperparameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "value", "quote"],
                            "properties": {"name": {"type": "string"}, "value": {"type": "string"},
                                           "quote": {"type": "string"}},
                        },
                    },
                    "preprocessing": {"type": ["string", "null"]},
                    "is_main_result": {"type": "boolean"},
                },
            },
        },
    },
}


class HyperparamIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=600)


class ClaimIn(BaseModel):
    experiment: str = Field(min_length=1, max_length=300)
    dataset: Optional[str] = None
    split: Optional[str] = None
    metric: str = Field(min_length=1, max_length=80)
    higher_is_better: bool = True
    reported_value: float
    reported_std: Optional[float] = Field(default=None, ge=0)
    unit: str = Field(pattern="^(percent|fraction|raw)$")
    source: Optional[str] = None
    page: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=1500)
    hyperparameters: list[HyperparamIn] = []
    preprocessing: Optional[str] = None
    is_main_result: bool = False

    @field_validator("metric")
    @classmethod
    def _clean_metric(cls, v: str) -> str:
        return v.strip()


# ------------------------------------------------------------------ helpers

def build_prompt(pages: list[str], budget: int = MAX_PROMPT_CHARS) -> tuple[str, list[int]]:
    """Paper text with explicit page markers. If too long, keep the result-heavy pages."""
    chunks = [f"=== PAGE {i + 1} ===\n{text.strip()}" for i, text in enumerate(pages)]
    if sum(len(c) for c in chunks) <= budget:
        return "\n\n".join(chunks), []

    def score(i: int) -> float:
        t = pages[i]
        decimals = len(re.findall(r"\d+\.\d+", t))
        bonus = 30 if re.search(r"\b(table|results?|experiments?|evaluation)\b", t, re.I) else 0
        return decimals + bonus - (60 if re.search(r"^\s*references\s*$", t, re.I | re.M) else 0)

    keep, used = set(), 0
    for i in sorted(range(len(pages)), key=lambda i: (-score(i), i)):
        if used + len(chunks[i]) <= budget:
            keep.add(i)
            used += len(chunks[i])
    skipped = [i + 1 for i in range(len(pages)) if i not in keep]
    return "\n\n".join(chunks[i] for i in sorted(keep)), skipped


METRIC_WORDS = r"(accuracy|acc\.?|f1(?:[- ]score)?|macro[- ]f1|micro[- ]f1|auc|roc[- ]auc|precision|recall|rmse|mae|mse|bleu|rouge[- ]?\w*|perplexity|error rate)"


def dev_sample(pages: list[str]) -> dict[str, Any]:
    """No-LLM fallback: a deterministic regex scan for "<metric> ... <number>" sentences.
    Always labelled 'dev_sample' downstream; it is a heuristic, not an analysis."""
    claims: list[dict[str, Any]] = []
    pattern = re.compile(rf"{METRIC_WORDS}[^.\n]{{0,60}}?(\d{{1,3}}\.\d{{1,4}})\s*(%?)", re.I)
    for pno, text in enumerate(pages, 1):
        for line in text.splitlines():
            m = pattern.search(line)
            if not m:
                continue
            value = float(m.group(2))
            metric = m.group(1).lower().rstrip(".")
            claims.append({
                "experiment": line.strip()[:80], "dataset": None, "split": None,
                "metric": "accuracy" if metric.startswith("acc") else metric,
                "higher_is_better": metric not in ("rmse", "mae", "mse", "perplexity", "error rate"),
                "reported_value": value, "reported_std": None,
                "unit": "percent" if (m.group(3) or value > 1) else "fraction",
                "source": "regex scan (dev mode)", "page": pno, "quote": line.strip(),
                "hyperparameters": [], "preprocessing": None, "is_main_result": False,
            })
            if len(claims) >= 10:
                return {"paper_title": None, "claims": claims}
    return {"paper_title": None, "claims": claims}


def _dedupe(claims: list[ClaimIn]) -> list[ClaimIn]:
    seen, out = set(), []
    for c in claims:
        key = (c.experiment.lower(), c.metric.lower(), round(c.reported_value, 6), (c.dataset or "").lower())
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# -------------------------------------------------------------------- stage

def run(job_id: str, job: dict[str, Any], log: Callable[..., None]) -> dict[str, Any]:
    options = json.loads(job.get("options") or "{}")
    max_claims = int(options.get("max_claims", 6))
    main_only = bool(options.get("main_results_only", False))

    pages = storage.read_pages(job_id)
    pages_norm = [normalise(p) for p in pages]
    prompt, skipped = build_prompt(pages)
    if skipped:
        log(f"Paper is long; sending result-heavy pages only (skipped pages {skipped})", level="warning")

    log("Asking the LLM for quantitative claims" if job["llm_mode"] == "live"
        else "Dev mode: scanning text with a regex heuristic (no LLM)",
        data={"prompt_chars": len(prompt)})
    try:
        result = llm.complete_json(name="paper_claims", schema=CLAIM_SCHEMA, system=SYSTEM, user=prompt,
                                   dev_sample=lambda: dev_sample(pages), max_tokens=16000)
    except llm.LLMError as exc:
        raise InputError(str(exc)) from exc
    origin = result.origin
    if result.cached:
        log("Using cached LLM response for identical input")

    raw_claims = result.data.get("claims", [])
    parsed: list[ClaimIn] = []
    rejected = 0
    for item in raw_claims:
        try:
            parsed.append(ClaimIn.model_validate(item))
        except ValidationError:
            rejected += 1
    if rejected:
        log(f"Discarded {rejected} malformed claim(s) that failed schema validation", level="warning")
    parsed = _dedupe(parsed)
    if main_only:
        parsed = [c for c in parsed if c.is_main_result] or parsed

    # Ground everything, then prefer grounded main results when trimming to max_claims.
    grounded_rows = []
    for c in parsed:
        g = ground_claim(c.quote, c.reported_value, c.page, pages_norm)
        hps = [{"name": h.name, "value": h.value, "quote": h.quote, "verified": ground_text(h.quote, pages_norm)}
               for h in c.hyperparameters]
        grounded_rows.append((c, g, hps))
    grounded_rows.sort(key=lambda row: (not row[1]["grounded"], not row[0].is_main_result))
    if len(grounded_rows) > max_claims:
        log(f"{len(grounded_rows)} claims found; keeping {max_claims} (max claims setting)")
        grounded_rows = grounded_rows[:max_claims]

    claims_out: list[dict[str, Any]] = []
    corrected_pages = 0
    with session() as s:
        s.exec(delete(Claim).where(Claim.job_id == job_id))
        for i, (c, g, hps) in enumerate(grounded_rows, 1):
            cid = f"C{i}"
            # The verified page wins over the LLM's citation; keep the LLM's for transparency.
            g["llm_page"] = c.page
            if g["grounded"] and g["page_found"] and g["page_found"] != c.page:
                c = c.model_copy(update={"page": g["page_found"]})
                corrected_pages += 1
            row = Claim(
                job_id=job_id, id=cid, experiment=c.experiment, dataset=c.dataset, split=c.split,
                metric=c.metric, higher_is_better=c.higher_is_better, unit=c.unit,
                reported_value=c.reported_value, reported_std=c.reported_std, source=c.source,
                page=c.page, quote=c.quote, hyperparameters=json.dumps(hps),
                preprocessing=c.preprocessing, is_main_result=c.is_main_result,
                grounded=g["grounded"], grounding=json.dumps(g), origin=origin)
            s.add(row)
            claims_out.append({"id": cid, **c.model_dump(), "hyperparameters": hps, "grounding": g,
                               "grounded": g["grounded"], "origin": origin})
        s.commit()

    n_grounded = sum(1 for c in claims_out if c["grounded"])
    dropped_hp = sum(1 for c in claims_out for h in c["hyperparameters"] if not h["verified"])
    log(f"{len(claims_out)} claim(s) extracted, {n_grounded} grounded in the PDF text"
        + (f"; {len(claims_out) - n_grounded} unverified (excluded from scoring)" if n_grounded < len(claims_out) else ""),
        data={"claims": [{"id": c["id"], "metric": c["metric"], "value": c["reported_value"],
                          "page": c["page"], "grounded": c["grounded"]} for c in claims_out]})
    if corrected_pages:
        log(f"Corrected the page number of {corrected_pages} claim(s) to where the text was actually found")
    if dropped_hp:
        log(f"{dropped_hp} hyperparameter(s) not found verbatim in the paper; not treated as paper-stated",
            level="warning")
    if not claims_out:
        log("No quantitative claims found in the paper", level="warning")

    return {"origin": origin, "model": result.model, "cached": result.cached, "usage": result.usage,
            "paper_title": result.data.get("paper_title"), "skipped_pages": skipped,
            "rejected_malformed": rejected, "claims": claims_out}
