"""Typed API contracts. `frontend/src/types.ts` mirrors these — change both together."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

STAGES: list[str] = ["ingest", "extract", "scan", "plan", "execute", "parse", "compare"]
STAGE_LABELS: dict[str, str] = {
    "ingest": "Ingest",
    "extract": "Claim Extraction",
    "scan": "Repository Scan",
    "plan": "Run Planning",
    "execute": "Experiment Execution",
    "parse": "Result Parsing",
    "compare": "Comparison & Diagnosis",
}
StageStatus = Literal["pending", "running", "done", "failed", "skipped", "not_implemented"]

Verdict = Literal["REPRODUCED", "PARTIAL", "NOT_REPRODUCED", "INCONCLUSIVE",
                  "NOT_RUN", "NO_METRIC", "UNVERIFIED_CLAIM"]


class JobOptions(BaseModel):
    main_results_only: bool = False
    max_claims: int = Field(default=6, ge=1, le=20)
    enable_hypothesis_reruns: bool = True
    timeout_s: int = Field(default=600, ge=30, le=3600)


class JobOut(BaseModel):
    id: str
    created_at: str
    updated_at: str
    paper_title: Optional[str]
    pdf_filename: Optional[str]
    pdf_pages: Optional[int]
    repo_source: str
    repo_url: Optional[str]
    repo_filename: Optional[str]
    repo_commit: Optional[str]
    status: str
    current_stage: Optional[str]
    stage_detail: Optional[str]
    stages: dict[str, str]
    options: JobOptions
    llm_mode: str
    error: Optional[str]


class EventOut(BaseModel):
    job_id: str
    seq: int
    ts: str
    stage: Optional[str]
    level: str
    message: str
    data: Any = None


# ------------------------------------------------------------- report

class Grounding(BaseModel):
    quote_score: Optional[float] = None
    number_found: Optional[bool] = None
    page_found: Optional[int] = None


class ComparisonOut(BaseModel):
    reported: float
    obtained: Optional[float]
    abs_delta: Optional[float]
    rel_delta: Optional[float]
    tolerance: Optional[float]
    tolerance_basis: Optional[str]
    seed_values: list[float] = []
    verdict: Verdict
    better_than_reported: bool = False
    run_status: Optional[str] = None
    note: Optional[str] = None


class Hypothesis(BaseModel):
    text: str
    evidence_ids: list[str]
    likelihood: Literal["high", "medium", "low"]
    suggested_check: Optional[str] = None


class ClaimReport(BaseModel):
    id: str
    experiment: str
    dataset: Optional[str]
    split: Optional[str]
    metric: str
    unit: str
    higher_is_better: bool
    reported_value: float
    reported_std: Optional[float]
    source: Optional[str]
    page: int
    quote: str
    hyperparameters: dict[str, Any]
    is_main_result: bool
    grounded: bool
    grounding: Grounding
    origin: str
    comparison: Optional[ComparisonOut]
    run_ids: list[str] = []
    finding_ids: list[str] = []
    deviation_ids: list[str] = []
    hypotheses: list[Hypothesis] = []
    diagnosis_origin: Optional[str] = None


class RepoEvidence(BaseModel):
    file: str
    line: Optional[int] = None
    snippet: Optional[str] = None


class FindingOut(BaseModel):
    id: str
    rule: str
    severity: Literal["high", "medium", "low"]
    claim_id: Optional[str]
    claim_ids: list[str] = []
    summary: str
    impact: Optional[str]
    paper_evidence: Optional[dict[str, Any]]
    repo_evidence: list[RepoEvidence]
    run_evidence: Optional[dict[str, Any]]
    test: Optional[dict[str, Any]]


class DeviationOut(BaseModel):
    id: str
    run_id: Optional[str]
    kind: str
    detail: str


class RunOut(BaseModel):
    id: str
    plan_id: str
    claim_id: str
    kind: str
    command: str
    overrides: dict[str, Any]
    status: str
    exit_code: Optional[int]
    seconds: Optional[float]
    image_tag: Optional[str]
    environment: dict[str, Any]
    log_available: bool
    metric_evidence: Optional[dict[str, Any]] = None


class PlanOut(BaseModel):
    id: str
    claim_id: str
    script: str
    command: str
    source_of_command: Optional[str]
    expected_metric_location: Optional[str]
    estimated_minutes: Optional[float]
    confidence: Optional[str]
    paper_overrides: dict[str, Any]
    validation: dict[str, Any]


class RepoFactOut(BaseModel):
    id: str
    kind: str
    name: str
    value: Optional[str]
    file: Optional[str]
    line: Optional[int]


class ReportSummary(BaseModel):
    total_claims: int
    grounded_claims: int
    reproduced: int
    partial: int
    not_reproduced: int
    inconclusive: int
    not_run: int
    no_metric: int
    unverified: int
    better_than_reported: int


class Checklist(BaseModel):
    seed_set: Optional[bool]
    deps_pinned: Optional[bool]
    all_hparams_stated: Optional[bool]
    split_stated: Optional[bool]


class ReportJob(BaseModel):
    id: str
    paper_title: Optional[str]
    pdf_filename: Optional[str]
    repo_source: str
    repo_url: Optional[str]
    repo_filename: Optional[str]
    repo_commit: Optional[str]
    created_at: str
    status: str
    llm_mode: str


class Report(BaseModel):
    is_fixture: bool = False
    fixture_note: Optional[str] = None
    generated_at: str
    job: ReportJob
    summary: ReportSummary
    checklist: Checklist
    claims: list[ClaimReport]
    findings: list[FindingOut]
    deviations: list[DeviationOut]
    runs: list[RunOut]
    plans: list[PlanOut]
    repo_facts: list[RepoFactOut]
    tolerance_policy: dict[str, str]
    reruns: list[dict[str, Any]] = []          # user-approved hypothesis reruns


class ErrorOut(BaseModel):
    detail: str
