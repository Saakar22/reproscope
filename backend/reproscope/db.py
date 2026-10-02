"""SQLite persistence (SQLModel).

Large blobs (PDFs, repository copies, logs, per-stage JSON) live in the per-job
artifact directory; these tables hold the queryable rows the UI and report need.
JSON-valued columns are stored as TEXT and (de)serialised by the stage modules.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator, Optional

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from .config import get_settings


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ tables

class Job(SQLModel, table=True):
    __tablename__ = "jobs"
    id: str = Field(primary_key=True)
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    paper_title: Optional[str] = None
    pdf_filename: Optional[str] = None
    pdf_pages: Optional[int] = None
    repo_source: str                      # "git" | "zip"
    repo_url: Optional[str] = None
    repo_filename: Optional[str] = None
    repo_commit: Optional[str] = None
    status: str = "queued"               # queued | running | done | failed
    current_stage: Optional[str] = None
    stage_detail: Optional[str] = None
    stages: str = "{}"                   # JSON {stage: pending|running|done|failed|skipped}
    options: str = "{}"                  # JSON analysis options
    llm_mode: str = "dev"                # live | dev  (dev = labelled sample outputs)
    error: Optional[str] = None


class Claim(SQLModel, table=True):
    __tablename__ = "claims"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    id: str = Field(primary_key=True)    # C1, C2 ...
    experiment: str
    dataset: Optional[str] = None
    split: Optional[str] = None
    metric: str
    higher_is_better: bool = True
    unit: str = "raw"                    # percent | fraction | raw
    reported_value: float
    reported_std: Optional[float] = None
    source: Optional[str] = None
    page: int
    quote: str
    hyperparameters: str = "{}"          # JSON, paper-stated only
    preprocessing: Optional[str] = None
    is_main_result: bool = False
    grounded: bool = False
    grounding: str = "{}"                # JSON {quote_score, number_found, page_found}
    origin: str = "llm"                  # llm | dev_sample | user_edit


class RepoFact(SQLModel, table=True):
    __tablename__ = "repo_facts"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    id: str = Field(primary_key=True)    # F1, F2 ...
    kind: str                            # entrypoint | arg | config_value | requirement | seed_call | ...
    name: str
    value: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    extra: str = "{}"


class RunPlan(SQLModel, table=True):
    __tablename__ = "run_plans"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    id: str = Field(primary_key=True)    # P1 ...
    claim_id: str
    script: str
    command: str
    source_of_command: Optional[str] = None
    prefetch: Optional[str] = None
    expected_metric_location: Optional[str] = None
    paper_overrides: str = "{}"
    estimated_minutes: Optional[float] = None
    confidence: Optional[str] = None
    validation: str = "{}"               # JSON {ok, errors[]}


class Run(SQLModel, table=True):
    __tablename__ = "runs"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    id: str = Field(primary_key=True)    # R1 ...
    plan_id: str
    claim_id: str
    kind: str = "primary"                # primary | seed | hypothesis
    command: str
    overrides: str = "{}"
    seed: Optional[int] = None
    status: str = "pending"              # pending | running | ok | error | timeout | not_runnable
    exit_code: Optional[int] = None
    seconds: Optional[float] = None
    log_path: Optional[str] = None
    image_tag: Optional[str] = None
    environment: str = "{}"
    started_at: Optional[str] = None


class Result(SQLModel, table=True):
    __tablename__ = "results"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    run_id: str = Field(primary_key=True)
    metric: str = Field(primary_key=True)
    claim_id: str
    value: float
    raw_value: float
    source: str                          # file | regex | llm | manual
    evidence: str = "{}"                 # JSON {file|log_path, line_no, line_text}


class Comparison(SQLModel, table=True):
    __tablename__ = "comparisons"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    claim_id: str = Field(primary_key=True)
    reported: float
    obtained: Optional[float] = None
    abs_delta: Optional[float] = None
    rel_delta: Optional[float] = None
    tolerance: Optional[float] = None
    tolerance_basis: Optional[str] = None
    seed_values: str = "[]"
    verdict: str
    better_than_reported: bool = False
    run_status: Optional[str] = None
    note: Optional[str] = None


class Finding(SQLModel, table=True):
    __tablename__ = "findings"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    id: str = Field(primary_key=True)    # E1 ...
    rule: str
    severity: str                        # high | medium | low
    claim_id: Optional[str] = None       # primary claim (first of claim_ids)
    claim_ids: str = "[]"                # JSON: every claim the finding applies to
    summary: str
    impact: Optional[str] = None
    paper_evidence: str = "null"         # JSON {page, quote}
    repo_evidence: str = "[]"            # JSON [{file, line, snippet}]
    run_evidence: str = "null"           # JSON {run_id, line_no, line_text}
    test: str = "null"                   # JSON hypothesis-rerun outcome


class Deviation(SQLModel, table=True):
    __tablename__ = "deviations"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    id: str = Field(primary_key=True)    # D1 ...
    run_id: Optional[str] = None
    kind: str
    detail: str


class Diagnosis(SQLModel, table=True):
    __tablename__ = "diagnoses"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    claim_id: str = Field(primary_key=True)
    hypotheses: str = "[]"               # JSON [{text, evidence_ids[], likelihood, suggested_check}]
    origin: str = "llm"                  # llm | dev_sample


class PipelineEvent(SQLModel, table=True):
    __tablename__ = "events"
    job_id: str = Field(primary_key=True, foreign_key="jobs.id")
    seq: int = Field(primary_key=True)
    ts: str = Field(default_factory=utcnow)
    stage: Optional[str] = None
    level: str = "info"                  # info | warning | error | stage | log
    message: str
    data: str = "null"                   # JSON payload for the UI


# ------------------------------------------------------------------ engine

_engine: Engine | None = None


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{settings.db_path.as_posix()}",
                                connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(_engine)
        _add_missing_columns(_engine)
    return _engine


def _add_missing_columns(engine: Engine) -> None:
    """Minimal additive migration: add columns introduced after a database was created."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                default = col.default.arg if col.default is not None and not callable(col.default.arg) else None
                ddl = f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {col.type.compile(engine.dialect)}'
                if default is not None:
                    ddl += f" DEFAULT '{default}'" if isinstance(default, str) else f" DEFAULT {default}"
                conn.execute(text(ddl))


def reset_engine() -> None:
    """Drop the cached engine (tests switch data directories)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def session() -> Session:
    return Session(get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session() as s:
        yield s
