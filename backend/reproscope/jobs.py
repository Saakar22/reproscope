"""Job creation and the background pipeline runner."""
from __future__ import annotations

import asyncio
import json
import shutil
import traceback
import uuid
from typing import Any, Optional

from sqlmodel import select

from . import storage
from .config import get_settings
from .db import Job, session, utcnow
from .events import TERMINAL_EVENT, bus
from .grounding import ground_text, normalise
from .schemas import STAGE_LABELS, STAGES, JobOptions, JobOut
from .stages import STAGE_FUNCS
from .stages.ingest import InputError

FINISHED_STATUSES = {"done", "failed", "incomplete"}
_tasks: dict[str, asyncio.Task] = {}


def new_job_id() -> str:
    return "j_" + uuid.uuid4().hex[:10]


def job_to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id, created_at=job.created_at, updated_at=job.updated_at,
        paper_title=job.paper_title, pdf_filename=job.pdf_filename, pdf_pages=job.pdf_pages,
        repo_source=job.repo_source, repo_url=job.repo_url, repo_filename=job.repo_filename,
        repo_commit=job.repo_commit, status=job.status, current_stage=job.current_stage,
        stage_detail=job.stage_detail, stages=json.loads(job.stages),
        options=JobOptions(**json.loads(job.options)), llm_mode=job.llm_mode, error=job.error)


def get_job(job_id: str) -> Optional[Job]:
    with session() as s:
        return s.get(Job, job_id)


def list_jobs(limit: int = 50) -> list[Job]:
    with session() as s:
        return list(s.exec(select(Job).order_by(Job.created_at.desc()).limit(limit)).all())


def update_job(job_id: str, **fields: Any) -> Job:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise KeyError(job_id)
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = utcnow()
        s.add(job)
        s.commit()
        s.refresh(job)
        return job


def set_stage(job_id: str, stage: str, status: str, detail: Optional[str] = None) -> None:
    job = get_job(job_id)
    stages = json.loads(job.stages)
    stages[stage] = status
    fields: dict[str, Any] = {"stages": json.dumps(stages)}
    if status == "running":
        fields.update(current_stage=stage, stage_detail=detail)
    update_job(job_id, **fields)


# ------------------------------------------------------------------ create

def create_job(*, pdf_bytes: bytes, pdf_filename: str, repo_url: Optional[str],
               zip_bytes: Optional[bytes], zip_filename: Optional[str],
               options: JobOptions) -> Job:
    """Persist uploads and the job row. Inputs must already be validated."""
    job_id = new_job_id()
    root = storage.job_dir(job_id)
    root.mkdir(parents=True)
    storage.paper_path(job_id).write_bytes(pdf_bytes)
    if zip_bytes is not None:
        storage.uploads_dir(job_id).mkdir(parents=True)
        (storage.uploads_dir(job_id) / "repo.zip").write_bytes(zip_bytes)
    job = Job(
        id=job_id, pdf_filename=pdf_filename,
        repo_source="git" if repo_url else "zip", repo_url=repo_url, repo_filename=zip_filename,
        stages=json.dumps({s: "pending" for s in STAGES}),
        options=options.model_dump_json(), llm_mode=get_settings().llm_mode)
    with session() as s:
        s.add(job)
        s.commit()
        s.refresh(job)
    bus.emit(job_id, "Job created", level="info",
             data={"repo_source": job.repo_source, "llm_mode": job.llm_mode})
    if job.llm_mode == "dev":
        bus.emit(job_id, "No LLM API key configured: LLM stages will use clearly labelled "
                 "development samples, not real analysis", level="warning")
    return job


def start_job(job_id: str) -> None:
    task = asyncio.create_task(run_pipeline(job_id), name=f"pipeline-{job_id}")
    _tasks[job_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(job_id, None))


def delete_job_files(job_id: str) -> None:
    shutil.rmtree(storage.job_dir(job_id), ignore_errors=True)


# ------------------------------------------------------------------ run

async def run_pipeline(job_id: str) -> None:
    update_job(job_id, status="running")
    bus.emit(job_id, "Pipeline started", level="info")
    job_dict = get_job(job_id).model_dump()
    missing: list[str] = []

    for stage in STAGES:
        fn = STAGE_FUNCS.get(stage)
        if fn is None:
            set_stage(job_id, stage, "not_implemented")
            missing.append(stage)
            continue
        set_stage(job_id, stage, "running")
        bus.emit(job_id, f"{STAGE_LABELS[stage]} started", stage=stage, level="stage",
                 data={"status": "running"})

        def log(message: str, *, level: str = "info", data: Any = None, _stage: str = stage) -> None:
            bus.emit(job_id, message, stage=_stage, level=level, data=data)

        try:
            output = await asyncio.to_thread(fn, job_id, job_dict, log)
        except InputError as exc:
            _fail(job_id, stage, str(exc))
            return
        except Exception as exc:  # unexpected bug: keep the traceback in the event log
            _fail(job_id, stage, f"{type(exc).__name__}: {exc}",
                  data={"traceback": traceback.format_exc(limit=8)})
            return

        storage.write_stage(job_id, stage, output)
        _apply_stage_output(job_id, stage, output)
        job_dict = get_job(job_id).model_dump()
        set_stage(job_id, stage, "done")
        bus.emit(job_id, f"{STAGE_LABELS[stage]} finished", stage=stage, level="stage",
                 data={"status": "done"})

    if missing:
        labels = ", ".join(STAGE_LABELS[s] for s in missing)
        update_job(job_id, status="incomplete", current_stage=None,
                   stage_detail=f"Not implemented yet: {labels}")
        bus.emit(job_id, f"Pipeline stopped: stages not implemented in this build ({labels})",
                 level="warning", data={"not_implemented": missing})
    else:
        update_job(job_id, status="done", current_stage=None, stage_detail=None)
        bus.emit(job_id, "Pipeline finished", level="info")
    bus.emit(job_id, TERMINAL_EVENT, level="info", data={"status": get_job(job_id).status})


def _fail(job_id: str, stage: str, message: str, data: Any = None) -> None:
    set_stage(job_id, stage, "failed")
    for later in STAGES[STAGES.index(stage) + 1:]:
        set_stage(job_id, later, "skipped")
    update_job(job_id, status="failed", error=message, current_stage=stage)
    bus.emit(job_id, f"{STAGE_LABELS[stage]} failed: {message}", stage=stage, level="error", data=data)
    bus.emit(job_id, TERMINAL_EVENT, level="info", data={"status": "failed"})


def _apply_stage_output(job_id: str, stage: str, output: dict[str, Any]) -> None:
    """Copy headline fields from a stage's output onto the job row."""
    if stage == "ingest":
        update_job(job_id, paper_title=output["pdf"]["title"], pdf_pages=output["pdf"]["pages"],
                   repo_commit=output["repo"]["commit"])
    elif stage == "extract" and output.get("origin") == "llm" and output.get("paper_title"):
        # Only trust the LLM's title if it actually appears on page 1.
        pages = storage.read_pages(job_id)
        if pages and ground_text(output["paper_title"], [normalise(pages[0])]):
            update_job(job_id, paper_title=output["paper_title"])


def recover_interrupted_jobs() -> int:
    """Jobs left 'running'/'queued' by a server restart can't resume; mark them failed."""
    count = 0
    with session() as s:
        for job in s.exec(select(Job).where(Job.status.in_(["running", "queued"]))).all():
            job.status = "failed"
            job.error = "Interrupted by a server restart; please start a new analysis."
            job.updated_at = utcnow()
            s.add(job)
            count += 1
        s.commit()
    return count
