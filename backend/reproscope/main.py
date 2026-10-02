"""ReproScope FastAPI application."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import ValidationError

from . import jobs, llm, reruns, storage
from .reruns import RerunRequest
from .config import get_settings
from sqlmodel import select

from .db import Run, get_engine
from .db import session as db_session
from .events import TERMINAL_EVENT, bus, sse_format
from .report import build_report, export_filename, render_markdown
from .schemas import STAGE_LABELS, STAGES, EventOut, JobOptions, JobOut, Report
from .stages.ingest import InputError, validate_git_url, validate_pdf, validate_zip

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_report.json"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_engine()
    jobs.recover_interrupted_jobs()
    yield


app = FastAPI(title="ReproScope API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(get_settings().cors_origins),
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ helpers

async def _read_limited(upload: UploadFile, limit_mb: int, what: str) -> bytes:
    limit = limit_mb * 1024 * 1024
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"{what} exceeds the {limit_mb} MB limit.")
    return data


def _require_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found.")
    return job


_docker_cache: dict[str, object] = {"at": 0.0, "value": None}


def _docker_status() -> dict[str, object]:
    """Is a Docker engine reachable? Cached for 10 s."""
    if time.monotonic() - float(_docker_cache["at"]) < 10 and _docker_cache["value"] is not None:
        return _docker_cache["value"]  # type: ignore[return-value]
    try:
        import docker
        client = docker.from_env(timeout=3)
        client.ping()
        value = {"available": True, "version": client.version().get("Version")}
    except Exception as exc:
        value = {"available": False, "error": str(exc).splitlines()[0][:200]}
    _docker_cache.update(at=time.monotonic(), value=value)
    return value


# ------------------------------------------------------------------ routes

@app.get("/api/health")
async def health() -> dict:
    settings = get_settings()
    docker_info = await asyncio.to_thread(_docker_status)
    return {"status": "ok", "llm_mode": settings.llm_mode, "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "docker": docker_info,
            "stages": [{"id": s, "label": STAGE_LABELS[s]} for s in STAGES],
            "limits": {"max_pdf_mb": settings.max_pdf_mb, "max_zip_mb": settings.max_zip_mb}}


@app.post("/api/llm/check")
async def llm_check() -> dict:
    """Make one tiny live call to confirm the key, model and structured output work."""
    return await asyncio.to_thread(llm.ping)


@app.post("/api/jobs", response_model=JobOut, status_code=201)
async def create_job(
    pdf: UploadFile = File(..., description="Paper PDF"),
    repo_url: Optional[str] = Form(None),
    repo_zip: Optional[UploadFile] = File(None),
    options: Optional[str] = Form(None, description="JSON-encoded JobOptions"),
) -> JobOut:
    settings = get_settings()
    has_url = bool(repo_url and repo_url.strip())
    has_zip = repo_zip is not None and bool(repo_zip.filename)
    if has_url == has_zip:
        raise HTTPException(422, "Provide exactly one of repo_url or repo_zip.")
    try:
        opts = JobOptions(**json.loads(options)) if options else JobOptions()
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(422, f"Invalid options: {exc}") from exc

    pdf_bytes = await _read_limited(pdf, settings.max_pdf_mb, "The PDF")
    zip_bytes = await _read_limited(repo_zip, settings.max_zip_mb, "The ZIP") if has_zip else None
    try:
        await asyncio.to_thread(validate_pdf, pdf_bytes, pdf.filename)
        url = validate_git_url(repo_url) if has_url else None
        if zip_bytes is not None:
            await asyncio.to_thread(validate_zip, zip_bytes, repo_zip.filename)
    except InputError as exc:
        raise HTTPException(400, str(exc)) from exc

    job = jobs.create_job(pdf_bytes=pdf_bytes, pdf_filename=pdf.filename or "paper.pdf",
                          repo_url=url, zip_bytes=zip_bytes,
                          zip_filename=repo_zip.filename if has_zip else None, options=opts)
    jobs.start_job(job.id)
    return jobs.job_to_out(job)


@app.get("/api/jobs", response_model=list[JobOut])
async def list_jobs(limit: int = Query(50, ge=1, le=200)) -> list[JobOut]:
    return [jobs.job_to_out(j) for j in jobs.list_jobs(limit)]


@app.get("/api/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str) -> JobOut:
    return jobs.job_to_out(_require_job(job_id))


@app.get("/api/jobs/{job_id}/stages/{stage}")
async def get_stage_output(job_id: str, stage: str) -> dict:
    """Structured output of one pipeline stage (the persisted JSON)."""
    _require_job(job_id)
    if stage not in STAGES:
        raise HTTPException(404, f"Unknown stage {stage!r}.")
    data = storage.read_stage(job_id, stage)
    if data is None:
        raise HTTPException(404, f"Stage {stage!r} has no output for this job yet.")
    return data


@app.get("/api/jobs/{job_id}/events/history", response_model=list[EventOut])
async def event_history(job_id: str, after: int = Query(0, ge=0)) -> list[EventOut]:
    _require_job(job_id)
    return bus.history(job_id, after)


@app.get("/api/jobs/{job_id}/events")
async def stream_events(request: Request, job_id: str, after: int = Query(0, ge=0),
                        replay: bool = False, speed: float = Query(4.0, gt=0, le=100)):
    """Server-Sent Events. Live mode streams history then follows the job;
    replay mode re-streams a finished job's stored events with scaled timing."""
    job = _require_job(job_id)
    last_event_id = request.headers.get("last-event-id")
    if last_event_id and last_event_id.isdigit():
        after = max(after, int(last_event_id))
    if replay and job.status not in jobs.FINISHED_STATUSES:
        raise HTTPException(409, "Only finished jobs can be replayed.")
    gen = _replay_stream(job_id, speed) if replay else _live_stream(request, job_id, after)
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _live_stream(request: Request, job_id: str, after: int) -> AsyncIterator[str]:
    queue = bus.subscribe(job_id)
    try:
        last = after
        backlog = bus.history(job_id, after)
        for event in backlog:
            yield sse_format(event)
            last = event.seq
        if any(e.message == TERMINAL_EVENT for e in backlog) or \
                jobs.get_job(job_id).status in jobs.FINISHED_STATUSES:
            return
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if event.seq <= last:
                continue
            yield sse_format(event)
            last = event.seq
            if event.message == TERMINAL_EVENT:
                return
    finally:
        bus.unsubscribe(job_id, queue)


async def _replay_stream(job_id: str, speed: float) -> AsyncIterator[str]:
    events = bus.history(job_id, 0)
    previous: Optional[datetime] = None
    for event in events:
        ts = datetime.fromisoformat(event.ts)
        if previous is not None:
            await asyncio.sleep(min(max((ts - previous).total_seconds(), 0) / speed, 1.5))
        previous = ts
        yield sse_format(event.model_copy(update={"data": {"replay": True, **(event.data or {})}
                                                  if isinstance(event.data, dict) or event.data is None
                                                  else event.data}))


@app.post("/api/jobs/{job_id}/replay")
async def replay(job_id: str, speed: float = Query(4.0, gt=0, le=100)) -> dict:
    job = _require_job(job_id)
    if job.status not in jobs.FINISHED_STATUSES:
        raise HTTPException(409, "Only finished jobs can be replayed.")
    return {"job_id": job_id, "events_url": f"/api/jobs/{job_id}/events?replay=true&speed={speed}",
            "event_count": len(bus.history(job_id, 0))}


MAX_VIEW_BYTES = 400_000
LANGUAGES = {".py": "python", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml", ".md": "markdown",
             ".txt": "text", ".cfg": "ini", ".ini": "ini", ".sh": "bash", ".rst": "text"}


@app.get("/api/jobs/{job_id}/file")
async def get_repo_file(job_id: str, path: str = Query(..., min_length=1, max_length=500)) -> dict:
    """A text file from the analysed repository (read-only), for inspecting evidence."""
    _require_job(job_id)
    root = storage.repo_dir(job_id)
    try:
        target = storage.safe_join(root, path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not target.is_file():
        raise HTTPException(404, f"{path} is not a file in the repository.")
    data = target.read_bytes()[:MAX_VIEW_BYTES + 1]
    if b"\0" in data[:8000]:
        raise HTTPException(415, f"{path} looks like a binary file.")
    text = data[:MAX_VIEW_BYTES].decode("utf-8", errors="replace")
    return {"path": target.relative_to(root.resolve()).as_posix(), "language": LANGUAGES.get(target.suffix.lower(), "text"),
            "lines": text.splitlines(), "truncated": len(data) > MAX_VIEW_BYTES}


@app.post("/api/jobs/{job_id}/rerun", status_code=202)
async def request_rerun(job_id: str, body: RerunRequest) -> dict:
    """Start a user-approved hypothesis rerun: the claim's command with up to 3 validated overrides."""
    job = _require_job(job_id)
    if job.status != "done":
        raise HTTPException(409, "Reruns are available once the analysis has finished.")
    try:
        return await asyncio.to_thread(reruns.start, job_id, body)
    except reruns.RerunError as exc:
        raise HTTPException(409 if "already running" in str(exc) else 400, str(exc)) from exc


@app.get("/api/jobs/{job_id}/reruns")
async def get_reruns(job_id: str) -> list[dict]:
    _require_job(job_id)
    return reruns.list_reruns(job_id)


@app.get("/api/jobs/{job_id}/logs")
async def list_logs(job_id: str) -> list[dict]:
    """Every experiment run of the job, with its command, status and log URL."""
    _require_job(job_id)
    with db_session() as s:
        rows = s.exec(select(Run).where(Run.job_id == job_id).order_by(Run.id)).all()
    return [{"run_id": r.id, "kind": r.kind, "claim_ids": json.loads(r.environment).get("claim_ids", [r.claim_id]),
             "command": r.command, "status": r.status, "exit_code": r.exit_code, "seconds": r.seconds,
             "log_url": f"/api/jobs/{job_id}/runs/{r.id}/log" if r.log_path else None} for r in rows]


@app.get("/api/jobs/{job_id}/runs/{run_id}/log")
async def get_run_log(job_id: str, run_id: str) -> Response:
    _require_job(job_id)
    with db_session() as s:
        row = s.get(Run, (job_id, run_id))
    if row is None or not row.log_path:
        raise HTTPException(404, f"No log stored for run {run_id}.")
    try:
        path = storage.safe_join(storage.job_dir(job_id), row.log_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, "Log file missing.")
    return Response(path.read_text(encoding="utf-8", errors="replace"), media_type="text/plain; charset=utf-8")


@app.get("/api/jobs/{job_id}/pdf")
async def get_pdf(job_id: str) -> FileResponse:
    _require_job(job_id)
    path = storage.paper_path(job_id)
    if not path.is_file():
        raise HTTPException(404, "PDF not found.")
    return FileResponse(path, media_type="application/pdf")


def _job_report(job_id: str) -> Report:
    job = _require_job(job_id)
    if job.status != "done" or json.loads(job.stages).get("compare") != "done":
        raise HTTPException(409, f"No report yet: job status is '{job.status}'"
                                 + (f" ({job.error})" if job.error else "") + ".")
    return build_report(job_id)


@app.get("/api/jobs/{job_id}/report", response_model=Report)
async def get_report(job_id: str) -> Report:
    return await asyncio.to_thread(_job_report, job_id)


@app.get("/api/jobs/{job_id}/export")
async def export_report(job_id: str, format: Literal["markdown", "json"] = "markdown") -> Response:
    return _export_response(await asyncio.to_thread(_job_report, job_id), format)


def _load_fixture() -> Report:
    report = Report.model_validate_json(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not report.is_fixture:
        raise HTTPException(500, "Fixture file must set is_fixture=true.")
    return report


def _export_response(report: Report, fmt: str) -> Response:
    if fmt == "json":
        body, media, ext = report.model_dump_json(indent=2), "application/json", "json"
    else:
        body, media, ext = render_markdown(report), "text/markdown; charset=utf-8", "md"
    filename = export_filename(report, ext)
    return Response(body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/fixtures/sample-report", response_model=Report)
async def sample_report() -> Report:
    """Illustrative report used to build the UI. Flagged is_fixture=true; not real results."""
    return _load_fixture()


@app.get("/api/fixtures/sample-report/export")
async def export_sample_report(format: Literal["markdown", "json"] = "markdown") -> Response:
    return _export_response(_load_fixture(), format)
