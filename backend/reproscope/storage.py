"""Per-job artifact directory layout.

    artifacts/{job_id}/
        paper.pdf
        pages.json             # page texts, index 0 = page 1
        repo/                  # pristine repository copy (never executed in place)
        uploads/repo.zip       # original upload, when a ZIP was given
        stages/01_ingest.json  # structured output of every pipeline stage
        runs/{run_id}/         # per-run writable copy, outputs, log.txt
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import get_settings
from .schemas import STAGES


def job_dir(job_id: str) -> Path:
    return get_settings().artifacts_dir / job_id


def paper_path(job_id: str) -> Path:
    return job_dir(job_id) / "paper.pdf"


def pages_path(job_id: str) -> Path:
    return job_dir(job_id) / "pages.json"


def repo_dir(job_id: str) -> Path:
    return job_dir(job_id) / "repo"


def uploads_dir(job_id: str) -> Path:
    return job_dir(job_id) / "uploads"


def runs_dir(job_id: str) -> Path:
    return job_dir(job_id) / "runs"


def stage_path(job_id: str, stage: str) -> Path:
    index = STAGES.index(stage) + 1
    return job_dir(job_id) / "stages" / f"{index:02d}_{stage}.json"


def write_stage(job_id: str, stage: str, payload: Any) -> Path:
    path = stage_path(job_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def read_stage(job_id: str, stage: str) -> Any | None:
    path = stage_path(job_id, stage)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_pages(job_id: str) -> list[str]:
    return json.loads(pages_path(job_id).read_text(encoding="utf-8"))


def safe_join(root: Path, relative: str) -> Path:
    """Resolve `relative` under `root`, refusing anything that escapes it."""
    root = root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes the repository: {relative!r}")
    return target
