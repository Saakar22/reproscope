"""Pipeline stages. Each stage module exposes
`run(job_id, job: dict, log: Callable) -> dict` and is registered here.
Stages not yet registered are reported as `not_implemented` — never faked."""
from __future__ import annotations

from typing import Any, Callable

from . import compare, execute, extract, ingest, parse, plan, scan

StageFn = Callable[[str, dict[str, Any], Callable[..., None]], dict[str, Any]]

STAGE_FUNCS: dict[str, StageFn] = {
    "ingest": ingest.run,
    "extract": extract.run,
    "scan": scan.run,
    "plan": plan.run,
    "execute": execute.run,
    "parse": parse.run,
    "compare": compare.run,
}
