"""Stage 5 — Experiment execution in the Docker sandbox.

One container run per distinct planned command (claims that share a command share a run).
Failures are recorded as they happened. The only automatic changes are disclosed as
deviations: installing a module the code imports but the repo didn't list, and one retry
with network access when the code tries to download something at run time.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from sqlmodel import delete

from .. import sandbox, storage
from ..db import Deviation, Run, session, utcnow

MAX_MODULE_FIXES = 2
MAX_STREAMED_LINES = 300


def run(job_id: str, job: dict[str, Any], log: Callable[..., None]) -> dict[str, Any]:
    plan_out = storage.read_stage(job_id, "plan") or {"plans": []}
    card = storage.read_stage(job_id, "scan") or {"facts": []}
    options = json.loads(job.get("options") or "{}")
    timeout_s = int(options.get("timeout_s", 600))

    groups: dict[str, list[dict[str, Any]]] = {}
    for p in plan_out["plans"]:
        if p["runnable"] and p["argv"]:
            groups.setdefault(p["command"], []).append(p)

    _reset_rows(job_id)
    out: dict[str, Any] = {"docker_available": None, "image": None, "python": None, "packages": [],
                           "runs": [], "plan_runs": {}, "deviations": []}
    if not groups:
        log("No runnable plans; nothing to execute", level="warning")
        return out

    deviations: list[dict[str, Any]] = []

    def deviate(kind: str, detail: str, run_id: str | None = None) -> None:
        d = {"id": f"D{len(deviations) + 1}", "run_id": run_id, "kind": kind, "detail": detail}
        deviations.append(d)
        with session() as s:
            s.add(Deviation(job_id=job_id, **d))
            s.commit()
        log(f"Deviation {d['id']}: {detail}", level="warning")

    runs = []
    for k, (command, plans) in enumerate(groups.items(), 1):
        rid = f"R{k}"
        runs.append({"id": rid, "command": command, "argv": plans[0]["argv"], "plans": plans})
        for p in plans:
            out["plan_runs"][p["id"]] = rid

    # ---- Docker available?
    try:
        sandbox.client()
        out["docker_available"] = True
    except sandbox.SandboxUnavailable as exc:
        out["docker_available"] = False
        log(f"{exc}. No experiment was executed; affected claims will be reported as NOT_RUN.", level="error")
        for r in runs:
            _save_run(job_id, r, status="not_executed", reason=str(exc))
            out["runs"].append(_summary(r, "not_executed", reason=str(exc)))
        return out

    # ---- environment image (network used only here, for pip install)
    facts = card.get("facts", [])
    extra_packages: list[str] = []
    spec = sandbox.make_spec(facts)
    out["python"] = spec.python
    log(f"Python {spec.python}: {spec.python_reason}")
    if "using" in spec.python_reason or "no Python version stated" in spec.python_reason:
        deviate("python_version_assumed", f"Python {spec.python} used ({spec.python_reason})")
    if not spec.requirements:
        log("No dependencies declared by the repository; the image contains only the Python standard library",
            level="warning")

    def build(current: sandbox.EnvSpec) -> str | None:
        log(f"Preparing environment image ({len(current.requirements)} package(s)"
            + (", CPU-only PyTorch wheels" if current.needs_torch_cpu else "") + ")")
        try:
            tag, reused = sandbox.ensure_image(current, lambda line: log(line, level="log"))
        except sandbox.BuildFailed as exc:
            log(f"Environment build failed: {exc}", level="error", data={"log_tail": exc.log_tail})
            return None
        log("Reusing cached environment image" if reused else "Environment image built", data={"image": tag})
        return tag

    image = build(spec)
    if image is None:
        for r in runs:
            _save_run(job_id, r, status="error", reason="environment build failed")
            out["runs"].append(_summary(r, "error", reason="environment build failed (see event log)"))
        out["deviations"] = deviations
        return out
    out["image"] = image
    out["packages"] = sandbox.image_freeze(image)

    # ---- runs
    for r in runs:
        rid = r["id"]
        claim_ids = [p["claim_id"] for p in r["plans"]]
        run_dir = storage.runs_dir(job_id) / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        _save_run(job_id, r, status="running", image=image)
        log(f"{rid} starting for {', '.join(claim_ids)}: {r['command']}",
            data={"run_id": rid, "claims": claim_ids, "timeout_s": timeout_s})

        attempts: list[dict[str, Any]] = []
        network = False
        fixes = 0
        while True:
            streamed = 0

            def on_line(line: str) -> None:
                nonlocal streamed
                streamed += 1
                if streamed <= MAX_STREAMED_LINES:
                    log(line[:500], level="log", data={"run_id": rid})
                elif streamed == MAX_STREAMED_LINES + 1:
                    log("… further output is in the full run log", level="log", data={"run_id": rid})

            res = sandbox.run_in_sandbox(image, r["argv"], storage.repo_dir(job_id), run_dir / "outputs",
                                         timeout_s=timeout_s, network=network, on_line=on_line)
            n = len(attempts) + 1
            (run_dir / f"log.attempt{n}.txt").write_text(res.log, encoding="utf-8")
            attempts.append({"attempt": n, "status": res.status, "exit_code": res.exit_code,
                             "seconds": res.seconds, "network": res.network, "failure": res.failure,
                             "log": f"runs/{rid}/log.attempt{n}.txt"})

            if res.failure == "missing_module" and fixes < MAX_MODULE_FIXES:
                pkg = sandbox.package_for_module(res.missing_module)
                if pkg in extra_packages:
                    break
                fixes += 1
                extra_packages.append(pkg)
                deviate("added_dependency", f"Installed '{pkg}': the code imports '{res.missing_module}' "
                        "but the repository does not list it", rid)
                spec = sandbox.make_spec(facts, extra_packages)
                new_image = build(spec)
                if new_image is None:
                    break
                image = out["image"] = new_image
                out["packages"] = sandbox.image_freeze(image)
                continue
            if res.failure == "network" and not network:
                network = True
                deviate("network_at_runtime", "Re-ran with network access: the code tried to download "
                        "data or weights while running", rid)
                continue
            break

        log_path = f"runs/{rid}/log.txt"
        (run_dir / "log.txt").write_text(res.log, encoding="utf-8")
        reason = {"gpu": "the code requires a CUDA GPU; the sandbox is CPU-only",
                  "timeout": f"stopped after the {timeout_s}s time limit",
                  "missing_module": f"missing module '{res.missing_module}' could not be resolved",
                  "network": "the code needs network access"}.get(res.failure or "", None)
        if res.failure == "timeout":
            deviate("timeout", f"{rid} exceeded the {timeout_s}s limit and was stopped; no metric from a full run",
                    rid)
        _save_run(job_id, r, status=res.status, exit_code=res.exit_code, seconds=res.seconds, log_path=log_path,
                  image=image, network=res.network, outputs=res.outputs, attempts=attempts,
                  packages=out["packages"], python=spec.python, reason=reason)
        level = "info" if res.status == "ok" else "warning"
        log(f"{rid} finished: {res.status}" + (f" (exit {res.exit_code})" if res.exit_code is not None else "")
            + f" in {res.seconds:.1f}s" + (f"; {reason}" if reason else "")
            + (f"; {len(res.outputs)} output file(s)" if res.outputs else ""),
            level=level, data={"run_id": rid, "status": res.status})
        out["runs"].append(_summary(r, res.status, exit_code=res.exit_code, seconds=res.seconds,
                                    network=res.network, outputs=res.outputs, failure=res.failure,
                                    attempts=attempts, log_path=log_path, reason=reason))

    out["deviations"] = deviations
    ok = sum(1 for r in out["runs"] if r["status"] == "ok")
    log(f"{ok} of {len(out['runs'])} run(s) completed successfully")
    return out


# ------------------------------------------------------------------ rows

def _reset_rows(job_id: str) -> None:
    with session() as s:
        s.exec(delete(Run).where(Run.job_id == job_id, Run.kind == "primary"))
        s.exec(delete(Deviation).where(Deviation.job_id == job_id))
        s.commit()


def _save_run(job_id: str, r: dict[str, Any], *, status: str, exit_code: int | None = None,
              seconds: float | None = None, log_path: str | None = None, image: str | None = None,
              network: str = "none", outputs: list | None = None, attempts: list | None = None,
              packages: list | None = None, python: str | None = None, reason: str | None = None) -> None:
    from ..config import get_settings
    settings = get_settings()
    env = {"python": python, "cpus": settings.sandbox_cpus, "memory": settings.sandbox_memory,
           "network": network, "claim_ids": [p["claim_id"] for p in r["plans"]],
           "plan_ids": [p["id"] for p in r["plans"]], "argv": r["argv"], "outputs": outputs or [],
           "attempts": attempts or [], "packages": packages or [], "reason": reason}
    with session() as s:
        row = s.get(Run, (job_id, r["id"])) or Run(job_id=job_id, id=r["id"], plan_id=r["plans"][0]["id"],
                                                  claim_id=r["plans"][0]["claim_id"], command=r["command"],
                                                  started_at=utcnow())
        row.status, row.exit_code, row.seconds, row.log_path = status, exit_code, seconds, log_path
        row.image_tag = image or row.image_tag
        row.environment = json.dumps(env)
        s.add(row)
        s.commit()


def _summary(r: dict[str, Any], status: str, **kw: Any) -> dict[str, Any]:
    return {"id": r["id"], "command": r["command"], "argv": r["argv"],
            "plan_ids": [p["id"] for p in r["plans"]], "claim_ids": [p["claim_id"] for p in r["plans"]],
            "status": status, **kw}
