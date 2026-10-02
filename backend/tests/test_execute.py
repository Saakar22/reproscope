"""Stage 5 end to end: runs grouped commands, records runs/deviations, auto-fixes honestly."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest
from sqlmodel import select

from reproscope import db, sandbox, storage
from reproscope.db import Deviation, Job, Run
from reproscope.stages import execute
from reproscope.stages.scan import scan_repo

from .test_sandbox import docker

PLANTED = Path(__file__).parent / "fixtures" / "planted_repo"


def setup_job(job_id: str, repo_src: Path | None, plans: list[dict], repo_files: dict[str, str] | None = None):
    dest = storage.repo_dir(job_id)
    if repo_src:
        shutil.copytree(repo_src, dest)
    else:
        dest.mkdir(parents=True)
    for name, text in (repo_files or {}).items():
        (dest / name).write_text(text)
    card, facts = scan_repo(dest)
    card["facts"] = [{"id": f"F{i}", **asdict(f)} for i, f in enumerate(facts, 1)]
    storage.write_stage(job_id, "scan", card)
    storage.write_stage(job_id, "plan", {"plans": plans})
    with db.session() as s:
        s.add(Job(id=job_id, repo_source="zip", options=json.dumps({"timeout_s": 120})))
        s.commit()
    return {"id": job_id, "llm_mode": "live", "options": json.dumps({"timeout_s": 120})}


def plan(pid, cid, argv, runnable=True):
    return {"id": pid, "claim_id": cid, "runnable": runnable, "argv": argv,
            "command": " ".join(argv) if argv else None}


def collect():
    logs = []
    return logs, (lambda msg, **kw: logs.append((kw.get("level", "info"), msg)))


def test_no_runnable_plans():
    job = setup_job("j_e0", PLANTED, [plan("P1", "C1", None, runnable=False)])
    logs, log = collect()
    out = execute.run(job["id"], job, log)
    assert out["runs"] == [] and any("nothing to execute" in m for _, m in logs)


def test_docker_unavailable_is_reported_not_simulated(monkeypatch):
    job = setup_job("j_e1", PLANTED, [plan("P1", "C1", ["python", "baseline.py"])])

    def down():
        raise sandbox.SandboxUnavailable("Docker is not available: daemon not running")
    monkeypatch.setattr(sandbox, "client", down)
    logs, log = collect()
    out = execute.run(job["id"], job, log)
    assert out["docker_available"] is False
    assert out["runs"][0]["status"] == "not_executed" and "daemon not running" in out["runs"][0]["reason"]
    assert any(level == "error" and "No experiment was executed" in m for level, m in logs)
    with db.session() as s:
        assert s.get(Run, ("j_e1", "R1")).status == "not_executed"


@docker
def test_planted_runs_grouped_and_recorded():
    job = setup_job("j_e2", PLANTED, [
        plan("P1", "C1", ["python", "baseline.py"]),
        plan("P2", "C2", ["python", "baseline.py"]),                    # same command -> same run
        plan("P3", "C3", ["python", "train.py", "--out", "results.json"]),
        plan("P4", "C4", None, runnable=False),
    ])
    logs, log = collect()
    out = execute.run(job["id"], job, log)
    assert out["plan_runs"] == {"P1": "R1", "P2": "R1", "P3": "R2"}
    runs = {r["id"]: r for r in out["runs"]}
    assert runs["R1"]["status"] == "ok" and runs["R1"]["claim_ids"] == ["C1", "C2"]
    assert runs["R2"]["status"] == "ok" and runs["R2"]["outputs"][0]["path"] == "results.json"
    assert any(p.lower().startswith("scikit-learn==") for p in out["packages"])     # exact versions recorded
    assert any(level == "log" and m.startswith("Test accuracy:") for level, m in logs)  # live log lines
    log_text = (storage.job_dir("j_e2") / runs["R1"]["log_path"]).read_text()
    assert "Test accuracy:" in log_text
    with db.session() as s:
        row = s.get(Run, ("j_e2", "R2"))
    env = json.loads(row.environment)
    assert row.status == "ok" and row.exit_code == 0 and env["network"] == "none" and env["python"] == "3.10"
    assert json.loads((storage.runs_dir("j_e2") / "R2" / "outputs" / "results.json").read_text())["accuracy"] > 0.5


@docker
def test_missing_module_is_installed_and_disclosed():
    job = setup_job("j_e3", None, [plan("P1", "C1", ["python", "main.py"])], repo_files={
        "main.py": "import six\nprint('Test accuracy: 50.00')\nif __name__ == '__main__':\n    pass\n",
        "requirements.txt": "",
        "README.md": "Tested with Python 3.10\n"})
    logs, log = collect()
    out = execute.run(job["id"], job, log)
    run = out["runs"][0]
    assert run["status"] == "ok" and len(run["attempts"]) == 2
    assert run["attempts"][0]["failure"] == "missing_module"
    kinds = [d["kind"] for d in out["deviations"]]
    assert kinds == ["added_dependency"] and "six" in out["deviations"][0]["detail"]
    with db.session() as s:
        rows = s.exec(select(Deviation).where(Deviation.job_id == "j_e3")).all()
    assert [r.kind for r in rows] == ["added_dependency"] and rows[0].run_id == "R1"


@docker
def test_gpu_only_failure_is_not_runnable():
    job = setup_job("j_e4", None, [plan("P1", "C1", ["python", "main.py"])], repo_files={
        "main.py": "raise AssertionError('Torch not compiled with CUDA enabled')\n",
        "README.md": "Python 3.10\n"})
    _, log = collect()
    out = execute.run(job["id"], job, log)
    assert out["runs"][0]["status"] == "not_runnable" and "CUDA" in out["runs"][0]["reason"]
