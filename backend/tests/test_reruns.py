"""User-approved reruns: strict validation, API contract, and (with Docker) a real rerun."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from reproscope import db, reruns, storage
from reproscope.db import Claim, Comparison, Job
from reproscope.stages.scan import scan_repo

from .test_sandbox import docker

PLANTED = Path(__file__).parent / "fixtures" / "planted_repo"


def make_job(jid: str, status: str = "done", image: str | None = None) -> str:
    shutil.copytree(PLANTED, storage.repo_dir(jid))
    card, facts = scan_repo(storage.repo_dir(jid))
    card["facts"] = [{"id": f"F{i}", **asdict(f)} for i, f in enumerate(facts, 1)]
    storage.write_stage(jid, "scan", card)
    storage.write_stage(jid, "plan", {"plans": [
        {"id": "P1", "claim_id": "C1", "runnable": True, "script": "train.py",
         "argv": ["python", "train.py", "--out", "results.json"], "metric_key": "accuracy"},
        {"id": "P2", "claim_id": "C2", "runnable": True, "script": "train.py",
         "argv": ["python", "train.py", "--out", "results.json"], "metric_key": "f1"},
        {"id": "P3", "claim_id": "C3", "runnable": False, "script": "train_gpu.py", "argv": None},
    ]})
    storage.write_stage(jid, "execute", {"image": image, "plan_runs": {"P1": "R1", "P2": "R1"}, "runs": []})
    with db.session() as s:
        s.add(Job(id=jid, repo_source="zip", status=status, options=json.dumps({"timeout_s": 120})))
        s.commit()
        for cid, metric, unit, rep in [("C1", "Accuracy", "percent", 99.9), ("C2", "Macro-F1", "fraction", 0.999),
                                       ("C3", "Accuracy", "percent", 98.4)]:
            s.add(Claim(job_id=jid, id=cid, experiment="MLP", metric=metric, unit=unit, reported_value=rep, page=2,
                        quote="q", grounded=True))
        s.add(Comparison(job_id=jid, claim_id="C1", reported=99.9, obtained=97.0, abs_delta=-2.9, tolerance=1.0,
                         seed_values=json.dumps([97.0]), verdict="PARTIAL"))
        s.add(Comparison(job_id=jid, claim_id="C2", reported=0.999, obtained=0.97, abs_delta=-0.029, tolerance=0.01,
                         seed_values=json.dumps([0.97]), verdict="PARTIAL"))
        s.add(Comparison(job_id=jid, claim_id="C3", reported=98.4, verdict="NOT_RUN"))
        s.commit()
    return jid


def req(overrides, claim="C1"):
    return reruns.RerunRequest(claim_id=claim, overrides=overrides)


def test_valid_override_replaces_or_appends_flag():
    jid = make_job("j_rr1")
    v = reruns.validate(jid, req({"--lr": "0.001"}))
    assert v["argv"] == ["python", "train.py", "--out", "results.json", "--lr", "0.001"]
    assert v["claim_ids"] == ["C1", "C2"] and v["repeats"] == 1          # shares run R1 with C2


@pytest.mark.parametrize("overrides,msg", [
    ({"--batch": "8"}, "not a CLI flag"),
    ({"--config": "configs/mlp128.yaml"}, "config or output"),
    ({"--out": "x.json"}, "config or output"),
    ({"--lr": "0.001; rm -rf /"}, "Unsafe"),
    ({"--lr": "../../etc"}, "Unsafe"),
    ({"--lr": "$(id)"}, "Unsafe"),
    ({"--hidden": "lots"}, "expects int"),
])
def test_invalid_overrides_rejected(overrides, msg):
    jid = make_job(f"j_rr_{abs(hash(json.dumps(overrides))) % 10**8}")
    with pytest.raises(reruns.RerunError, match=msg):
        reruns.validate(jid, req(overrides))


def test_not_runnable_or_unmeasured_claims_rejected():
    jid = make_job("j_rr2")
    with pytest.raises(reruns.RerunError, match="no runnable command"):
        reruns.validate(jid, req({"--lr": "0.1"}, claim="C3"))
    with pytest.raises(reruns.RerunError, match="no runnable command"):
        reruns.validate(jid, req({"--lr": "0.1"}, claim="C9"))


def test_request_model_limits():
    with pytest.raises(ValidationError, match="at most 3"):
        reruns.RerunRequest(claim_id="C1", overrides={"--a": "1", "--b": "2", "--c": "3", "--d": "4"})
    with pytest.raises(ValidationError):
        reruns.RerunRequest(claim_id="C1", overrides={})


def test_api_contract(client):
    jid = make_job("j_rr3", status="running")
    assert client.post(f"/api/jobs/{jid}/rerun", json={"claim_id": "C1", "overrides": {"--lr": "0.1"}}).status_code == 409
    with db.session() as s:
        job = s.get(Job, jid)
        job.status = "done"
        s.add(job)
        s.commit()
    r = client.post(f"/api/jobs/{jid}/rerun", json={"claim_id": "C1", "overrides": {"--nope": "1"}})
    assert r.status_code == 400 and "not a CLI flag" in r.json()["detail"]
    assert client.post(f"/api/jobs/{jid}/rerun", json={"claim_id": "C1"}).status_code == 422
    assert client.get(f"/api/jobs/{jid}/reruns").json() == []


def test_repo_file_viewer(client):
    jid = make_job("j_rr4")
    body = client.get(f"/api/jobs/{jid}/file", params={"path": "train.py"}).json()
    assert body["language"] == "python" and "--lr" in body["lines"][20] and not body["truncated"]
    assert client.get(f"/api/jobs/{jid}/file", params={"path": "../../../etc/passwd"}).status_code == 400
    assert client.get(f"/api/jobs/{jid}/file", params={"path": "nope.py"}).status_code == 404
    (storage.repo_dir(jid) / "w.bin").write_bytes(b"\0\1\2")
    assert client.get(f"/api/jobs/{jid}/file", params={"path": "w.bin"}).status_code == 415


@docker
def test_real_rerun_changes_only_the_flag(client):
    from reproscope import sandbox
    card = storage.read_stage(make_job("j_rr5"), "scan")
    spec = sandbox.make_spec(card["facts"])
    image, _ = sandbox.ensure_image(spec, lambda line: None)
    storage.write_stage("j_rr5", "execute", {"image": image, "plan_runs": {"P1": "R1", "P2": "R1"}, "runs": []})
    r = client.post("/api/jobs/j_rr5/rerun", json={"claim_id": "C1", "overrides": {"--lr": "0.001"}, "note": "paper lr"})
    assert r.status_code == 202 and r.json()["id"] == "U1"
    for _ in range(120):
        rec = client.get("/api/jobs/j_rr5/reruns").json()[0]
        if rec["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert rec["status"] == "done", rec.get("error")
    assert rec["argv"][-2:] == ["--lr", "0.001"] and rec["run_ids"] == ["U1"]
    assert set(rec["results"]) == {"C1", "C2"}
    assert rec["results"]["C1"]["outcome"] in ("supports", "partially_supports", "does_not_support")
    log = client.get("/api/jobs/j_rr5/runs/U1/log").text
    assert "Test accuracy" in log
