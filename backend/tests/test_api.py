"""API endpoints, job persistence, SSE streaming and the labelled fixture."""
from __future__ import annotations

import json

from reproscope.events import TERMINAL_EVENT, bus
from reproscope.schemas import STAGES, Report

from .conftest import make_pdf, wait_for_job


def _create(client, pdf: bytes, zip_bytes: bytes | None = None, repo_url: str | None = None,
            options: dict | None = None):
    files = {"pdf": ("paper.pdf", pdf, "application/pdf")}
    data = {}
    if zip_bytes is not None:
        files["repo_zip"] = ("repo.zip", zip_bytes, "application/zip")
    if repo_url is not None:
        data["repo_url"] = repo_url
    if options is not None:
        data["options"] = json.dumps(options)
    return client.post("/api/jobs", files=files, data=data)


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_mode"] == "dev"            # no API key in tests
    assert [s["id"] for s in body["stages"]] == STAGES
    assert "available" in body["docker"]


def test_create_job_runs_full_pipeline(client, sample_pdf, sample_zip):
    resp = _create(client, sample_pdf, sample_zip, options={"max_claims": 3})
    assert resp.status_code == 201, resp.text
    job = resp.json()
    assert job["repo_source"] == "zip" and job["options"]["max_claims"] == 3

    final = wait_for_job(client, job["id"])
    assert final["status"] == "done"
    assert [final["stages"][s] for s in STAGES] == ["done"] * len(STAGES)
    assert final["pdf_pages"] == 2
    assert final["paper_title"] == "A Tiny MLP Baseline for Digits"

    ingest_out = client.get(f"/api/jobs/{job['id']}/stages/ingest").json()
    assert ingest_out["repo"]["python_files"] == 1
    extract_out = client.get(f"/api/jobs/{job['id']}/stages/extract").json()
    assert extract_out["origin"] == "dev_sample"      # no key in tests: labelled heuristic
    scan_out = client.get(f"/api/jobs/{job['id']}/stages/scan").json()
    assert scan_out["summary"]["entrypoints"] == ["train.py"] and scan_out["fact_counts"]["arg"] == 1
    assert client.get(f"/api/jobs/{job['id']}/stages/plan").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}/stages/execute").json()["runs"] == []   # dev mode: no claims
    assert client.get(f"/api/jobs/{job['id']}/logs").json() == []
    assert client.get(f"/api/jobs/{job['id']}/runs/R1/log").status_code == 404

    listed = client.get("/api/jobs").json()
    assert [j["id"] for j in listed] == [job["id"]]


def test_job_persists_across_new_client(sample_pdf, sample_zip):
    from fastapi.testclient import TestClient
    from reproscope.main import app
    with TestClient(app) as c1:
        job_id = _create(c1, sample_pdf, sample_zip).json()["id"]
        wait_for_job(c1, job_id)
    with TestClient(app) as c2:
        assert c2.get(f"/api/jobs/{job_id}").json()["stages"]["ingest"] == "done"
        assert len(c2.get(f"/api/jobs/{job_id}/events/history").json()) > 3


def test_events_history_and_sse_stream(client, sample_pdf, sample_zip):
    job_id = _create(client, sample_pdf, sample_zip).json()["id"]
    wait_for_job(client, job_id)

    history = client.get(f"/api/jobs/{job_id}/events/history").json()
    seqs = [e["seq"] for e in history]
    assert seqs == sorted(seqs) and seqs[0] == 1 and len(set(seqs)) == len(seqs)
    assert history[-1]["message"] == TERMINAL_EVENT
    assert any(e["stage"] == "ingest" and e["level"] == "stage" for e in history)

    # Finished job: the live stream sends the backlog then closes.
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    frames = [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]
    assert [f["seq"] for f in frames] == seqs

    # Resume after a given event id.
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": str(seqs[-3])}) as resp:
        body = "".join(resp.iter_text())
    resumed = [json.loads(l[6:])["seq"] for l in body.splitlines() if l.startswith("data: ")]
    assert resumed == seqs[-2:]


def test_replay_streams_stored_events(client, sample_pdf, sample_zip):
    job_id = _create(client, sample_pdf, sample_zip).json()["id"]
    wait_for_job(client, job_id)
    info = client.post(f"/api/jobs/{job_id}/replay?speed=100").json()
    assert info["event_count"] > 3
    with client.stream("GET", info["events_url"]) as resp:
        body = "".join(resp.iter_text())
    frames = [json.loads(l[6:]) for l in body.splitlines() if l.startswith("data: ")]
    assert len(frames) == info["event_count"]
    assert all(f["data"].get("replay") is True for f in frames if isinstance(f["data"], dict))


def test_create_job_validation_errors(client, sample_pdf, sample_zip):
    # neither / both repo inputs
    assert _create(client, sample_pdf).status_code == 422
    assert _create(client, sample_pdf, sample_zip, repo_url="https://github.com/a/b").status_code == 422
    # bad pdf
    r = _create(client, b"not a pdf", sample_zip)
    assert r.status_code == 400 and "PDF" in r.json()["detail"]
    # bad url
    r = _create(client, sample_pdf, repo_url="http://example.com/a/b")
    assert r.status_code == 400
    # bad options
    r = _create(client, sample_pdf, sample_zip, options={"max_claims": 999})
    assert r.status_code == 422
    assert client.get("/api/jobs").json() == []   # nothing was created


def test_ingest_failure_marks_job_failed(client, sample_zip):
    blank_pdf = make_pdf([""])
    job_id = _create(client, blank_pdf, sample_zip).json()["id"]
    final = wait_for_job(client, job_id)
    assert final["status"] == "failed"
    assert "no extractable text" in final["error"]
    assert final["stages"]["ingest"] == "failed"
    assert all(final["stages"][s] == "skipped" for s in STAGES[1:])


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/j_missing").status_code == 404
    assert client.get("/api/jobs/j_missing/events").status_code == 404


def test_report_available_after_pipeline_and_not_for_failed_job(client, sample_pdf, sample_zip):
    job_id = _create(client, sample_pdf, sample_zip).json()["id"]
    wait_for_job(client, job_id)
    report = Report.model_validate(client.get(f"/api/jobs/{job_id}/report").json())
    assert report.is_fixture is False and report.job.id == job_id and report.job.llm_mode == "dev"
    assert report.summary.total_claims == len(report.claims)

    failed_id = _create(client, make_pdf([""]), sample_zip).json()["id"]      # no text layer -> fails
    wait_for_job(client, failed_id)
    r = client.get(f"/api/jobs/{failed_id}/report")
    assert r.status_code == 409 and "No report yet" in r.json()["detail"]


def test_fixture_is_valid_and_labelled(client):
    body = client.get("/api/fixtures/sample-report").json()
    report = Report.model_validate(body)
    assert report.is_fixture is True
    assert "NOT produced" in report.fixture_note
    s = report.summary
    assert s.total_claims == len(report.claims)
    counted = sum(1 for c in report.claims if c.comparison and c.comparison.verdict == "REPRODUCED")
    assert counted == s.reproduced


def test_event_bus_sequence_is_monotonic():
    from reproscope import db
    from reproscope.db import Job
    with db.session() as s:
        s.add(Job(id="j_bus", repo_source="zip"))
        s.commit()
    a = bus.emit("j_bus", "one")
    b = bus.emit("j_bus", "two", level="warning", data={"x": 1})
    bus.forget("j_bus")                           # simulate restart: counter reloads from DB
    c = bus.emit("j_bus", "three")
    assert (a.seq, b.seq, c.seq) == (1, 2, 3)
    assert [e.message for e in bus.history("j_bus", 1)] == ["two", "three"]


def test_fixture_markdown_export_is_labelled(client):
    r = client.get("/api/fixtures/sample-report/export?format=markdown")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert "SAMPLE-FIXTURE" in r.headers["content-disposition"]
    md = r.text
    assert md.startswith("> **SAMPLE FIXTURE — NOT A REAL ANALYSIS.**")
    assert "| C1 | MLP, 1 hidden layer (64 units) | accuracy (%) | 97.2 | 93.1 | -4.1 | 1 | NOT_REPRODUCED |" in md
    assert "`train.py:8`" in md                      # repo evidence survives
    assert "NOT verified" in md                      # ungrounded claim is called out


def test_fixture_json_export_roundtrips(client):
    r = client.get("/api/fixtures/sample-report/export?format=json")
    assert r.status_code == 200
    assert Report.model_validate_json(r.text).is_fixture is True
    assert client.get("/api/fixtures/sample-report/export?format=pdf").status_code == 422


def test_job_exports(client, sample_pdf, sample_zip):
    job_id = _create(client, sample_pdf, sample_zip).json()["id"]
    wait_for_job(client, job_id)
    j = client.get(f"/api/jobs/{job_id}/export?format=json")
    assert j.status_code == 200 and Report.model_validate_json(j.text).job.id == job_id
    md = client.get(f"/api/jobs/{job_id}/export?format=markdown")
    assert md.status_code == 200 and "SAMPLE FIXTURE" not in md.text
    assert md.text.startswith("# Reproducibility report") and job_id in md.headers["content-disposition"]
