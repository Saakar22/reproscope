"""Stage 2: LLM output is schema-validated, grounded, trimmed and persisted."""
from __future__ import annotations

import json

import pytest
from sqlmodel import select

from reproscope import db, llm, storage
from reproscope.db import Claim, Job
from reproscope.stages import extract

PAGES = [
    "A Tiny MLP Baseline for Digits\nAbstract. We study small MLPs.",
    "Table 1: Results on digits (test accuracy %)\nMLP (64 hidden) 97.2\nLogistic regression 96.1 ± 0.3\n"
    "We train with Adam at a learning rate of 0.001 for 200 epochs.",
]


def claim(**kw):
    base = {"experiment": "MLP (64 hidden)", "dataset": "digits", "split": "test", "metric": "accuracy",
            "higher_is_better": True, "reported_value": 97.2, "reported_std": None, "unit": "percent",
            "source": "Table 1", "page": 2, "quote": "MLP (64 hidden) 97.2",
            "hyperparameters": [], "preprocessing": None, "is_main_result": True}
    return {**base, **kw}


@pytest.fixture()
def job():
    job_id = "j_extract"
    storage.job_dir(job_id).mkdir(parents=True)
    storage.pages_path(job_id).write_text(json.dumps(PAGES))
    with db.session() as s:
        s.add(Job(id=job_id, repo_source="zip", llm_mode="live", options=json.dumps({"max_claims": 6})))
        s.commit()
    return {"id": job_id, "llm_mode": "live", "options": json.dumps({"max_claims": 6})}


def fake_llm(monkeypatch, data):
    monkeypatch.setattr(llm, "complete_json",
                        lambda **kw: llm.LLMResult(data=data, origin="llm", model="test-model"))


def run(job):
    logs = []
    out = extract.run(job["id"], job, lambda msg, **kw: logs.append((msg, kw.get("level", "info"))))
    return out, logs


def test_grounded_and_hallucinated_claims(monkeypatch, job):
    fake_llm(monkeypatch, {"paper_title": "A Tiny MLP Baseline for Digits", "claims": [
        claim(hyperparameters=[
            {"name": "lr", "value": "0.001", "quote": "learning rate of 0.001"},
            {"name": "batch_size", "value": "64", "quote": "batch size of 64"},      # not in paper
        ]),
        claim(experiment="Logistic regression", reported_value=96.1, reported_std=0.3,
              quote="Logistic regression 96.1 ± 0.3", is_main_result=False),
        claim(experiment="CNN", reported_value=98.9, quote="CNN 98.9"),              # hallucinated
    ]})
    out, logs = run(job)
    by_exp = {c["experiment"]: c for c in out["claims"]}
    assert by_exp["MLP (64 hidden)"]["grounded"] and by_exp["Logistic regression"]["grounded"]
    assert not by_exp["CNN"]["grounded"]
    assert [c["id"] for c in out["claims"]] == ["C1", "C2", "C3"]
    assert out["claims"][-1]["experiment"] == "CNN"                   # unverified sorted last
    hps = {h["name"]: h["verified"] for h in by_exp["MLP (64 hidden)"]["hyperparameters"]}
    assert hps == {"lr": True, "batch_size": False}
    assert any("not found verbatim" in m for m, _ in logs)

    with db.session() as s:
        rows = s.exec(select(Claim).where(Claim.job_id == job["id"]).order_by(Claim.id)).all()
    assert len(rows) == 3 and rows[0].origin == "llm" and rows[0].grounded


def test_wrong_page_is_corrected_to_verified_page(monkeypatch, job):
    fake_llm(monkeypatch, {"paper_title": None, "claims": [claim(page=1)]})     # table is on page 2
    out, logs = run(job)
    c = out["claims"][0]
    assert c["grounded"] and c["page"] == 2 and c["grounding"]["llm_page"] == 1
    assert any("Corrected the page number" in m for m, _ in logs)
    with db.session() as s:
        assert s.get(Claim, (job["id"], "C1")).page == 2


def test_malformed_items_are_discarded(monkeypatch, job):
    fake_llm(monkeypatch, {"paper_title": None, "claims": [
        claim(), {"experiment": "broken"}, claim(page=0), claim(unit="percentage"),
    ]})
    out, logs = run(job)
    assert len(out["claims"]) == 1 and out["rejected_malformed"] == 3
    assert any("malformed" in m for m, _ in logs)


def test_duplicates_removed_and_max_claims_prefers_grounded(monkeypatch, job):
    job["options"] = json.dumps({"max_claims": 1})
    fake_llm(monkeypatch, {"paper_title": None, "claims": [
        claim(experiment="Fake", reported_value=99.9, quote="Fake 99.9"),
        claim(), claim(),
    ]})
    out, _ = run(job)
    assert len(out["claims"]) == 1 and out["claims"][0]["grounded"]


def test_llm_error_fails_stage_with_clear_message(monkeypatch, job):
    def boom(**kw):
        raise llm.LLMError("LLM request failed: HTTP 401 (check GROQ_API_KEY / LLM_API_KEY)")
    monkeypatch.setattr(llm, "complete_json", boom)
    from reproscope.stages.ingest import InputError
    with pytest.raises(InputError, match="401"):
        run(job)


def test_dev_mode_regex_scan_is_labelled(job):
    job["llm_mode"] = "dev"
    out, logs = run(job)                       # no key in tests -> dev mode
    assert out["origin"] == "dev_sample"
    assert all(c["origin"] == "dev_sample" for c in out["claims"])
    assert any("Dev mode" in m for m, _ in logs)


def test_long_paper_prompt_keeps_result_pages():
    pages = ["intro " * 4000, "Table 3 results 91.2 88.4 77.1", "References\n" + "x " * 4000]
    prompt, skipped = extract.build_prompt(pages, budget=30_000)
    assert "=== PAGE 2 ===" in prompt and skipped
