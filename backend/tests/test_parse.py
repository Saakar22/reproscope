"""Stage 6: metrics come only from files, log lines, or a verified LLM-quoted line."""
from __future__ import annotations

import json

import pytest
from sqlmodel import select

from reproscope import db, llm, storage
from reproscope.db import Claim, Job, Result, Run
from reproscope.stages import parse

LOG = """Epoch 1/3 train loss 0.91 train accuracy: 71.20
Epoch 3/3 train loss 0.12 train accuracy: 99.80
Validation accuracy: 95.10
Test accuracy: 96.48
Test F1: 0.9648
Test top-5 accuracy: 99.9"""


# ------------------------------------------------------------- helpers

@pytest.mark.parametrize("metric,family,variant", [
    ("Accuracy", "accuracy", None), ("Top-1 accuracy (%)", "accuracy", None), ("Macro-F1", "f1", "macro"),
    ("F1 score", "f1", None), ("RMSE", "rmse", None), ("ROC-AUC", "auc", None), ("perplexity", "perplexity", None),
    ("BLEU-4", "bleu", None), ("balanced accuracy", "balanced_accuracy", None), ("colour", None, None),
])
def test_family_and_variant(metric, family, variant):
    assert parse.family_of(metric) == family and parse.variant_of(metric) == variant


@pytest.mark.parametrize("raw,unit,pct,expected", [
    (0.9648, "percent", False, 96.48), (96.48, "percent", True, 96.48), (96.48, "percent", False, 96.48),
    (96.48, "fraction", False, 0.9648), (0.9648, "fraction", False, 0.9648), (12.3, "raw", False, 12.3),
])
def test_normalise(raw, unit, pct, expected):
    assert parse.normalise_value(raw, unit, pct)[0] == pytest.approx(expected)


def test_log_prefers_test_over_train_and_val():
    hit, n = parse.from_log(LOG, "accuracy", None)
    assert hit["raw_value"] == 96.48 and hit["evidence"]["line_no"] == 4 and n >= 4
    assert parse.from_log(LOG, "f1", "macro")[0]["raw_value"] == 0.9648
    assert parse.from_log(LOG, "rmse", None) == (None, 0)


def test_log_percent_sign_is_detected():
    hit, _ = parse.from_log("final eval acc = 0.912\nTest accuracy 91.2%", "accuracy", None)
    assert hit["raw_value"] == 91.2 and hit["printed_percent"]


def test_files_prefer_test_keys_and_hint(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(
        {"train": {"accuracy": 0.99}, "test": {"accuracy": 0.9648, "f1_macro": 0.95, "f1": 0.96}}))
    (tmp_path / "hist.csv").write_text("epoch,val_acc\n1,0.8\n2,0.9\n")
    outs = [{"path": "metrics.json", "size": 1}, {"path": "hist.csv", "size": 1}]
    acc = parse.from_files(tmp_path, outs, "accuracy", None, None)
    assert acc["raw_value"] == 0.9648 and acc["evidence"] == {"file": "metrics.json", "key": "test.accuracy"}
    f1 = parse.from_files(tmp_path, outs, "f1", "macro", None)
    assert f1["evidence"]["key"] == "test.f1_macro"
    assert parse.from_files(tmp_path, outs, "rmse", None, None) is None


# ---------------------------------------------------------- LLM fallback

def fake_llm(monkeypatch, data):
    monkeypatch.setattr(llm, "complete_json", lambda **kw: llm.LLMResult(data=data, origin="llm"))


CLAIM = {"metric": "BLEU", "experiment": "x", "dataset": None, "split": None}
ODD_LOG = "step 100\nscore(bleu-ish)=27.3 on newstest\ndone"


def test_llm_value_accepted_only_with_verbatim_line(monkeypatch):
    fake_llm(monkeypatch, {"found": True, "value": 27.3, "line": "score(bleu-ish)=27.3 on newstest", "reason": "r"})
    hit, _ = parse.from_llm(ODD_LOG, CLAIM)
    assert hit["raw_value"] == 27.3 and hit["source"] == "llm" and hit["evidence"]["line_no"] == 2


def test_llm_invented_line_rejected(monkeypatch):
    fake_llm(monkeypatch, {"found": True, "value": 27.3, "line": "BLEU = 27.3", "reason": "r"})
    hit, why = parse.from_llm(ODD_LOG, CLAIM)
    assert hit is None and "does not appear in the log" in why


def test_llm_value_not_on_line_rejected(monkeypatch):
    fake_llm(monkeypatch, {"found": True, "value": 29.0, "line": "score(bleu-ish)=27.3 on newstest", "reason": "r"})
    hit, why = parse.from_llm(ODD_LOG, CLAIM)
    assert hit is None and "does not contain the value" in why


# ------------------------------------------------------------ full stage

@pytest.fixture()
def job():
    jid = "j_parse"
    with db.session() as s:
        s.add(Job(id=jid, repo_source="zip"))
        s.commit()
        for cid, metric, unit in [("C1", "Accuracy", "percent"), ("C2", "Macro-F1", "fraction"),
                                  ("C3", "Accuracy", "percent"), ("C4", "Macro-F1", "fraction"),
                                  ("C5", "Accuracy", "percent"), ("C6", "Accuracy", "percent")]:
            s.add(Claim(job_id=jid, id=cid, experiment="e", metric=metric, unit=unit, reported_value=1,
                        page=1, quote="q", grounded=True))
        s.commit()
    # R1: baseline (stdout only), R2: train.py (results.json), R3: failed run
    for rid, log, outputs in [("R1", "Test accuracy: 96.39\n", []),
                              ("R2", "Test accuracy: 96.48\nTest F1: 0.9648\n", [{"path": "results.json", "size": 40}]),
                              ("R3", "Traceback...\nValueError: boom\n", [])]:
        d = storage.runs_dir(jid) / rid
        (d / "outputs").mkdir(parents=True)
        (d / "log.txt").write_text(log)
        if outputs:
            (d / "outputs" / "results.json").write_text(json.dumps({"accuracy": 0.9648, "f1": 0.9648}))
        with db.session() as s:
            s.add(Run(job_id=jid, id=rid, plan_id="P", claim_id="C1", command="python x.py",
                      status="ok" if rid != "R3" else "error", exit_code=0 if rid != "R3" else 1,
                      log_path=f"runs/{rid}/log.txt", environment=json.dumps({"outputs": outputs})))
            s.commit()
    plans = [{"id": f"P{i}", "claim_id": f"C{i}", "runnable": True, "metric_key": None} for i in range(1, 6)]
    plans.append({"id": "P6", "claim_id": "C6", "runnable": False, "reason": "requires a GPU", "metric_key": None})
    storage.write_stage(jid, "plan", {"plans": plans})
    storage.write_stage(jid, "execute", {
        "plan_runs": {"P1": "R1", "P2": "R1", "P3": "R2", "P4": "R2", "P5": "R3"},
        "runs": [{"id": "R3", "status": "error", "reason": None}]})
    return {"id": jid, "llm_mode": "dev"}


def test_stage_results_with_evidence(job):
    logs = []
    out = parse.run(job["id"], job, lambda m, **kw: logs.append((kw.get("level", "info"), m)))
    r = {x["claim_id"]: x for x in out["results"]}

    assert r["C1"]["status"] == "found" and r["C1"]["value"] == 96.39 and r["C1"]["source"] == "regex"
    assert r["C1"]["evidence"] == {"line_no": 1, "line_text": "Test accuracy: 96.39"}

    assert r["C2"]["status"] == "no_metric"                      # baseline.py never prints F1
    assert any("LLM fallback" in a for a in r["C2"]["attempts"])

    assert r["C3"]["value"] == pytest.approx(96.48) and r["C3"]["source"] == "file"
    assert r["C3"]["evidence"] == {"file": "results.json", "key": "accuracy"}
    assert "fraction to percent" in r["C3"]["normalisation"]

    assert r["C4"]["value"] == pytest.approx(0.9648) and r["C4"]["normalisation"] is None
    assert "paper reports macro f1" in r["C4"]["variant_note"]

    assert r["C5"]["status"] == "not_run" and r["C5"]["run_status"] == "error"
    assert r["C6"]["status"] == "not_run" and "GPU" in r["C6"]["reason"]

    with db.session() as s:
        rows = {x.claim_id: x for x in s.exec(select(Result).where(Result.job_id == job["id"])).all()}
    assert set(rows) == {"C1", "C3", "C4"}
    assert json.loads(rows["C3"].evidence)["file"] == "results.json"
    assert any(level == "warning" and "C4" in m and "macro" in m for level, m in logs)
