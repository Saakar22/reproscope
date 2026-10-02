"""Stage 7: verdicts, deterministic findings (planted answer key), explanations, end to end."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest
from sqlmodel import select

from reproscope import db, diagnose, llm, storage
from reproscope.db import Claim, Comparison, Finding, Job, Run
from reproscope.report import build_report, render_markdown
from reproscope.rules import Ctx, run_all
from reproscope.stages import compare, execute, parse
from reproscope.stages import plan as plan_stage
from reproscope.stages.scan import scan_repo
from reproscope.verdicts import tolerance, verdict

from .test_sandbox import docker

PLANTED = Path(__file__).parent / "fixtures" / "planted_repo"
PAGES = [
    "Small Networks, Solid Baselines\nAbstract. We study small MLPs on digits.",
    "We use an 80/20 train/test split. The MLP has 64 hidden units and is trained with Adam at a\n"
    "learning rate of 0.001 for 200 epochs.\nTable 1\nLogistic regression 96.1 ± 0.3 0.960\n"
    "MLP (64 hidden) 97.2 ± 0.4 0.971\nCNN 98.4",
]
HPS = [{"name": "learning_rate", "value": "0.001", "quote": "learning rate of 0.001", "verified": True},
       {"name": "epochs", "value": "200", "quote": "200 epochs", "verified": True}]
CLAIMS = [
    dict(id="C1", experiment="Logistic regression", metric="Accuracy", unit="percent", reported_value=96.1,
         reported_std=0.3, quote="Logistic regression 96.1 ± 0.3", hyperparameters=[]),
    dict(id="C2", experiment="Logistic regression", metric="Macro-F1", unit="fraction", reported_value=0.960,
         reported_std=None, quote="Logistic regression 96.1 ± 0.3 0.960", hyperparameters=[]),
    dict(id="C3", experiment="MLP (64 hidden)", metric="Accuracy", unit="percent", reported_value=97.2,
         reported_std=0.4, quote="MLP (64 hidden) 97.2 ± 0.4", hyperparameters=HPS),
    dict(id="C4", experiment="MLP (64 hidden)", metric="Macro-F1", unit="fraction", reported_value=0.971,
         reported_std=None, quote="MLP (64 hidden) 97.2 ± 0.4 0.971", hyperparameters=HPS),
    dict(id="C5", experiment="CNN", metric="Accuracy", unit="percent", reported_value=98.4,
         reported_std=None, quote="CNN 98.4", hyperparameters=[]),
]


# --------------------------------------------------------------- verdicts

def c(**kw):
    return {"reported_value": 97.2, "reported_std": None, "unit": "percent", "higher_is_better": True, **kw}


def test_tolerance_bases():
    assert tolerance(c(reported_std=0.4), "accuracy", []) == (0.8, "2 × reported std (0.4)")
    assert tolerance(c(), "accuracy", [])[0] == 1.0
    assert tolerance(c(unit="fraction", reported_value=0.97), "f1", [])[0] == 0.01
    assert tolerance(c(unit="raw", reported_value=2.5), "rmse", [])[0] == pytest.approx(0.075)
    tol, basis = tolerance(c(), "accuracy", [95.0, 97.0, 99.0])
    assert tol == pytest.approx(4.0) and "widened" in basis


@pytest.mark.parametrize("values,expected", [([97.5], "REPRODUCED"), ([98.9], "PARTIAL"), ([92.0], "NOT_REPRODUCED")])
def test_verdict_thresholds(values, expected):
    assert verdict(c(), values, "accuracy", set())["verdict"] == expected


def test_better_than_reported_is_flagged_not_rewarded():
    v = verdict(c(), [99.9], "accuracy", set())
    assert v["better_than_reported"] and v["verdict"] == "PARTIAL" and "not evidence of reproduction" in v["note"]
    lower_better = verdict(c(reported_value=2.0, unit="raw", higher_is_better=False), [1.0], "rmse", set())
    assert lower_better["better_than_reported"]


def test_reduced_budget_cannot_fail():
    v = verdict(c(), [80.0], "accuracy", {"timeout"})
    assert v["verdict"] == "INCONCLUSIVE" and "not the full experiment" in v["note"]


def test_mean_of_repeats():
    v = verdict(c(), [96.0, 97.0, 98.0], "accuracy", set())
    assert v["obtained"] == 97.0 and v["seed_values"] == [96.0, 97.0, 98.0] and "Mean of 3 runs" in v["note"]


# ------------------------------------------------------------------ rules

def planted_ctx(measurements=None):
    card, facts = scan_repo(PLANTED)
    card["facts"] = [{"id": f"F{i}", **asdict(f)} for i, f in enumerate(facts, 1)]
    claims = {x["id"]: {**x, "page": 2, "grounded": True, "dataset": "digits", "split": "80/20"} for x in CLAIMS}
    cands = {k["id"]: k for k in plan_stage.build_candidates(card, PLANTED)}
    plans = []
    for i, (cid, cand) in enumerate([("C1", "K2"), ("C2", "K2"), ("C3", "K1"), ("C4", "K1"), ("C5", "K3")], 1):
        v = plan_stage.validate_plan({"candidate_id": cand, "extra_args": [], "runnable": True}, claims[cid],
                                     cands, card, PLANTED)
        from reproscope.params import compare_hyperparameters, effective_values
        eff = effective_values(v["candidate"]["script"], v["argv"][2:], card["facts"], v["candidate"]["source_ref"])
        plans.append({"id": f"P{i}", "claim_id": cid, "runnable": v["runnable"], "reason": v["reason"],
                      "argv": v["argv"], "command": " ".join(v["argv"]), "script": v["candidate"]["script"],
                      "source_ref": v["candidate"]["source_ref"],
                      "overrides": compare_hyperparameters(claims[cid]["hyperparameters"], eff)})
    return Ctx(repo=PLANTED, pages=PAGES, claims=claims, plans=plans, card=card, exec_out={"deviations": [], "runs": []},
               comparisons={}, measurements=measurements or {})


def by_rule(findings):
    out = {}
    for f in findings:
        out.setdefault(f["rule"], []).append(f)
    return out


def test_planted_findings_match_answer_key():
    ctx = planted_ctx({"C3": {"values": [97.41, 96.48, 97.59], "runs": ["R2", "R2.2", "R2.3"]}})
    f = by_rule(run_all(ctx))
    truth = json.loads((PLANTED.parent / "planted_truth.json").read_text())["planted"]
    assert set(truth) == {"P1", "P2", "P3", "P4", "P5", "P6"}

    lr = [x for x in f["PARAM_MISMATCH"] if "lr" in x["key"]]                       # P1
    assert {x["claim_ids"][0] for x in lr} == {"C3", "C4"}
    assert lr[0]["repo_evidence"][0] == {"file": "train.py", "line": 21,
                                         "snippet": 'ap.add_argument("--lr", type=float, default=0.01, help="learning rate")'}
    assert lr[0]["paper_evidence"]["quote"] == "learning rate of 0.001"

    split = f["SPLIT_MISMATCH"]                                                     # P2
    assert len(split) == 1 and split[0]["claim_ids"] == ["C3", "C4"] and "test_size=0.3" in split[0]["summary"]
    assert split[0]["paper_evidence"]["page"] == 2 and "80/20" in split[0]["paper_evidence"]["quote"]

    seeds = f["SEED_NOT_SET"]                                                       # P3
    assert [s["key"] for s in seeds] == ["SEED_NOT_SET:train.py", "SEED_NOT_SET:train_gpu.py"] or \
        {s["key"] for s in seeds} == {"SEED_NOT_SET:train.py", "SEED_NOT_SET:train_gpu.py"}
    assert "baseline.py" not in json.dumps(seeds)

    md = f["METRIC_DEFINITION"]                                                     # P4
    assert [m["claim_ids"] for m in md] == [["C4"]]                                  # NOT C2 (baseline.py)
    assert "average='micro'" in md[0]["summary"] and md[0]["repo_evidence"][0]["line"] == 43
    assert 'average="micro"' in md[0]["repo_evidence"][0]["snippet"]

    deps = f["DEPS_UNPINNED"][0]                                                    # P5
    assert "scikit-learn" in deps["summary"] and "pyyaml" not in deps["summary"]

    gpu = f["GPU_ONLY"]                                                              # P6
    assert gpu[0]["claim_ids"] == ["C5"] and gpu[0]["severity"] == "high"

    torch = f["MISSING_DEPENDENCY"]
    assert [t["key"] for t in torch] == ["MISSING_DEPENDENCY:torch"] and torch[0]["repo_evidence"][0]["line"] == 3

    nd = f["NONDETERMINISM"][0]
    assert nd["claim_ids"] == ["C3"] and "97.41, 96.48, 97.59" in nd["summary"]

    ids = [x["id"] for x in run_all(ctx)]
    assert ids == [f"E{i}" for i in range(1, len(ids) + 1)]
    severities = [x["severity"] for x in run_all(ctx)]
    assert severities == sorted(severities, key={"high": 0, "medium": 1, "low": 2}.get)


def test_no_false_findings_on_clean_seeded_script():
    ctx = planted_ctx()
    baseline_findings = [x for x in run_all(ctx) if set(x["claim_ids"]) & {"C1", "C2"}
                         and x["rule"] not in ("PARAM_UNSTATED",)]
    assert baseline_findings == []


# --------------------------------------------------------------- diagnose

def test_diagnosis_drops_uncited_and_unknown_evidence(monkeypatch):
    target = {"claim_id": "C3", "claim": {"experiment": "MLP", "metric": "Accuracy", "reported_value": 97.2},
              "comparison": {"verdict": "NOT_REPRODUCED"},
              "evidence": [{"id": "E1", "kind": "finding", "rule": "PARAM_MISMATCH", "summary": "lr differs"}]}
    monkeypatch.setattr(llm, "complete_json", lambda **kw: llm.LLMResult(origin="llm", data={"diagnoses": [
        {"claim_id": "C3", "unexplained": False, "hypotheses": [
            {"text": "lr differs", "evidence_ids": ["E1"], "likelihood": "medium", "suggested_check": None},
            {"text": "GPU nondeterminism", "evidence_ids": [], "likelihood": "low", "suggested_check": None},
            {"text": "data leak", "evidence_ids": ["E9"], "likelihood": "high", "suggested_check": None},
            {"text": "mixed", "evidence_ids": ["E1", "E7"], "likelihood": "high", "suggested_check": None}]},
        {"claim_id": "C99", "unexplained": False, "hypotheses": []}]}))
    out = diagnose.diagnose([target], lambda *a, **k: None)
    assert list(out) == ["C3"] and [h["text"] for h in out["C3"]["hypotheses"]] == ["lr differs"]
    assert out["C3"]["dropped"] == 3


def test_dev_mode_diagnosis_is_labelled():
    target = {"claim_id": "C3", "claim": {}, "comparison": {"verdict": "NOT_REPRODUCED"},
              "evidence": [{"id": "E1", "kind": "finding", "rule": "PARAM_MISMATCH", "summary": "lr differs",
                            "test": {"supports_hypothesis": "supports", "statement": "gap closed"}}]}
    out = diagnose.diagnose([target], lambda *a, **k: None)
    assert out["C3"]["origin"] == "dev_sample" and out["C3"]["hypotheses"][0]["likelihood"] == "high"


# ------------------------------------------------------- end to end (docker)

@docker
def test_stage7_end_to_end_with_repeats_and_hypothesis(monkeypatch):
    jid = "j_cmp"
    shutil.copytree(PLANTED, storage.repo_dir(jid))
    storage.pages_path(jid).write_text(json.dumps(PAGES))
    card, facts = scan_repo(storage.repo_dir(jid))
    card["facts"] = [{"id": f"F{i}", **asdict(f)} for i, f in enumerate(facts, 1)]
    for e in card["entrypoints"]:
        pass
    storage.write_stage(jid, "scan", card)
    opts = json.dumps({"timeout_s": 120, "enable_hypothesis_reruns": True})
    with db.session() as s:
        s.add(Job(id=jid, repo_source="zip", llm_mode="dev", options=opts))
        s.commit()
        for x in CLAIMS:
            s.add(Claim(job_id=jid, id=x["id"], experiment=x["experiment"], metric=x["metric"], unit=x["unit"],
                        reported_value=x["reported_value"], reported_std=x["reported_std"], page=2, quote=x["quote"],
                        hyperparameters=json.dumps(x["hyperparameters"]), grounded=True, split="80/20"))
        s.commit()
    job = {"id": jid, "llm_mode": "live", "options": opts}
    plans = [{"claim_id": cid, "candidate_id": k, "extra_args": [], "runnable": True, "reason": "",
              "metric_location": "stdout", "metric_key": None, "estimated_minutes": 1, "confidence": "high"}
             for cid, k in [("C1", "K2"), ("C2", "K2"), ("C3", "K1"), ("C4", "K1"), ("C5", "K3")]]
    real_llm = llm.complete_json
    monkeypatch.setattr(llm, "complete_json", lambda **kw: llm.LLMResult(data={"plans": plans}, origin="llm")
                        if kw["name"] == "run_plans" else real_llm(**kw))
    log = lambda *a, **k: None  # noqa: E731
    for stage in (plan_stage, execute, parse):
        storage.write_stage(jid, stage.__name__.rsplit(".", 1)[1], stage.run(jid, job, log))
    out = compare.run(jid, job, log)

    comps = out["comparisons"]
    assert comps["C1"]["verdict"] in ("REPRODUCED", "PARTIAL") and comps["C1"]["seed_values"] == [96.39]
    assert len(comps["C3"]["seed_values"]) == 3                       # unseeded: measured 3 times
    assert comps["C2"]["verdict"] == "NO_METRIC" and comps["C5"]["verdict"] == "NOT_RUN"

    with db.session() as s:
        runs = {r.id: r for r in s.exec(select(Run).where(Run.job_id == jid)).all()}
        findings = s.exec(select(Finding).where(Finding.job_id == jid)).all()
        assert s.exec(select(Comparison).where(Comparison.job_id == jid)).all()
    assert {"R2.2", "R2.3"} <= set(runs) and runs["R2.2"].kind == "repeat"
    assert "R1.2" not in runs                                          # seeded baseline is not repeated
    rules = {f.rule for f in findings}
    assert {"SPLIT_MISMATCH", "METRIC_DEFINITION", "SEED_NOT_SET", "GPU_ONLY", "DEPS_UNPINNED"} <= rules

    failing = [cid for cid in ("C3", "C4") if comps[cid]["verdict"] in ("PARTIAL", "NOT_REPRODUCED")]
    if failing:                                                        # depends on the random draw
        hyp = [r for r in runs.values() if r.kind == "hypothesis"]
        assert hyp and all("--lr 0.001" in r.command for r in hyp)
        tested = [f for f in findings if f.rule == "PARAM_MISMATCH" and json.loads(f.test or "null")]
        assert tested and json.loads(tested[0].test)["supports_hypothesis"] in (
            "supports", "partially_supports", "does_not_support")

    report = build_report(jid)
    assert report.summary.total_claims == 5 and report.summary.no_metric == 1 and report.summary.not_run == 1
    assert report.checklist.seed_set is False and report.checklist.deps_pinned is False
    md = render_markdown(report)
    assert "SAMPLE FIXTURE" not in md and "SPLIT_MISMATCH" in md


@docker
def test_hypothesis_rerun_mechanics_are_deterministic(monkeypatch):
    """Force a failing claim so the paper's lr is tested: one parameter changed, same repeat count."""
    jid = "j_hyp"
    shutil.copytree(PLANTED, storage.repo_dir(jid))
    storage.pages_path(jid).write_text(json.dumps(PAGES))
    card, facts = scan_repo(storage.repo_dir(jid))
    card["facts"] = [{"id": f"F{i}", **asdict(f)} for i, f in enumerate(facts, 1)]
    storage.write_stage(jid, "scan", card)
    opts = json.dumps({"timeout_s": 120, "enable_hypothesis_reruns": True})
    with db.session() as s:
        s.add(Job(id=jid, repo_source="zip", llm_mode="dev", options=opts))
        s.commit()
        s.add(Claim(job_id=jid, id="C1", experiment="MLP (64 hidden)", metric="Accuracy", unit="percent",
                    reported_value=99.9, page=2, quote="MLP (64 hidden) 97.2 ± 0.4",       # far above reality
                    hyperparameters=json.dumps(HPS), grounded=True))
        s.commit()
    job = {"id": jid, "llm_mode": "live", "options": opts}
    plans = [{"claim_id": "C1", "candidate_id": "K1", "extra_args": [], "runnable": True, "reason": "",
              "metric_location": "results.json", "metric_key": "accuracy", "estimated_minutes": 1, "confidence": "high"}]
    real_llm = llm.complete_json
    monkeypatch.setattr(llm, "complete_json", lambda **kw: llm.LLMResult(data={"plans": plans}, origin="llm")
                        if kw["name"] == "run_plans" else real_llm(**kw))
    log = lambda *a, **k: None  # noqa: E731
    for stage in (plan_stage, execute, parse):
        storage.write_stage(jid, stage.__name__.rsplit(".", 1)[1], stage.run(jid, job, log))
    out = compare.run(jid, job, log)

    # PARTIAL or NOT_REPRODUCED depending on the measured spread (which widens the tolerance)
    assert out["comparisons"]["C1"]["verdict"] in ("PARTIAL", "NOT_REPRODUCED")
    [t] = out["hypothesis_tests"]
    assert t["param"] == "lr" and t["override"] == {"--lr": "0.001"}
    assert t["run_ids"] == ["H1", "H1.2", "H1.3"] and len(t["values"]) == 3      # matches the 3-run primary
    assert t["supports_hypothesis"] in ("supports", "partially_supports", "does_not_support")
    assert "prove" not in t["statement"] or "does not prove" in t["statement"]   # never claims proof
    with db.session() as s:
        hyp = [r for r in s.exec(select(Run).where(Run.job_id == jid)).all() if r.kind == "hypothesis"]
        f = next(x for x in s.exec(select(Finding).where(Finding.job_id == jid)).all() if x.rule == "PARAM_MISMATCH")
    assert len(hyp) == 3 and all(r.command == "python train.py --out results.json --lr 0.001" for r in hyp)
    assert json.loads(hyp[0].overrides) == {"--lr": "0.001"}
    assert json.loads(f.test)["run_ids"] == ["H1", "H1.2", "H1.3"]


@pytest.mark.parametrize("before,after,outcome,phrase", [
    (-2.9, -0.4, "supports", "does not prove"),
    (-2.9, -1.2, "partially_supports", "one of several factors"),
    (-2.9, -2.8, "does_not_support", "does not explain"),
    (-0.04, 0.52, "not_applicable", "does not alter the verdict"),
    (-0.04, 1.5, "not_applicable", "fits the reported number better"),
])
def test_hypothesis_outcome_wording(before, after, outcome, phrase):
    from reproscope.verdicts import classify_hypothesis
    got, text = classify_hypothesis("--lr 0.001", before, after, 0.8)
    assert got == outcome and phrase in text and "proves" not in text
