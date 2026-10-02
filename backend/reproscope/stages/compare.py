"""Stage 7 — Comparison & diagnosis.

1. Repeat runs for scripts without a fixed seed (measured spread, not assumed).
2. Verdict per claim with a stated tolerance.
3. Optional hypothesis reruns: the paper's value for ONE mismatched parameter at a time.
4. Deterministic discrepancy findings with paper / repo / log evidence.
5. Evidence-constrained explanations for claims that did not reproduce.
"""
from __future__ import annotations

import json
import statistics
from typing import Any, Callable, Optional

from sqlmodel import delete, select

from .. import sandbox, storage
from ..db import Claim, Comparison, Diagnosis, Finding, Run, session, utcnow
from ..diagnose import diagnose
from ..rules import Ctx, run_all
from ..verdicts import classify_hypothesis, simple, verdict
from .parse import family_of, measure

REPEATS = 3                       # total runs for unseeded scripts (primary + 2)
REPEAT_MAX_PRIMARY_S = 60         # only repeat runs that are cheap
MAX_HYPOTHESES_PER_GROUP = 2
MAX_HYPOTHESES_PER_JOB = 6
FAILING = {"PARTIAL", "NOT_REPRODUCED"}


def run(job_id: str, job: dict[str, Any], log: Callable[..., None]) -> dict[str, Any]:
    options = json.loads(job.get("options") or "{}")
    timeout_s = int(options.get("timeout_s", 600))
    plan_out = storage.read_stage(job_id, "plan") or {"plans": []}
    exec_out = storage.read_stage(job_id, "execute") or {"runs": [], "plan_runs": {}, "deviations": []}
    parse_out = storage.read_stage(job_id, "parse") or {"results": []}
    card = storage.read_stage(job_id, "scan") or {"facts": [], "entrypoints": [], "summary": {}}
    pages = storage.read_pages(job_id)
    with session() as s:
        claim_rows = s.exec(select(Claim).where(Claim.job_id == job_id).order_by(Claim.id)).all()
        s.exec(delete(Run).where(Run.job_id == job_id, Run.kind != "primary"))
        for table in (Comparison, Finding, Diagnosis):
            s.exec(delete(table).where(table.job_id == job_id))
        s.commit()
    claims = {c.id: {**c.model_dump(), "hyperparameters": _hps(c.hyperparameters)} for c in claim_rows}
    plans = {p["claim_id"]: p for p in plan_out["plans"]}
    parsed = {r["claim_id"]: r for r in parse_out["results"]}
    run_info = {r["id"]: r for r in exec_out.get("runs", [])}
    unseeded = set(card.get("summary", {}).get("unseeded_entrypoints", []))
    image = exec_out.get("image")
    docker_ok = _docker_ok(image, log)

    # ---- group claims by the primary run that produced them
    groups: dict[str, list[str]] = {}
    for cid, r in parsed.items():
        if r["status"] == "found" and r.get("run_id"):
            groups.setdefault(r["run_id"], []).append(cid)

    # ---- 1. repeats for unseeded scripts
    measurements: dict[str, dict[str, Any]] = {cid: {"values": [parsed[cid]["value"]], "runs": [parsed[cid]["run_id"]]}
                                               for cids in groups.values() for cid in cids}
    for rid, cids in groups.items():
        plan = plans[cids[0]]
        primary = run_info.get(rid, {})
        if plan.get("script") not in unseeded or not docker_ok:
            continue
        if (primary.get("seconds") or 0) > REPEAT_MAX_PRIMARY_S:
            log(f"{rid}: no seed, but the run is too long to repeat cheaply; single-run comparison", level="warning")
            continue
        log(f"{rid}: {plan['script']} sets no random seed; repeating the identical command "
            f"{REPEATS - 1} more time(s) to measure run-to-run spread")
        for k in range(2, REPEATS + 1):
            values = _run_and_measure(job_id, f"{rid}.{k}", "repeat", plan["argv"], {}, cids, claims, plans,
                                      image, timeout_s, primary.get("network") == "on", log)
            for cid, v in values.items():
                if v is not None:
                    measurements[cid]["values"].append(v)
                    measurements[cid]["runs"].append(f"{rid}.{k}")

    # ---- 2. verdicts
    deviation_kinds: dict[str, set[str]] = {}
    for d in exec_out.get("deviations", []):
        if d.get("run_id"):
            deviation_kinds.setdefault(d["run_id"], set()).add(d["kind"])
    comparisons: dict[str, dict[str, Any]] = {}
    for cid, c in claims.items():
        r = parsed.get(cid)
        if not c["grounded"]:
            comparisons[cid] = simple(c, "UNVERIFIED_CLAIM", "The quote or number was not found in the PDF text; "
                                      "excluded from scoring.")
        elif r is None:
            comparisons[cid] = simple(c, "NOT_RUN", "No plan was made for this claim.")
        elif r["status"] == "not_run":
            comparisons[cid] = simple(c, "NOT_RUN", r.get("reason") or "not executed", r.get("run_status"))
        elif r["status"] == "no_metric":
            comparisons[cid] = simple(c, "NO_METRIC", r.get("reason") or "no value found", "ok")
        else:
            comp = verdict(c, measurements[cid]["values"], family_of(c["metric"]),
                           deviation_kinds.get(r["run_id"], set()))
            comp["run_status"] = "ok"
            if r.get("variant_note"):
                comp["note"] = f"{comp['note']} {r['variant_note']}." if comp["note"] else f"{r['variant_note']}."
            comparisons[cid] = comp

    # ---- 3. hypothesis reruns
    tests: dict[tuple[str, str], dict[str, Any]] = {}
    if options.get("enable_hypothesis_reruns", True) and docker_ok:
        tests = _hypothesis_reruns(job_id, groups, claims, plans, comparisons, measurements, run_info,
                                   image, timeout_s, log)
    elif not docker_ok:
        log("Docker unavailable: repeat and hypothesis reruns skipped", level="warning")

    # ---- 4. findings
    ctx = Ctx(repo=storage.repo_dir(job_id), pages=pages, claims=claims, plans=list(plans.values()), card=card,
              exec_out=exec_out, comparisons=comparisons, measurements=measurements)
    findings = run_all(ctx)
    for f in findings:
        if f["rule"] == "PARAM_MISMATCH":
            param = f["key"].rsplit(":", 1)[1]
            t = tests.get((f["claim_ids"][0], param))
            if t:
                f["test"] = t

    # ---- 5. explanations
    targets = []
    for cid, comp in comparisons.items():
        if comp["verdict"] not in FAILING | {"INCONCLUSIVE", "NO_METRIC"} and not comp["better_than_reported"]:
            continue
        rid = (parsed.get(cid) or {}).get("run_id")
        evidence = [{"id": f["id"], "kind": "finding", "rule": f["rule"], "summary": f["summary"],
                     **({"test": f["test"]} if f.get("test") else {})}
                    for f in findings if cid in f["claim_ids"] or (not f["claim_ids"] and f["rule"] == "DEPS_UNPINNED")]
        evidence += [{"id": d["id"], "kind": "deviation", "summary": d["detail"]}
                     for d in exec_out.get("deviations", []) if rid and d.get("run_id") == rid]
        targets.append({"claim_id": cid, "claim": claims[cid], "comparison": comp, "evidence": evidence})
    if targets:
        log(f"Generating evidence-constrained explanations for {len(targets)} claim(s)")
    diagnoses = diagnose(targets, log)

    _persist(job_id, comparisons, findings, diagnoses)
    counts: dict[str, int] = {}
    for comp in comparisons.values():
        counts[comp["verdict"]] = counts.get(comp["verdict"], 0) + 1
    for cid, comp in comparisons.items():
        msg = f"{cid}: {comp['verdict']}"
        if comp.get("obtained") is not None:
            msg += (f" — obtained {comp['obtained']:.4g} vs reported {comp['reported']:g} "
                    f"(Δ {comp['abs_delta']:+.4g}, tolerance ±{comp['tolerance']:.4g})")
        log(msg, level="info" if comp["verdict"] == "REPRODUCED" else "warning")
    log(f"{len(findings)} finding(s): " + ", ".join(f"{f['id']} {f['rule']}" for f in findings[:12])
        + ("…" if len(findings) > 12 else ""))
    log("Verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), data={"counts": counts})
    return {"counts": counts, "comparisons": comparisons, "findings": [f["id"] for f in findings],
            "measurements": measurements,
            "hypothesis_tests": [{"claim_id": k[0], "param": k[1], **v} for k, v in tests.items()],
            "diagnosis_origin": next((d["origin"] for d in diagnoses.values()), None)}


# --------------------------------------------------------------- helpers

def _hps(raw: str) -> list[dict[str, Any]]:
    data = json.loads(raw or "[]")
    return data if isinstance(data, list) else []


def _docker_ok(image: Optional[str], log: Callable[..., None]) -> bool:
    if not image:
        return False
    try:
        sandbox.client()
        return True
    except sandbox.SandboxUnavailable as exc:
        log(str(exc), level="warning")
        return False


def _run_and_measure(job_id: str, run_id: str, kind: str, argv: list[str], overrides: dict[str, Any],
                     cids: list[str], claims: dict[str, dict[str, Any]], plans: dict[str, dict[str, Any]],
                     image: str, timeout_s: int, network: bool, log: Callable[..., None]) -> dict[str, Optional[float]]:
    run_dir = storage.runs_dir(job_id) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    res = sandbox.run_in_sandbox(image, argv, storage.repo_dir(job_id), run_dir / "outputs",
                                 timeout_s=timeout_s, network=network)
    (run_dir / "log.txt").write_text(res.log, encoding="utf-8")
    values: dict[str, Optional[float]] = {}
    evidence: dict[str, Any] = {}
    for cid in cids:
        if res.status != "ok":
            values[cid] = None
            continue
        found, _ = measure(res.log, res.outputs, run_dir / "outputs", claims[cid], plans[cid].get("metric_key"))
        values[cid] = found["value"] if found else None
        if found:
            evidence[cid] = {"value": found["value"], "source": found["source"], **found["evidence"]}
    with session() as s:
        s.merge(Run(job_id=job_id, id=run_id, plan_id=plans[cids[0]]["id"], claim_id=cids[0], kind=kind,
                    command=" ".join(argv), overrides=json.dumps(overrides), status=res.status,
                    exit_code=res.exit_code, seconds=res.seconds, log_path=f"runs/{run_id}/log.txt",
                    image_tag=image, started_at=utcnow(),
                    environment=json.dumps({"network": res.network, "claim_ids": cids, "argv": argv,
                                            "outputs": res.outputs, "metric_evidence": evidence})))
        s.commit()
    shown = ", ".join(f"{cid}={v:.4g}" if v is not None else f"{cid}=—" for cid, v in values.items())
    log(f"{run_id} ({kind}) {res.status} in {res.seconds:.1f}s: {shown}",
        level="info" if res.status == "ok" else "warning", data={"run_id": run_id})
    return values


def _hypothesis_reruns(job_id: str, groups: dict[str, list[str]], claims: dict, plans: dict, comparisons: dict,
                       measurements: dict, run_info: dict, image: str, timeout_s: int,
                       log: Callable[..., None]) -> dict[tuple[str, str], dict[str, Any]]:
    tests: dict[tuple[str, str], dict[str, Any]] = {}
    budget = MAX_HYPOTHESES_PER_JOB
    h = 0
    for rid, cids in groups.items():
        failing = [cid for cid in cids if comparisons[cid]["verdict"] in FAILING]
        if not failing or budget <= 0:
            continue
        candidates: dict[str, dict[str, Any]] = {}
        for cid in failing:
            for o in plans[cid].get("overrides", []):
                if o["status"] != "mismatch" or not o.get("testable"):
                    continue
                if o.get("repo_source") == "config":
                    log(f"{cid}: {o['param']} comes from a config file; a command-line override might be ignored, "
                        "so it is not tested automatically", level="warning")
                    continue
                candidates.setdefault(o["param"], o)
        for param, o in list(candidates.items())[:MAX_HYPOTHESES_PER_GROUP]:
            if budget <= 0:
                break
            budget -= 1
            h += 1
            argv = _with_flag(plans[cids[0]]["argv"], o["flag"], str(o["paper_value"]))
            n = len(measurements[cids[0]]["values"])
            log(f"Hypothesis H{h}: rerun {rid}'s command with {o['flag']} {o['paper_value']} (paper) instead of "
                f"{o['repo_value']} (repository), changing nothing else"
                + (f"; {n} runs to match the primary measurement" if n > 1 else ""))
            values: dict[str, list[float]] = {cid: [] for cid in cids}
            run_ids = []
            for k in range(1, n + 1):
                run_id = f"H{h}" if k == 1 else f"H{h}.{k}"
                run_ids.append(run_id)
                got = _run_and_measure(job_id, run_id, "hypothesis", argv, {o["flag"]: o["paper_value"]}, cids,
                                       claims, plans, image, timeout_s, run_info.get(rid, {}).get("network") == "on",
                                       log)
                for cid, v in got.items():
                    if v is not None:
                        values[cid].append(v)
            for cid in failing:
                comp = comparisons[cid]
                if not values[cid]:
                    tests[(cid, param)] = {"run_id": run_ids[0], "run_ids": run_ids, "override": {o["flag"]: o["paper_value"]},
                                           "supports_hypothesis": "inconclusive",
                                           "statement": "The rerun produced no value, so the hypothesis could not be tested."}
                    continue
                after_mean = statistics.fmean(values[cid])
                gap_before, gap_after = comp["abs_delta"], after_mean - comp["reported"]
                tol = comp["tolerance"]
                outcome, text = classify_hypothesis(param, gap_before, gap_after, tol)
                tests[(cid, param)] = {"run_id": run_ids[0], "run_ids": run_ids, "override": {o["flag"]: o["paper_value"]},
                                       "values": [round(v, 6) for v in values[cid]], "obtained_after": round(after_mean, 6),
                                       "gap_before": round(gap_before, 6), "gap_after": round(gap_after, 6),
                                       "tolerance": tol, "supports_hypothesis": outcome, "statement": text}
                log(f"H{h} for {cid}: {text}", level="info" if outcome == "supports" else "warning")
    return tests


def _with_flag(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    for i, tok in enumerate(out):
        if tok == flag and i + 1 < len(out):
            out[i + 1] = value
            return out
        if tok.startswith(flag + "="):
            out[i] = f"{flag}={value}"
            return out
    return out + [flag, value]


def _persist(job_id: str, comparisons: dict, findings: list, diagnoses: dict) -> None:
    with session() as s:
        for cid, c in comparisons.items():
            s.add(Comparison(job_id=job_id, claim_id=cid, reported=c["reported"], obtained=c["obtained"],
                             abs_delta=c["abs_delta"], rel_delta=c["rel_delta"], tolerance=c["tolerance"],
                             tolerance_basis=c["tolerance_basis"], seed_values=json.dumps(c["seed_values"]),
                             verdict=c["verdict"], better_than_reported=c["better_than_reported"],
                             run_status=c.get("run_status"), note=c["note"]))
        for f in findings:
            s.add(Finding(job_id=job_id, id=f["id"], rule=f["rule"], severity=f["severity"],
                          claim_id=f["claim_ids"][0] if f["claim_ids"] else None, claim_ids=json.dumps(f["claim_ids"]),
                          summary=f["summary"], impact=f["impact"], paper_evidence=json.dumps(f["paper_evidence"]),
                          repo_evidence=json.dumps(f["repo_evidence"]), run_evidence=json.dumps(f["run_evidence"]),
                          test=json.dumps(f["test"])))
        for cid, d in diagnoses.items():
            s.add(Diagnosis(job_id=job_id, claim_id=cid, hypotheses=json.dumps(d["hypotheses"]), origin=d["origin"]))
        s.commit()
