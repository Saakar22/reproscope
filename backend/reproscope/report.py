"""Report rendering. Building a Report from a job's stage outputs arrives with the
comparison stage; rendering (Markdown / JSON) works for any validated Report."""
from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined

from .schemas import Report

_env = Environment(loader=PackageLoader("reproscope", "templates"), autoescape=False,
                   undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if digits else text


def _signed(value: float | None) -> str:
    if value is None:
        return "—"
    return ("+" if value > 0 else "") + _fmt(value)


def _cell(text: str | None) -> str:
    """Make arbitrary text safe inside a Markdown table cell."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip() or "—"


_env.filters.update(fmt=_fmt, signed=_signed, cell=_cell)


def build_report(job_id: str) -> Report:
    """Assemble the report for a finished job from persisted rows + stage outputs."""
    import json
    from sqlmodel import select

    from . import storage
    from .db import Claim, Comparison, Deviation, Diagnosis, Finding, Job, RepoFact, Run, RunPlan, session, utcnow
    from .schemas import (Checklist, ClaimReport, ComparisonOut, DeviationOut, FindingOut, Grounding, Hypothesis,
                          PlanOut, RepoFactOut, ReportJob, ReportSummary, RunOut)
    from .reruns import list_reruns
    from .verdicts import TOLERANCE_POLICY

    with session() as s:
        job = s.get(Job, job_id)
        q = lambda model: s.exec(select(model).where(model.job_id == job_id)).all()  # noqa: E731
        claims, comps, findings = q(Claim), {c.claim_id: c for c in q(Comparison)}, q(Finding)
        devs, diags, runs, plans, facts = q(Deviation), {d.claim_id: d for d in q(Diagnosis)}, q(Run), q(RunPlan), q(RepoFact)
    plan_out = storage.read_stage(job_id, "plan") or {"plans": []}
    exec_out = storage.read_stage(job_id, "execute") or {"plan_runs": {}}
    card = storage.read_stage(job_id, "scan") or {"summary": {}}

    finding_claims = {f.id: json.loads(f.claim_ids or "[]") for f in findings}
    plan_runs = exec_out.get("plan_runs", {})
    plan_by_claim = {p["claim_id"]: p for p in plan_out["plans"]}
    run_claims = {r.id: json.loads(r.environment or "{}").get("claim_ids", [r.claim_id]) for r in runs}

    claim_reports = []
    for c in sorted(claims, key=lambda c: int(c.id[1:]) if c.id[1:].isdigit() else 0):
        comp = comps.get(c.id)
        hps = json.loads(c.hyperparameters or "[]")
        hp_dict = {h["name"]: h["value"] for h in hps if isinstance(h, dict) and h.get("verified")} \
            if isinstance(hps, list) else hps
        rid = plan_runs.get((plan_by_claim.get(c.id) or {}).get("id", ""))
        diag = diags.get(c.id)
        claim_reports.append(ClaimReport(
            id=c.id, experiment=c.experiment, dataset=c.dataset, split=c.split, metric=c.metric, unit=c.unit,
            higher_is_better=c.higher_is_better, reported_value=c.reported_value, reported_std=c.reported_std,
            source=c.source, page=c.page, quote=c.quote, hyperparameters=hp_dict, is_main_result=c.is_main_result,
            grounded=c.grounded, grounding=Grounding(**{k: v for k, v in json.loads(c.grounding or "{}").items()
                                                        if k in ("quote_score", "number_found", "page_found")}),
            origin=c.origin,
            comparison=ComparisonOut(reported=comp.reported, obtained=comp.obtained, abs_delta=comp.abs_delta,
                                     rel_delta=comp.rel_delta, tolerance=comp.tolerance,
                                     tolerance_basis=comp.tolerance_basis, seed_values=json.loads(comp.seed_values),
                                     verdict=comp.verdict, better_than_reported=comp.better_than_reported,
                                     run_status=comp.run_status, note=comp.note) if comp else None,
            run_ids=sorted([r for r, cl in run_claims.items() if c.id in cl], key=_run_sort),
            finding_ids=[fid for fid, cl in finding_claims.items() if c.id in cl],
            deviation_ids=[d.id for d in devs if d.run_id and d.run_id == rid],
            hypotheses=[Hypothesis(**h) for h in json.loads(diag.hypotheses)] if diag else [],
            diagnosis_origin=diag.origin if diag else None))

    def verdict_count(name: str) -> int:
        return sum(1 for c in claim_reports if c.comparison and c.comparison.verdict == name)

    summary = ReportSummary(
        total_claims=len(claim_reports), grounded_claims=sum(c.grounded for c in claim_reports),
        reproduced=verdict_count("REPRODUCED"), partial=verdict_count("PARTIAL"),
        not_reproduced=verdict_count("NOT_REPRODUCED"), inconclusive=verdict_count("INCONCLUSIVE"),
        not_run=verdict_count("NOT_RUN"), no_metric=verdict_count("NO_METRIC"),
        unverified=verdict_count("UNVERIFIED_CLAIM"),
        better_than_reported=sum(1 for c in claim_reports if c.comparison and c.comparison.better_than_reported))
    rules = {f.rule for f in findings}
    sm = card.get("summary", {})
    checklist = Checklist(
        seed_set=sm.get("seed_set_in_entrypoints"),
        deps_pinned=sm.get("deps_all_pinned"),
        all_hparams_stated=("PARAM_UNSTATED" not in rules) if plans else None,
        split_stated=any(c.split for c in claims) if claims else None)

    return Report(
        is_fixture=False, fixture_note=None, generated_at=utcnow(),
        job=ReportJob(id=job.id, paper_title=job.paper_title, pdf_filename=job.pdf_filename,
                      repo_source=job.repo_source, repo_url=job.repo_url, repo_filename=job.repo_filename,
                      repo_commit=job.repo_commit, created_at=job.created_at, status=job.status, llm_mode=job.llm_mode),
        summary=summary, checklist=checklist, claims=claim_reports,
        findings=[FindingOut(id=f.id, rule=f.rule, severity=f.severity, claim_id=f.claim_id,
                             claim_ids=finding_claims.get(f.id, []), summary=f.summary,
                             impact=f.impact, paper_evidence=json.loads(f.paper_evidence),
                             repo_evidence=json.loads(f.repo_evidence), run_evidence=json.loads(f.run_evidence),
                             test=json.loads(f.test)) for f in sorted(findings, key=lambda f: int(f.id[1:]))],
        deviations=[DeviationOut(id=d.id, run_id=d.run_id, kind=d.kind, detail=d.detail) for d in devs],
        runs=[RunOut(id=r.id, plan_id=r.plan_id, claim_id=r.claim_id, kind=r.kind, command=r.command,
                     overrides=json.loads(r.overrides or "{}"), status=r.status, exit_code=r.exit_code,
                     seconds=r.seconds, image_tag=r.image_tag,
                     environment={k: v for k, v in json.loads(r.environment or "{}").items()
                                  if k in ("python", "cpus", "memory", "network")},
                     log_available=bool(r.log_path) and (storage.job_dir(job_id) / r.log_path).is_file(),
                     metric_evidence=json.loads(r.environment or "{}").get("metric_evidence"))
              for r in sorted(runs, key=lambda r: _run_sort(r.id))],
        plans=[PlanOut(id=p.id, claim_id=p.claim_id, script=p.script, command=p.command,
                       source_of_command=p.source_of_command, expected_metric_location=p.expected_metric_location,
                       estimated_minutes=p.estimated_minutes, confidence=p.confidence,
                       paper_overrides={o["flag"]: o["paper_value"] for o in json.loads(p.paper_overrides or "[]")
                                        if o.get("status") == "mismatch" and o.get("flag")},
                       validation=json.loads(p.validation or "{}")) for p in plans],
        repo_facts=[RepoFactOut(id=f.id, kind=f.kind, name=f.name, value=f.value, file=f.file, line=f.line)
                    for f in facts if f.kind in ("entrypoint", "arg", "readme_cmd", "requirement", "seed_call",
                                                 "split_call", "metric_call", "gpu_only", "checkpoint")],
        tolerance_policy=TOLERANCE_POLICY,
        reruns=[r for r in list_reruns(job_id) if r["status"] in ("done", "failed")])


def _run_sort(run_id: str) -> tuple:
    head, _, tail = run_id.partition(".")
    return (head[0], int(head[1:]) if head[1:].isdigit() else 0, int(tail) if tail.isdigit() else 1)


def render_markdown(report: Report) -> str:
    findings = {f.id: f for f in report.findings}
    return _env.get_template("report.md.j2").render(r=report, findings=findings)


def export_filename(report: Report, ext: str) -> str:
    stem = "reproscope-SAMPLE-FIXTURE" if report.is_fixture else f"reproscope-{report.job.id}"
    return f"{stem}.{ext}"
