"""Deterministic discrepancy checks. Each finding carries the evidence it rests on:
paper page + quote, repository file + line + snippet, and/or run-log evidence."""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .params import canonical, effective_values
from .sandbox import package_for_module

IMPORTANT_TRAINING = ["lr", "batch_size", "epochs", "weight_decay", "dropout", "optimizer"]
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Ctx:
    repo: Path
    pages: list[str]
    claims: dict[str, dict[str, Any]]              # claim id -> claim (hyperparameters parsed)
    plans: list[dict[str, Any]]
    card: dict[str, Any]
    exec_out: dict[str, Any]
    comparisons: dict[str, dict[str, Any]]
    measurements: dict[str, dict[str, Any]] = field(default_factory=dict)   # claim -> {values, runs}

    @property
    def facts(self) -> list[dict[str, Any]]:
        return self.card.get("facts", [])

    def snippet(self, file: Optional[str], line: Optional[int]) -> Optional[str]:
        if not file or not line:
            return None
        try:
            lines = (self.repo / file).read_text(encoding="utf-8", errors="replace").splitlines()
            return lines[line - 1].strip()[:200] if 0 < line <= len(lines) else None
        except OSError:
            return None

    def repo_ev(self, file: Optional[str], line: Optional[int]) -> list[dict[str, Any]]:
        return [{"file": file, "line": line, "snippet": self.snippet(file, line)}] if file else []

    def paper_find(self, pattern: str) -> Optional[dict[str, Any]]:
        rx = re.compile(pattern, re.I)
        for pno, text in enumerate(self.pages, 1):
            for line in text.splitlines():
                if rx.search(line):
                    return {"page": pno, "quote": line.strip()[:300]}
        return None

    def planned_scripts(self) -> dict[str, list[str]]:
        """script -> claim ids planned to run it."""
        out: dict[str, list[str]] = {}
        for p in self.plans:
            if p.get("script"):
                out.setdefault(p["script"], []).append(p["claim_id"])
        return out

    def reach(self, script: str) -> set[str]:
        """Files `script` itself can import (per-entry-point reach from the scan)."""
        entry = next((e for e in self.card.get("entrypoints", []) if e["file"] == script), None)
        return {script} | set(entry.get("reach", [])) if entry else {script}


def finding(rule: str, severity: str, claim_ids: list[str], summary: str, impact: str, *,
            paper: Optional[dict] = None, repo: Optional[list] = None, run: Optional[dict] = None,
            test: Optional[dict] = None, key: Optional[str] = None) -> dict[str, Any]:
    return {"rule": rule, "severity": severity, "claim_ids": claim_ids, "summary": summary, "impact": impact,
            "paper_evidence": paper, "repo_evidence": repo or [], "run_evidence": run, "test": test,
            "key": key or f"{rule}:{','.join(claim_ids)}:{summary}"}


# ------------------------------------------------------------------ rules

def param_mismatch(ctx: Ctx) -> list[dict[str, Any]]:
    out = []
    for p in ctx.plans:
        claim = ctx.claims.get(p["claim_id"])
        for o in p.get("overrides", []):
            if o["status"] != "mismatch":
                continue
            sev = "high" if o["param"] in ("lr", "epochs", "batch_size") else "medium"
            out.append(finding(
                "PARAM_MISMATCH", sev, [p["claim_id"]],
                f"{o['param']}: the paper states {o['paper_value']}, the repository uses {o['repo_value']} "
                f"({o['repo_source']})",
                f"The primary run used the repository's {o['param']}; a different {o['param']} can change "
                "training dynamics and the final metric.",
                paper={"page": claim["page"] if claim else None, "quote": o["paper_quote"]},
                repo=ctx.repo_ev(o["repo_file"], o["repo_line"]),
                key=f"PARAM_MISMATCH:{p['claim_id']}:{o['param']}"))
    return out


def param_unstated(ctx: Ctx) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for p in ctx.plans:
        if not p.get("script") or not p.get("argv"):
            continue
        claim = ctx.claims.get(p["claim_id"])
        stated = {canonical(h["name"]) for h in (claim or {}).get("hyperparameters", []) if h.get("verified")}
        eff = effective_values(p["script"], p["argv"][2:], ctx.facts, p.get("source_ref"))
        for param in IMPORTANT_TRAINING:
            if param in eff and param not in stated and eff[param].get("value") not in (None, "None"):
                g = grouped.setdefault((p["script"], param), {"claims": [], "eff": eff[param]})
                g["claims"].append(p["claim_id"])
    return [finding("PARAM_UNSTATED", "low", g["claims"],
                    f"The paper does not state {param} for {', '.join(g['claims'])}; {script} uses "
                    f"{g['eff']['value']} ({g['eff']['source']})",
                    "Readers cannot tell whether the paper's results used this value.",
                    repo=ctx.repo_ev(g["eff"].get("file"), g["eff"].get("line")),
                    key=f"PARAM_UNSTATED:{script}:{param}")
            for (script, param), g in grouped.items()]


def seed_not_set(ctx: Ctx) -> list[dict[str, Any]]:
    unseeded = set(ctx.card.get("summary", {}).get("unseeded_entrypoints", []))
    out = []
    for script, claims in ctx.planned_scripts().items():
        if script in unseeded:
            splits = [f for f in ctx.facts if f["kind"] == "split_call" and f["file"] == script
                      and not f.get("extra", {}).get("seeded")]
            out.append(finding(
                "SEED_NOT_SET", "medium", claims,
                f"No random seed is set in {script} or the modules it imports",
                "Data splits, initialisation and shuffling differ on every run, so a single run "
                "cannot confirm or refute a reported number.",
                repo=ctx.repo_ev(script, splits[0]["line"]) if splits else ctx.repo_ev(script, 1),
                key=f"SEED_NOT_SET:{script}"))
    return out


def nondeterminism(ctx: Ctx) -> list[dict[str, Any]]:
    out = []
    for cid, m in ctx.measurements.items():
        vals = m.get("values", [])
        if len(vals) < 2 or max(vals) == min(vals):
            continue
        cmp = ctx.comparisons.get(cid, {})
        spread = max(vals) - min(vals)
        tol_base = cmp.get("tolerance") or 0
        sev = "high" if tol_base and spread > tol_base else "medium"
        out.append(finding(
            "NONDETERMINISM", sev, [cid],
            f"Repeated identical runs gave different {ctx.claims[cid]['metric']} values: "
            + ", ".join(f"{v:.4g}" for v in vals) + f" (range {spread:.4g}, std {statistics.stdev(vals):.3g})",
            "The reported value may be one draw from a distribution; the comparison uses the mean and a "
            "tolerance widened by our measured spread.",
            run={"run_ids": m.get("runs", []), "values": vals}, key=f"NONDETERMINISM:{cid}"))
    return out


def deps_unpinned(ctx: Ctx) -> list[dict[str, Any]]:
    loose = [f for f in ctx.facts if f["kind"] == "requirement" and f.get("extra", {}).get("pin") in ("none", "range")]
    if not loose:
        return []
    names = ", ".join(f["name"] + (f" {f['value']}" if f.get("value") else "") for f in loose[:8])
    return [finding("DEPS_UNPINNED", "low", [],
                    f"{len(loose)} dependenc{'y is' if len(loose) == 1 else 'ies are'} not pinned to an exact "
                    f"version: {names}",
                    "Newer library versions can change defaults and numerics; this reproduction used the "
                    "versions recorded in the run environment.",
                    repo=[e for f in loose[:5] for e in ctx.repo_ev(f["file"], f["line"])],
                    key="DEPS_UNPINNED")]


def missing_dependency(ctx: Ctx) -> list[dict[str, Any]]:
    declared = {re.sub(r"[-_.]+", "-", f["name"].lower()) for f in ctx.facts if f["kind"] == "requirement"}
    added = {d["detail"] for d in ctx.exec_out.get("deviations", []) if d["kind"] == "added_dependency"}
    planned = ctx.planned_scripts()
    out = []
    seen: set[str] = set()
    for f in ctx.facts:
        if f["kind"] != "import" or f["name"] in seen:
            continue
        pkg = re.sub(r"[-_.]+", "-", package_for_module(f["name"]).lower())
        if pkg in declared or re.sub(r"[-_.]+", "-", f["name"].lower()) in declared:
            continue
        seen.add(f["name"])
        caused = any(f"'{f['name']}'" in d for d in added)
        sev = "medium" if caused or f["file"] in planned else "low"
        claims = planned.get(f["file"], [])
        out.append(finding(
            "MISSING_DEPENDENCY", sev, claims,
            f"{f['file']} imports '{f['name']}' but the dependency files do not list '{package_for_module(f['name'])}'"
            + ("; ReproScope had to install it" if caused else ""),
            "A fresh environment built from the repository's dependency list cannot run this code.",
            repo=ctx.repo_ev(f["file"], f["line"]), key=f"MISSING_DEPENDENCY:{f['name']}"))
    return out


def missing_artefact(ctx: Ctx) -> list[dict[str, Any]]:
    planned = ctx.planned_scripts()
    return [finding("MISSING_ARTEFACT", "high" if f["file"] in planned else "medium", planned.get(f["file"], []),
                    f"{f['file']} loads '{f['name']}', which is not in the repository",
                    "Results that depend on this checkpoint or data file cannot be reproduced from the release.",
                    repo=ctx.repo_ev(f["file"], f["line"]), key=f"MISSING_ARTEFACT:{f['name']}")
            for f in ctx.facts if f["kind"] == "checkpoint" and f.get("value") == "missing"]


SPLIT_PATTERNS = [
    (r"\b(\d{2})\s*/\s*(\d{2})\b[^.\n]{0,25}\b(split|train|test)", lambda m: int(m.group(2)) / 100
     if int(m.group(1)) + int(m.group(2)) == 100 else None),
    (r"\b(\d{1,2})\s*%\s*(?:of the (?:data|samples) )?(?:for|as|held out for)\s*(?:the )?test", lambda m: int(m.group(1)) / 100),
    (r"test[_ ]size\s*(?:=|of)\s*(0?\.\d+)", lambda m: float(m.group(1))),
]


def split_mismatch(ctx: Ctx) -> list[dict[str, Any]]:
    paper_frac, paper_ev = None, None
    for pattern, conv in SPLIT_PATTERNS:
        ev = ctx.paper_find(pattern)
        if ev:
            m = re.search(pattern, ev["quote"], re.I)
            frac = conv(m) if m else None
            if frac:
                paper_frac, paper_ev = frac, ev
                break
    if paper_frac is None:
        return []
    out = []
    for script, claims in ctx.planned_scripts().items():
        for f in ctx.facts:
            if f["kind"] != "split_call" or f["file"] != script:
                continue
            ts = f.get("extra", {}).get("kwargs", {}).get("test_size")
            if isinstance(ts, (int, float)) and not isinstance(ts, bool) and 0 < ts < 1 \
                    and abs(ts - paper_frac) > 1e-9:
                out.append(finding(
                    "SPLIT_MISMATCH", "medium", claims,
                    f"The paper's split holds out {paper_frac:.0%} for testing; {script} uses test_size={ts}",
                    "A different train/test split changes both the training data and the test set the "
                    "metric is computed on.",
                    paper=paper_ev, repo=ctx.repo_ev(f["file"], f["line"]), key=f"SPLIT_MISMATCH:{script}"))
    return out


def metric_definition(ctx: Ctx) -> list[dict[str, Any]]:
    from .stages.parse import family_of, variant_of
    out = []
    for p in ctx.plans:
        claim = ctx.claims.get(p["claim_id"])
        if not claim or not p.get("script"):
            continue
        want = variant_of(claim["metric"])
        fam = family_of(claim["metric"])
        if not want or fam not in ("f1", "precision", "recall"):
            continue
        for f in ctx.facts:
            if f["kind"] == "metric_call" and f["name"].startswith(fam) and f["file"] in ctx.reach(p["script"]):
                got = f.get("value")
                if got != want:
                    out.append(finding(
                        "METRIC_DEFINITION", "medium", [p["claim_id"]],
                        f"The paper reports {want}-{fam.upper() if fam == 'f1' else fam}; the code computes "
                        f"{f['name']}(average={got!r})" if got else
                        f"The paper reports {want}-{fam}; the code calls {f['name']} without an 'average' argument",
                        "Macro, micro and weighted averages give different numbers whenever classes differ in "
                        "size or difficulty, so the obtained value may not measure the same quantity.",
                        paper={"page": claim["page"], "quote": claim["quote"]},
                        repo=ctx.repo_ev(f["file"], f["line"]), key=f"METRIC_DEFINITION:{p['claim_id']}"))
    return out


LEAK_RE = re.compile(r"(\.(?:fit|partial_fit|fit_transform)\s*\([^)]*\b(?:X_?test|x_?test|Xte|test_x|X_te)\b"
                     r"|validation_data\s*=\s*\(?\s*\w*test|early_stopping[^\n]*test|eval_set\s*=\s*\[?\(?\s*\w*test)")


def eval_leakage(ctx: Ctx) -> list[dict[str, Any]]:
    out = []
    for script, claims in ctx.planned_scripts().items():
        for rel in sorted(ctx.reach(script)):
            try:
                lines = (ctx.repo / rel).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for no, line in enumerate(lines, 1):
                if LEAK_RE.search(line.split("#", 1)[0]):
                    out.append(finding(
                        "EVAL_LEAKAGE", "medium", claims,
                        f"Possible test-set use during training or model selection in {rel}:{no}",
                        "If test data influences training or early stopping, reported test metrics are optimistic. "
                        "This is a pattern match and needs human review.",
                        repo=ctx.repo_ev(rel, no), key=f"EVAL_LEAKAGE:{rel}:{no}"))
    return out


def gpu_only(ctx: Ctx) -> list[dict[str, Any]]:
    gpu_files = set(ctx.card.get("summary", {}).get("gpu_only_files", []))
    out = []
    for script, claims in ctx.planned_scripts().items():
        plans = [p for p in ctx.plans if p.get("script") == script]
        if script in gpu_files or any("GPU" in (p.get("reason") or "") for p in plans):
            fact = next((f for f in ctx.facts if f["kind"] == "gpu_only" and f["file"] == script), None)
            out.append(finding(
                "GPU_ONLY", "high", claims,
                f"{script} requires a CUDA GPU (no CPU fallback); it was not run in the CPU-only sandbox",
                "These claims cannot be checked without GPU hardware or a code change.",
                repo=ctx.repo_ev(script, fact["line"] if fact else None), key=f"GPU_ONLY:{script}"))
    return out


def reduced_budget(ctx: Ctx) -> list[dict[str, Any]]:
    out = []
    runs = {r["id"]: r for r in ctx.exec_out.get("runs", [])}
    for d in ctx.exec_out.get("deviations", []):
        if d["kind"] in ("timeout", "reduced_budget"):
            claims = runs.get(d.get("run_id") or "", {}).get("claim_ids", [])
            out.append(finding("REDUCED_BUDGET", "high", claims, d["detail"],
                               "The experiment did not run to completion, so its result is not a full reproduction.",
                               run={"run_id": d.get("run_id")}, key=f"REDUCED_BUDGET:{d['id']}"))
    return out


ALL_RULES = [param_mismatch, split_mismatch, metric_definition, seed_not_set, nondeterminism, gpu_only,
             reduced_budget, missing_artefact, missing_dependency, eval_leakage, deps_unpinned, param_unstated]


def run_all(ctx: Ctx) -> list[dict[str, Any]]:
    """Run every rule; de-duplicate; order by severity, then rule order; assign E1, E2, ..."""
    seen: dict[str, dict[str, Any]] = {}
    rank: dict[str, int] = {}
    for i, rule in enumerate(ALL_RULES):
        for f in rule(ctx):
            rank.setdefault(f["rule"], i)
            seen.setdefault(f["key"], f)
    ordered = sorted(seen.values(), key=lambda f: (SEVERITY_ORDER[f["severity"]], rank[f["rule"]], f["claim_ids"][:1]))
    for i, f in enumerate(ordered, 1):
        f["id"] = f"E{i}"
    return ordered
