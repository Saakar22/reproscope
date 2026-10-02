"""Tolerances and reproducibility verdicts (pure functions; every choice is explained)."""
from __future__ import annotations

import statistics
from typing import Any, Optional

# Metrics on a 0–100 / 0–1 scale where "within 1 percentage point" is the conventional bar.
POINT_FAMILIES = {"accuracy", "balanced_accuracy", "f1", "precision", "recall", "auc"}
RELATIVE_DEFAULT = 0.03
# Deviations that make a definitive failure verdict unfair (the run was not the full experiment).
MATERIAL_DEVIATIONS = {"timeout", "reduced_budget"}

TOLERANCE_POLICY = {
    "reported_std": "2 × the standard deviation the paper reports for that value",
    "accuracy-like (accuracy, F1, precision, recall, AUC)": "1 percentage point (0.01 on a 0–1 scale) when the paper gives no std",
    "other metrics": "3% of the reported value when the paper gives no std",
    "run-to-run spread": "for scripts without a fixed seed: widened to 2 × the std of our repeated runs, and the mean is compared",
    "verdicts": "REPRODUCED ≤ 1× tolerance · PARTIAL ≤ 3× · NOT_REPRODUCED beyond; INCONCLUSIVE if the run was cut short",
}


def tolerance(claim: dict[str, Any], family: Optional[str], values: list[float]) -> tuple[float, str]:
    reported = float(claim["reported_value"])
    std = claim.get("reported_std")
    if std:
        tol, basis = 2 * float(std), f"2 × reported std ({std:g})"
    elif family in POINT_FAMILIES:
        if claim.get("unit") == "fraction" or (claim.get("unit") == "raw" and abs(reported) <= 1):
            tol, basis = 0.01, f"{family} default: 1 percentage point (0.01)"
        else:
            tol, basis = 1.0, f"{family} default: 1 percentage point"
    else:
        tol, basis = max(abs(reported) * RELATIVE_DEFAULT, 1e-12), f"default: 3% of the reported value"
    if len(values) >= 2:
        spread = 2 * statistics.stdev(values)
        if spread > tol:
            tol, basis = spread, f"{basis}, widened to 2 × std of our {len(values)} runs ({spread:.4g})"
    return tol, basis


def verdict(claim: dict[str, Any], values: list[float], family: Optional[str],
            deviation_kinds: set[str]) -> dict[str, Any]:
    """Compare the mean of our measured values with the reported value."""
    reported = float(claim["reported_value"])
    obtained = statistics.fmean(values)
    tol, basis = tolerance(claim, family, values)
    delta = obtained - reported
    rel = delta / abs(reported) if reported else None
    higher_better = claim.get("higher_is_better", True)
    improvement = delta if higher_better else -delta
    better = improvement > tol
    if abs(delta) <= tol:
        v = "REPRODUCED"
    elif abs(delta) <= 3 * tol:
        v = "PARTIAL"
    else:
        v = "NOT_REPRODUCED"
    note = None
    if better:
        note = ("The obtained result is better than reported beyond tolerance. That is not evidence of "
                "reproduction; check for a different setup, data leakage or a reporting error.")
    material = deviation_kinds & MATERIAL_DEVIATIONS
    if v != "REPRODUCED" and material:
        note = (f"Run was not the full experiment ({', '.join(sorted(material))}); "
                "no definitive failure verdict is given.")
        v = "INCONCLUSIVE"
    if len(values) > 1:
        extra = f"Mean of {len(values)} runs (no fixed seed): " + ", ".join(f"{x:.4g}" for x in values) + "."
        note = f"{extra} {note}" if note else extra
    return {"reported": reported, "obtained": round(obtained, 6), "abs_delta": round(delta, 6),
            "rel_delta": None if rel is None else round(rel, 6), "tolerance": round(tol, 6),
            "tolerance_basis": basis, "seed_values": [round(x, 6) for x in values], "verdict": v,
            "better_than_reported": better, "note": note}


def classify_hypothesis(change: str, gap_before: float, gap_after: float, tol: float) -> tuple[str, str]:
    """Outcome + carefully worded statement for a one-change rerun. Never claims proof."""
    if abs(gap_before) <= tol:
        # Nothing to explain: the original result was already within tolerance.
        if abs(gap_after) <= tol:
            return "not_applicable", (f"The original result was already within tolerance (gap {gap_before:+.4g}, "
                                      f"±{tol:.4g}). With {change} the gap is {gap_after:+.4g}, also within "
                                      "tolerance, so this change does not alter the verdict.")
        return "not_applicable", (f"The original result was already within tolerance (gap {gap_before:+.4g}, "
                                  f"±{tol:.4g}). With {change} the gap becomes {gap_after:+.4g}, outside "
                                  "tolerance: the repository's value fits the reported number better than this one.")
    if abs(gap_after) <= tol and abs(gap_after) < abs(gap_before):
        return "supports", (f"Changing only {change} moved the gap from {gap_before:+.4g} to {gap_after:+.4g} "
                            f"(tolerance ±{tol:.4g}). This supports, but does not prove, that {change} explains "
                            "the difference.")
    if abs(gap_after) < abs(gap_before) * 0.5:
        return "partially_supports", (f"Changing only {change} narrowed the gap from {gap_before:+.4g} to "
                                      f"{gap_after:+.4g}, but not within tolerance (±{tol:.4g}); it may be one "
                                      "of several factors.")
    return "does_not_support", (f"Changing only {change} did not close the gap ({gap_before:+.4g} → "
                                f"{gap_after:+.4g}); {change} alone does not explain the difference.")


def simple(claim: dict[str, Any], verdict_name: str, note: str, run_status: Optional[str] = None) -> dict[str, Any]:
    return {"reported": float(claim["reported_value"]), "obtained": None, "abs_delta": None, "rel_delta": None,
            "tolerance": None, "tolerance_basis": None, "seed_values": [], "verdict": verdict_name,
            "better_than_reported": False, "run_status": run_status, "note": note}
