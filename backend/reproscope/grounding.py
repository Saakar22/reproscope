"""Deterministic verification of LLM-extracted claims against the PDF's own text.

A claim is *grounded* only if (a) its quote fuzzy-matches text on a page near the cited
page and (b) the reported number literally appears on that page in some printed form.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from rapidfuzz import fuzz

QUOTE_THRESHOLD = 80.0

_DASHES = dict.fromkeys(map(ord, "−‐‑‒–—﹣－"), "-")


def normalise(text: str) -> str:
    """Lower-case, unify dashes/±/ligatures, re-join hyphenated line breaks, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "").translate(_DASHES)
    text = text.replace("+/-", "±").replace("+-", "±")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)      # "repro-\nducible" -> "reproducible"
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def number_forms(value: float) -> set[str]:
    """Printed forms a reported value might take in the PDF (91.2 <-> 0.912, 91.20, .912)."""
    forms: set[str] = set()

    def add(v: float) -> None:
        for digits in range(0, 5):
            s = f"{v:.{digits}f}"
            if float(s) == round(v, digits) and abs(float(s) - v) < 1e-9:
                forms.add(s)
                if s.startswith("0."):
                    forms.add(s[1:])
                if s.startswith("-0."):
                    forms.add("-" + s[2:])
        forms.add(f"{v:g}")

    add(value)
    if 0 < abs(value) <= 1:
        add(round(value * 100, 6))
    elif 1 < abs(value) <= 100:
        add(round(value / 100, 8))
    return {f for f in forms if f not in ("0", "1", "-0")}


def _contains_number(page_norm: str, value: float) -> bool:
    for form in number_forms(value):
        # the form must not be part of a longer number (e.g. "91.2" inside "191.25")
        if re.search(rf"(?<![\d.]){re.escape(form)}(?![\d])", page_norm):
            return True
    return False


def ground_claim(quote: str, value: float, page: int, pages_norm: list[str]) -> dict[str, Any]:
    """Return {quote_score, number_found, page_found, grounded}. Pages are 1-based in `page`."""
    def score_on(idx: int) -> tuple[float, bool]:
        text = pages_norm[idx]
        return fuzz.partial_ratio(normalise(quote), text), _contains_number(text, value)

    candidates = [p - 1 for p in (page, page - 1, page + 1) if 1 <= p <= len(pages_norm)]
    best = (-1.0, False, None)
    for idx in candidates:
        s, n = score_on(idx)
        if (n, s) > (best[1], best[0]):
            best = (s, n, idx)
    # Cited page wrong? Search the whole paper; record where it was actually found.
    if not (best[0] >= QUOTE_THRESHOLD and best[1]):
        for idx in range(len(pages_norm)):
            if idx in candidates:
                continue
            s, n = score_on(idx)
            if s >= QUOTE_THRESHOLD and n and (n, s) > (best[1], best[0]):
                best = (s, n, idx)
    score, num_ok, idx = best
    return {
        "quote_score": round(max(score, 0.0), 1),
        "number_found": bool(num_ok),
        "page_found": None if idx is None else idx + 1,
        "grounded": bool(score >= QUOTE_THRESHOLD and num_ok),
    }


def ground_text(snippet: str, pages_norm: Iterable[str]) -> bool:
    """Is `snippet` (e.g. a hyperparameter quote) present, near-verbatim, anywhere in the paper?"""
    target = normalise(snippet)
    if len(target) < 3:
        return False
    return any(fuzz.partial_ratio(target, page) >= 90 for page in pages_norm)
