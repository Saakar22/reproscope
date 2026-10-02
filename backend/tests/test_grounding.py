"""Grounding: quotes and numbers must really be in the PDF text."""
from __future__ import annotations

import pytest

from reproscope.grounding import ground_claim, ground_text, normalise, number_forms

PAGES = [normalise(p) for p in [
    "Introduction. Prior work reached 88.0% on this task.",
    "Table 2: Test accuracy (%). MLP (64 hidden) 97.2 ± 0.3 CNN 98.41",
    "We train with Adam at a learning‐rate of 0.001 for 200 epochs. Macro-F1 is 0.954.",
]]


def test_normalise_unifies_dashes_and_hyphenation():
    assert normalise("repro-\nducible  results−ok") == "reproducible results-ok"
    assert normalise("97.2 +/- 0.3") == "97.2 ± 0.3"


def test_number_forms_cover_percent_and_fraction():
    assert {"97.2", "97.20", "0.972", ".972"} <= number_forms(97.2)
    assert {"0.954", ".954", "95.4"} <= number_forms(0.954)
    assert "97" not in number_forms(97.2)                      # no rounding allowed


def test_grounded_on_cited_page():
    g = ground_claim("MLP (64 hidden) 97.2", 97.2, 2, PAGES)
    assert g["grounded"] and g["page_found"] == 2 and g["number_found"]


def test_off_by_one_page_is_tolerated():
    g = ground_claim("MLP (64 hidden) 97.2", 97.2, 3, PAGES)
    assert g["grounded"] and g["page_found"] == 2


def test_wrong_page_found_elsewhere():
    pages = PAGES + [normalise("filler")] * 5
    g = ground_claim("Macro-F1 is 0.954", 0.954, 8, pages)
    assert g["grounded"] and g["page_found"] == 3


def test_fraction_printed_as_percent_matches():
    assert ground_claim("Macro-F1 is 0.954", 95.4, 3, PAGES)["grounded"]


@pytest.mark.parametrize("quote,value,page", [
    ("MLP (64 hidden) 97.9", 97.9, 2),               # number not in the paper
    ("Transformer reaches 99.1", 99.1, 2),           # invented quote and number
    ("CNN 98.4", 98.4, 2),                           # 98.4 only appears inside 98.41
])
def test_hallucinated_claims_are_not_grounded(quote, value, page):
    assert not ground_claim(quote, value, page, PAGES)["grounded"]


def test_related_work_number_on_other_page_needs_matching_quote():
    g = ground_claim("our method achieves 88.0%", 88.0, 2, PAGES)
    assert not g["grounded"]


def test_ground_text_for_hyperparameters():
    assert ground_text("learning-rate of 0.001", PAGES)
    assert not ground_text("learning rate of 0.01 with cosine decay", PAGES)
