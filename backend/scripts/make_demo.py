"""Build the demo inputs and (optionally) submit them to a running ReproScope backend.

    python scripts/make_demo.py                 # writes demo/demo_paper.pdf + demo/planted_repo.zip
    python scripts/make_demo.py --submit        # ...and starts an analysis via the API

The paper is a SYNTHETIC demo document written for testing ReproScope; every page says so.
Its code is tests/fixtures/planted_repo, which contains deliberately planted discrepancies
(see tests/fixtures/planted_truth.json).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import fitz  # PyMuPDF
import httpx

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
OUT = ROOT / "demo"
FIXTURE = BACKEND / "tests" / "fixtures" / "planted_repo"
FOOTER = "Synthetic demo paper for ReproScope testing. Not a real publication."

PAGES = [
    """Small Networks, Solid Baselines: An MLP Study on Handwritten Digits

Abstract
We revisit simple baselines for handwritten digit recognition. A one-hidden-layer
multilayer perceptron reaches 97.2% test accuracy on the scikit-learn digits dataset,
outperforming logistic regression. We release code for all experiments.

1 Introduction
Prior work using kernel methods reported 98.9% accuracy on a related benchmark [3].
We focus on models that train in seconds on a laptop CPU.""",
    """2 Experimental setup
We use the scikit-learn digits dataset (1,797 images of 8x8 pixels, 10 classes).
Pixel values are scaled to [0, 1] by dividing by 16. We use an 80/20 train/test split.
The MLP has 64 hidden units with ReLU activations and is trained with Adam at a
learning rate of 0.001 for 200 epochs. Logistic regression uses the lbfgs solver.

3 Results
Table 1: Test results on digits (mean over 5 runs).

Model                     Accuracy (%)     Macro-F1
Logistic regression       96.1 ± 0.3       0.960
MLP (64 hidden)           97.2 ± 0.4       0.971
MLP (128 hidden)          97.5 ± 0.3       0.974""",
    """4 Discussion
Doubling the hidden layer to 128 units gives a small gain of 0.3 points.
Training the MLP takes under 10 seconds on a single CPU core.

References
[3] A. Author. Kernel methods for digits. 2015.""",
]


def build() -> tuple[Path, Path]:
    OUT.mkdir(exist_ok=True)
    pdf_path = OUT / "demo_paper.pdf"
    doc = fitz.open()
    for i, text in enumerate(PAGES, 1):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(60, 60, 550, 760), text, fontsize=10, fontname="helv")
        page.insert_text((60, 805), f"{FOOTER}  Page {i}", fontsize=7, color=(0.45, 0.45, 0.45))
    doc.set_metadata({"title": "", "subject": FOOTER})
    doc.save(pdf_path)
    zip_base = OUT / "planted_repo"
    shutil.make_archive(str(zip_base), "zip", root_dir=FIXTURE.parent, base_dir=FIXTURE.name)
    return pdf_path, zip_base.with_suffix(".zip")


def submit(api: str, pdf: Path, repo_zip: Path) -> str:
    with pdf.open("rb") as p, repo_zip.open("rb") as z:
        r = httpx.post(f"{api}/api/jobs", timeout=60,
                       files={"pdf": (pdf.name, p, "application/pdf"), "repo_zip": (repo_zip.name, z, "application/zip")},
                       data={"options": json.dumps({"max_claims": 10, "enable_hypothesis_reruns": True,
                                                    "main_results_only": False, "timeout_s": 600})})
    r.raise_for_status()
    return r.json()["id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--submit", action="store_true", help="start an analysis on a running backend")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    pdf, repo_zip = build()
    print(f"wrote {pdf.relative_to(ROOT)} and {repo_zip.relative_to(ROOT)}")
    if args.submit:
        try:
            job_id = submit(args.api, pdf, repo_zip)
        except httpx.HTTPError as exc:
            print(f"could not submit to {args.api}: {exc}", file=sys.stderr)
            return 1
        print(f"started job {job_id}: open http://localhost:5173/jobs/{job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
