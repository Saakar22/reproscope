"""Shared fixtures: every test gets a fresh temporary data directory and database."""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from reproscope import config, db
from reproscope.events import bus


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REPROSCOPE_DATA_DIR", str(tmp_path / "data"))
    for var in ("GROQ_API_KEY", "LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL",
                "LLM_STRICT_JSON", "LLM_MAX_RETRIES"):
        monkeypatch.delenv(var, raising=False)
    config.override_settings(config.Settings())
    db.reset_engine()
    bus._next_seq.clear()
    yield tmp_path / "data"
    db.reset_engine()


@pytest.fixture()
def client():
    from reproscope.main import app
    with TestClient(app) as c:
        yield c


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_zip(files: dict[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


SAMPLE_PDF_PAGES = [
    "A Tiny MLP Baseline for Digits\nAbstract. We train a small MLP.",
    "Table 1: Results\nMLP (64 hidden) 97.2\nWe use a learning rate of 0.001.",
]
SAMPLE_REPO = {
    "tiny-repo/train.py": "import argparse\nap = argparse.ArgumentParser()\n"
                          "ap.add_argument('--lr', type=float, default=0.01)\n",
    "tiny-repo/README.md": "# tiny\n\n```bash\npython train.py\n```\n",
    "tiny-repo/requirements.txt": "scikit-learn\n",
}


@pytest.fixture()
def sample_pdf() -> bytes:
    return make_pdf(SAMPLE_PDF_PAGES)


@pytest.fixture()
def sample_zip() -> bytes:
    return make_zip(SAMPLE_REPO)


def wait_for_job(client: TestClient, job_id: str, timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "incomplete"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")
