"""Stage 1: input validation, PDF text extraction and safe repository unpacking."""
from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest

from reproscope.stages import ingest
from reproscope.stages.ingest import InputError

from .conftest import SAMPLE_PDF_PAGES, make_pdf, make_zip


def test_validate_pdf_accepts_real_pdf(sample_pdf):
    assert ingest.validate_pdf(sample_pdf, "paper.pdf") == 2


@pytest.mark.parametrize("data,name,msg", [
    (b"", "p.pdf", "empty"),
    (b"hello world", "p.pdf", "not a PDF"),
    (b"%PDF-1.4 garbage", "p.pdf", "could not be opened"),
])
def test_validate_pdf_rejects_bad_input(data, name, msg):
    with pytest.raises(InputError, match=msg):
        ingest.validate_pdf(data, name)


def test_validate_pdf_rejects_wrong_extension(sample_pdf):
    with pytest.raises(InputError, match=".pdf"):
        ingest.validate_pdf(sample_pdf, "paper.docx")


def test_extract_pages_preserves_page_order(tmp_path: Path, sample_pdf):
    path = tmp_path / "p.pdf"
    path.write_bytes(sample_pdf)
    pages, _meta = ingest.extract_pages(path)
    assert len(pages) == 2
    assert "Abstract" in pages[0]
    assert "97.2" in pages[1] and "0.001" in pages[1]


def test_guess_title_uses_first_meaningful_line():
    assert ingest.guess_title(SAMPLE_PDF_PAGES, {}) == "A Tiny MLP Baseline for Digits"
    assert ingest.guess_title(SAMPLE_PDF_PAGES, {"title": "Real Title From Metadata"}) == \
        "Real Title From Metadata"


@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo",
    "https://github.com/owner/repo.git",
    "https://github.com/owner/repo/",
    "https://gitlab.com/some-org/some.repo",
])
def test_validate_git_url_accepts(url):
    assert ingest.validate_git_url(url).startswith("https://")


@pytest.mark.parametrize("url,msg", [
    ("http://github.com/owner/repo", "https"),
    ("git@github.com:owner/repo.git", "https"),
    ("https://evil.example.com/owner/repo", "not allowed"),
    ("https://user:pw@github.com/owner/repo", "credentials"),
    ("https://github.com/owner", "look like"),
    ("https://github.com/owner/repo/tree/main", "look like"),
    ("https://github.com/owner/repo?x=1", "query"),
    ("", "empty"),
])
def test_validate_git_url_rejects(url, msg):
    with pytest.raises(InputError, match=msg):
        ingest.validate_git_url(url)


def test_validate_zip_rejects_path_traversal():
    with pytest.raises(InputError, match="Unsafe path"):
        ingest.validate_zip(make_zip({"../evil.py": "x"}), "r.zip")
    with pytest.raises(InputError, match="Unsafe path"):
        ingest.validate_zip(make_zip({"/abs/evil.py": "x"}), "r.zip")


def test_validate_zip_rejects_symlinks():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("repo/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(InputError, match="Symbolic links"):
        ingest.validate_zip(buf.getvalue(), "r.zip")


def test_validate_zip_rejects_non_zip():
    with pytest.raises(InputError, match="not a valid ZIP"):
        ingest.validate_zip(b"not a zip", "r.zip")


def test_unpack_zip_strips_single_top_folder(tmp_path: Path, sample_zip):
    z = tmp_path / "r.zip"
    z.write_bytes(sample_zip)
    dest = tmp_path / "repo"
    ingest.unpack_zip(z, dest)
    assert (dest / "train.py").is_file()
    assert (dest / "README.md").is_file()
    stats = ingest.repo_stats(dest)
    assert stats["python_files"] == 1 and stats["files"] == 3


def test_unpack_zip_keeps_flat_layout(tmp_path: Path):
    z = tmp_path / "r.zip"
    z.write_bytes(make_zip({"a.py": "print(1)", "b/c.py": "print(2)"}))
    dest = tmp_path / "repo"
    ingest.unpack_zip(z, dest)
    assert (dest / "a.py").is_file() and (dest / "b" / "c.py").is_file()


def test_pdf_without_text_layer_is_rejected_by_stage(tmp_path: Path, monkeypatch):
    blank = make_pdf([""])
    from reproscope import storage
    job_id = "j_blank"
    storage.job_dir(job_id).mkdir(parents=True)
    storage.paper_path(job_id).write_bytes(blank)
    with pytest.raises(InputError, match="no extractable text"):
        ingest.run(job_id, {"repo_source": "zip"}, lambda *a, **k: None)
