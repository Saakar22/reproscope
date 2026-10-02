"""Stage 1 — Ingest.

Validates and stores the paper PDF and the repository (Git URL or ZIP), extracts
page text with page numbers preserved, and records the commit hash.
Validation helpers are also used synchronously by the upload endpoint so bad
inputs are rejected with a 4xx before a job is ever created.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse

import fitz  # PyMuPDF

from .. import storage
from ..config import get_settings

MAX_ZIP_ENTRIES = 20_000


class InputError(ValueError):
    """User-correctable input problem (maps to HTTP 400/413/422)."""


# ------------------------------------------------------------------- PDF

def validate_pdf(data: bytes, filename: str | None) -> int:
    """Raise InputError unless `data` is a readable PDF within limits. Returns page count."""
    settings = get_settings()
    if not data:
        raise InputError("The PDF file is empty.")
    if len(data) > settings.max_pdf_mb * 1024 * 1024:
        raise InputError(f"The PDF exceeds the {settings.max_pdf_mb} MB limit.")
    if filename and not filename.lower().endswith(".pdf"):
        raise InputError("The paper must be a .pdf file.")
    if not data.lstrip()[:5].startswith(b"%PDF-"):
        raise InputError("The uploaded file is not a PDF (missing %PDF header).")
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise InputError("The PDF is password-protected.")
            if doc.page_count == 0:
                raise InputError("The PDF has no pages.")
            return doc.page_count
    except InputError:
        raise
    except Exception as exc:  # corrupt file
        raise InputError(f"The PDF could not be opened: {exc}") from exc


def extract_pages(pdf_path: Path) -> tuple[list[str], dict[str, Any]]:
    """Return (page_texts, metadata). Index 0 is page 1."""
    with fitz.open(pdf_path) as doc:
        pages = [page.get_text("text") for page in doc]
        meta = dict(doc.metadata or {})
    return pages, meta


def guess_title(pages: list[str], meta: dict[str, Any]) -> str | None:
    title = (meta.get("title") or "").strip()
    if 4 <= len(title) <= 300 and not title.lower().endswith((".pdf", ".tex", ".dvi")):
        return title
    for line in (pages[0] if pages else "").splitlines():
        line = line.strip()
        if 8 <= len(line) <= 200 and not re.match(r"^(arxiv|preprint|\d)", line, re.I):
            return line
    return None


# ------------------------------------------------------------------- Git

GIT_URL_RE = re.compile(r"^/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?(\.git)?/?$")


def validate_git_url(url: str) -> str:
    """Accept only https://<allowed host>/<owner>/<repo>[.git]. Returns the normalised URL."""
    url = (url or "").strip()
    if not url:
        raise InputError("Repository URL is empty.")
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise InputError("Repository URL must use https://")
    host = (parsed.hostname or "").lower()
    allowed = get_settings().allowed_git_hosts
    if host not in allowed:
        raise InputError(f"Repository host '{host}' is not allowed (allowed: {', '.join(allowed)}).")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.port:
        raise InputError("Repository URL must not contain credentials, a port, a query or a fragment.")
    if not GIT_URL_RE.match(parsed.path):
        raise InputError("Repository URL must look like https://github.com/<owner>/<repo>")
    return f"https://{host}{parsed.path.rstrip('/')}"


def clone_repo(url: str, dest: Path) -> str:
    """Shallow-clone `url` into `dest` and return the HEAD commit hash."""
    settings = get_settings()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1",
           "GIT_ASKPASS": "echo"}
    cmd = ["git", "-c", "core.symlinks=false", "-c", "protocol.file.allow=never",
           "clone", "--depth", "1", "--no-tags", "--single-branch", "--", url, str(dest)]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=settings.git_clone_timeout_s)
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise InputError(f"git clone timed out after {settings.git_clone_timeout_s}s") from exc
    if proc.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        msg = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["unknown error"]
        raise InputError(f"git clone failed: {msg[0]}")
    return git_head(dest) or "unknown"


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        import git  # GitPython
        return git.Repo(path).head.commit.hexsha
    except Exception:
        return None


# ------------------------------------------------------------------- ZIP

def validate_zip(data: bytes, filename: str | None) -> None:
    settings = get_settings()
    if not data:
        raise InputError("The repository ZIP is empty.")
    if len(data) > settings.max_zip_mb * 1024 * 1024:
        raise InputError(f"The repository ZIP exceeds the {settings.max_zip_mb} MB limit.")
    if filename and not filename.lower().endswith(".zip"):
        raise InputError("The repository archive must be a .zip file.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            _check_zip_members(zf)
    except zipfile.BadZipFile as exc:
        raise InputError("The repository archive is not a valid ZIP file.") from exc


def _check_zip_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    settings = get_settings()
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_ENTRIES:
        raise InputError(f"The ZIP has too many entries (>{MAX_ZIP_ENTRIES}).")
    total = 0
    for info in infos:
        name = info.filename
        pure = PurePosixPath(name.replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts or re.match(r"^[A-Za-z]:", name):
            raise InputError(f"Unsafe path in ZIP: {name!r}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise InputError(f"Symbolic links are not allowed in the ZIP: {name!r}")
        total += info.file_size
    if total > settings.max_repo_unpacked_mb * 1024 * 1024:
        raise InputError(f"The unpacked repository exceeds {settings.max_repo_unpacked_mb} MB.")
    return infos


def unpack_zip(zip_path: Path, dest: Path) -> None:
    """Extract safely; if everything sits in one top-level folder, use that folder as the root."""
    staging = dest.with_name(dest.name + "_unzip")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in _check_zip_members(zf):
            target = storage.safe_join(staging, info.filename.replace("\\", "/"))
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    entries = [p for p in staging.iterdir() if p.name not in ("__MACOSX", ".DS_Store")]
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(root), str(dest))
    shutil.rmtree(staging, ignore_errors=True)


# ----------------------------------------------------------------- stage

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}


def repo_stats(root: Path) -> dict[str, Any]:
    files = py = size = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            files += 1
            if name.endswith(".py"):
                py += 1
            try:
                size += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return {"files": files, "python_files": py, "size_bytes": size}


def run(job_id: str, job: dict[str, Any], log: Callable[..., None]) -> dict[str, Any]:
    """Execute the ingest stage for an already-saved upload. Returns the stage output."""
    pdf = storage.paper_path(job_id)
    pages, meta = extract_pages(pdf)
    storage.pages_path(job_id).write_text(json.dumps(pages), encoding="utf-8")
    title = guess_title(pages, meta)
    empty_pages = [i + 1 for i, text in enumerate(pages) if not text.strip()]
    log(f"Extracted text from {len(pages)} page(s)"
        + (f"; {len(empty_pages)} page(s) have no text layer" if empty_pages else ""),
        data={"pages": len(pages), "title": title})
    if len(empty_pages) == len(pages):
        raise InputError("The PDF has no extractable text (scanned images?). OCR is not supported.")

    dest = storage.repo_dir(job_id)
    if job["repo_source"] == "git":
        log(f"Cloning {job['repo_url']} (shallow)")
        commit = clone_repo(job["repo_url"], dest)
    else:
        log("Unpacking repository ZIP")
        unpack_zip(storage.uploads_dir(job_id) / "repo.zip", dest)
        commit = git_head(dest)
    stats = repo_stats(dest)
    log(f"Repository ready: {stats['files']} files, {stats['python_files']} Python files"
        + (f", commit {commit[:10]}" if commit else ", no git metadata"),
        data={**stats, "commit": commit})
    if stats["python_files"] == 0:
        log("No Python files found; later stages only support Python repositories", level="warning")

    return {
        "pdf": {"pages": len(pages), "title": title, "metadata_title": meta.get("title"),
                "chars_per_page": [len(p) for p in pages], "empty_pages": empty_pages},
        "repo": {"source": job["repo_source"], "url": job.get("repo_url"), "commit": commit,
                 **stats},
    }
