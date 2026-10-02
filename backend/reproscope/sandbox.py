"""Docker sandbox for untrusted repository code.

* Environment images are built once per (python version, dependency set) with network
  access for `pip install` only, and cached by a content hash.
* Every experiment runs in a fresh container: non-root user (writes only to /work and a
  size-capped /tmp), all capabilities dropped, no-new-privileges, CPU / memory / PID
  limits, a wall-clock timeout and, by default, NO network.
* The host filesystem is never mounted: the repository is copied in as a tar archive and
  outputs are copied back out the same way (with path, symlink and size checks).
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional

from .config import get_settings

SUPPORTED_PYTHONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]
WORKDIR = "/work"
MAX_OUTPUT_FILE = 50 * 1024 * 1024
MAX_OUTPUT_TOTAL = 300 * 1024 * 1024
TAR_SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}

# import name -> PyPI package, for the ModuleNotFoundError auto-fix
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python-headless", "sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow",
    "skimage": "scikit-image", "bs4": "beautifulsoup4", "Crypto": "pycryptodome", "dateutil": "python-dateutil",
    "attr": "attrs", "google.protobuf": "protobuf", "dotenv": "python-dotenv", "jwt": "pyjwt",
    "torch": "torch", "torchvision": "torchvision", "tensorflow": "tensorflow-cpu", "tf_keras": "tf-keras",
    "pytorch_lightning": "pytorch-lightning", "lightning": "lightning", "sentencepiece": "sentencepiece",
}
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

NETWORK_ERROR_RE = re.compile(
    r"(Temporary failure in name resolution|Name or service not known|Network is unreachable|"
    r"urlopen error|Failed to establish a new connection|Max retries exceeded with url|getaddrinfo failed|"
    r"Could not resolve host|ConnectionRefusedError|socket\.gaierror|No route to host)")
GPU_ERROR_RE = re.compile(
    r"(Torch not compiled with CUDA enabled|No CUDA GPUs are available|Found no NVIDIA driver|"
    r"CUDA driver version is insufficient|cudaGetDeviceCount|libcudart\.so|CUDA error: no kernel image|"
    r"Attempting to deserialize object on a CUDA device)")
MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")


class SandboxUnavailable(RuntimeError):
    """Docker isn't reachable; runs are reported as not run (never simulated)."""


class BuildFailed(RuntimeError):
    def __init__(self, message: str, log_tail: str) -> None:
        super().__init__(message)
        self.log_tail = log_tail


@dataclass
class EnvSpec:
    python: str
    requirements: list[str]
    needs_torch_cpu: bool
    python_reason: str

    @property
    def tag(self) -> str:
        body = "\n".join([self.python, *sorted(self.requirements), str(self.needs_torch_cpu)])
        return f"reproscope-env:{hashlib.sha256(body.encode()).hexdigest()[:16]}"


@dataclass
class RunResult:
    status: str                    # ok | error | timeout | not_runnable
    exit_code: Optional[int]
    seconds: float
    log: str
    outputs: list[dict[str, Any]] = field(default_factory=list)
    network: str = "none"
    failure: Optional[str] = None  # network | gpu | missing_module | timeout | None
    missing_module: Optional[str] = None


# ---------------------------------------------------------------- docker

def client():
    try:
        import docker
        c = docker.from_env(timeout=30)
        c.ping()
        return c
    except Exception as exc:  # docker not installed / daemon down
        raise SandboxUnavailable(f"Docker is not available: {str(exc).splitlines()[0][:200]}") from exc


# ------------------------------------------------------- environment spec

def choose_python(hints: list[str]) -> tuple[str, str]:
    """Pick a concrete supported Python version from scan hints like '3.10', '>=3.9', '==3.8.*'."""
    default = get_settings().sandbox_python
    for hint in hints:
        m = re.search(r"(>=|==|~=|<=|<|>|≥)?\s*(3)\.(\d{1,2})", hint or "")
        if not m:
            continue
        op, ver = m.group(1) or "==", f"3.{m.group(3)}"
        if op in ("==", "~=") or not op:
            if ver in SUPPORTED_PYTHONS:
                return ver, f"repository asks for Python {hint.strip()}"
            closest = min(SUPPORTED_PYTHONS, key=lambda v: abs(int(v.split('.')[1]) - int(m.group(3))))
            return closest, f"repository asks for Python {hint.strip()}, which the sandbox lacks; using {closest}"
        if op in (">=", "≥", ">"):
            ok = [v for v in SUPPORTED_PYTHONS if int(v.split(".")[1]) >= int(m.group(3)) + (op == ">")]
            pick = default if default in ok else (ok[0] if ok else SUPPORTED_PYTHONS[-1])
            return pick, f"repository requires Python {hint.strip()}"
        if op in ("<", "<="):
            ok = [v for v in SUPPORTED_PYTHONS if int(v.split(".")[1]) <= int(m.group(3)) - (op == "<")]
            return (ok[-1] if ok else SUPPORTED_PYTHONS[0]), f"repository requires Python {hint.strip()}"
    return default, "no Python version stated; using the sandbox default"


def requirement_lines(facts: Iterable[dict[str, Any]]) -> list[str]:
    """Turn scanned requirement facts into pip lines (exactly as the repo pinned them)."""
    lines: list[str] = []
    seen: set[str] = set()
    for f in facts:
        if f["kind"] != "requirement" or f.get("extra", {}).get("pin") == "other":
            continue
        name = f["name"]
        if name.lower() in seen or name.lower() in ("python", "pip", "setuptools", "wheel"):
            continue
        seen.add(name.lower())
        spec = (f.get("value") or "").replace(" ", "")
        if spec and spec[0].isdigit():
            spec = "==" + spec
        lines.append(f"{name}{spec}")
    return lines


def make_spec(facts: list[dict[str, Any]], extra_packages: Iterable[str] = ()) -> EnvSpec:
    hints = [f["value"] for f in facts if f["kind"] == "python_version" and f.get("value")]
    python, reason = choose_python(hints)
    reqs = requirement_lines(facts) + [p for p in extra_packages]
    names = {re.split(r"[<>=!~\[ ]", r, maxsplit=1)[0].lower() for r in reqs}
    return EnvSpec(python=python, requirements=sorted(set(reqs)), needs_torch_cpu=bool(names & {"torch", "torchvision",
                                                                                                "torchaudio"}),
                   python_reason=reason)


def dockerfile(spec: EnvSpec) -> str:
    index = f" --extra-index-url {CPU_TORCH_INDEX}" if spec.needs_torch_cpu else ""
    return f"""FROM python:{spec.python}-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 runner && mkdir -p {WORKDIR} && chown runner:runner {WORKDIR}
COPY requirements.txt /opt/requirements.txt
RUN if [ -s /opt/requirements.txt ]; then pip install{index} -r /opt/requirements.txt; fi \\
    && pip freeze > /opt/pip-freeze.txt
USER runner
WORKDIR {WORKDIR}
"""


def ensure_image(spec: EnvSpec, log: Callable[[str], None]) -> tuple[str, bool]:
    """Build (or reuse) the environment image. Returns (tag, reused). Network is used here only."""
    docker_client = client()
    try:
        docker_client.images.get(spec.tag)
        return spec.tag, True
    except Exception:
        pass
    with tempfile.TemporaryDirectory(prefix="reproscope-build-") as ctx:
        Path(ctx, "Dockerfile").write_text(dockerfile(spec), encoding="utf-8")
        Path(ctx, "requirements.txt").write_text("\n".join(spec.requirements) + "\n", encoding="utf-8")
        tail: list[str] = []
        last_emit = 0.0
        try:
            stream = docker_client.api.build(path=ctx, tag=spec.tag, rm=True, forcerm=True, decode=True, pull=False)
            for chunk in stream:
                text = (chunk.get("stream") or chunk.get("status") or "").rstrip()
                if chunk.get("error"):
                    tail.append(chunk["error"])
                    raise BuildFailed(chunk["error"].strip()[:300], "\n".join(tail[-40:]))
                if text:
                    tail.append(text)
                    tail = tail[-200:]
                    # forward meaningful build lines, throttled
                    if (text.startswith(("Step ", "Successfully", "Collecting", "Installing", "ERROR"))
                            and time.monotonic() - last_emit > 0.5):
                        log(text[:200])
                        last_emit = time.monotonic()
        except BuildFailed:
            raise
        except Exception as exc:
            raise BuildFailed(f"image build failed: {exc}", "\n".join(tail[-40:])) from exc
    return spec.tag, False


def image_freeze(tag: str) -> list[str]:
    out = client().containers.run(tag, ["cat", "/opt/pip-freeze.txt"], remove=True, network_disabled=True,
                                  user="1000:1000")
    return [line for line in out.decode(errors="replace").splitlines() if line.strip()]


# ------------------------------------------------------------- tar I/O

def tar_directory(src: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in TAR_SKIP]
            for name in filenames:
                full = Path(dirpath) / name
                if full.is_symlink():
                    continue
                rel = full.relative_to(src).as_posix()
                info = tar.gettarinfo(str(full), arcname=rel)
                info.uid = info.gid = 1000
                info.uname = info.gname = "runner"
                info.mode = 0o755 if os.access(full, os.X_OK) else 0o644
                with open(full, "rb") as fh:
                    tar.addfile(info, fh)
            for d in dirnames:
                info = tarfile.TarInfo((Path(dirpath, d).relative_to(src)).as_posix())
                info.type, info.mode, info.uid, info.gid = tarfile.DIRTYPE, 0o755, 1000, 1000
                tar.addfile(info)
    return buf.getvalue()


def extract_outputs(chunks: Iterable[bytes], dest: Path, pristine: Path) -> list[dict[str, Any]]:
    """Safely extract the container's /work tar into `dest`; return files new or changed vs `pristine`."""
    data = b"".join(chunks)
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    changed: list[dict[str, Any]] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if parts and parts[0] == "work":
                parts = parts[1:]
            if not parts or ".." in parts or PurePosixPath(member.name).is_absolute():
                continue
            if not (member.isfile() or member.isdir()):
                continue                                   # no symlinks, devices, ...
            if any(p in TAR_SKIP for p in parts):
                continue
            rel = "/".join(parts)
            target = dest / rel
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.size > MAX_OUTPUT_FILE or total + member.size > MAX_OUTPUT_TOTAL:
                changed.append({"path": rel, "size": member.size, "skipped": "too large"})
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            content = fh.read()
            total += len(content)
            orig = pristine / rel
            if orig.is_file() and orig.read_bytes() == content:
                continue                                   # unchanged repo file: don't copy
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            changed.append({"path": rel, "size": len(content), "new": not orig.exists()})
    return changed


# --------------------------------------------------------------- running

def run_in_sandbox(image: str, argv: list[str], repo: Path, out_dir: Path, *, timeout_s: int,
                   network: bool = False, on_line: Optional[Callable[[str], None]] = None) -> RunResult:
    """Run `argv` (no shell) in a fresh locked-down container with a copy of `repo`."""
    settings = get_settings()
    docker_client = client()
    env = {"HOME": "/tmp", "MPLCONFIGDIR": "/tmp/mpl", "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg",
           "OMP_NUM_THREADS": str(max(1, int(settings.sandbox_cpus))), "PYTHONUNBUFFERED": "1",
           "SCIKIT_LEARN_DATA": "/tmp/sklearn_data", "TORCH_HOME": "/tmp/torch", "HF_HOME": "/tmp/hf"}
    container = docker_client.containers.create(
        image, argv, working_dir=WORKDIR, user="1000:1000", environment=env,
        network_disabled=not network, network_mode=None if network else "none",
        mem_limit=settings.sandbox_memory, memswap_limit=settings.sandbox_memory,
        nano_cpus=int(settings.sandbox_cpus * 1e9), pids_limit=256,
        # Root filesystem stays writable only where the non-root user owns files (/work);
        # /tmp is a size-capped tmpfs. Scripts must be able to write their own outputs.
        tmpfs={"/tmp": "rw,size=1g,mode=1777"},
        cap_drop=["ALL"], security_opt=["no-new-privileges"],
        labels={"reproscope": "run"}, detach=True)
    lines: list[str] = []
    t0 = time.monotonic()
    timed_out = threading.Event()
    try:
        container.put_archive(WORKDIR, tar_directory(repo))
        container.start()

        def watchdog() -> None:
            deadline = t0 + timeout_s
            while time.monotonic() < deadline:
                try:
                    container.reload()
                    if container.status != "running":
                        return
                except Exception:
                    return
                time.sleep(0.5)
            timed_out.set()
            try:
                container.kill()
            except Exception:
                pass

        threading.Thread(target=watchdog, daemon=True).start()
        buffer = ""
        for chunk in container.logs(stream=True, follow=True, stdout=True, stderr=True):
            buffer += chunk.decode(errors="replace")
            *complete, buffer = buffer.split("\n")
            for line in complete:
                line = line.rstrip("\r")
                lines.append(line)
                if on_line:
                    on_line(line)
        if buffer:
            lines.append(buffer)
            if on_line:
                on_line(buffer)
        exit_code = container.wait(timeout=30)["StatusCode"]
        seconds = round(time.monotonic() - t0, 2)
        log_text = "\n".join(lines)
        try:
            chunks, _ = container.get_archive(WORKDIR)
            outputs = extract_outputs(chunks, out_dir, repo)
        except Exception:
            outputs = []
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass

    if timed_out.is_set():
        return RunResult("timeout", None, seconds, log_text, outputs, "on" if network else "none", "timeout")
    status, failure, missing = classify(exit_code, log_text, network)
    return RunResult(status, exit_code, seconds, log_text, outputs, "on" if network else "none", failure, missing)


def classify(exit_code: int, log_text: str, network: bool) -> tuple[str, Optional[str], Optional[str]]:
    """(status, failure kind, missing module) for a finished run, from its exit code and log."""
    if exit_code == 0:
        return "ok", None, None
    if m := MISSING_MODULE_RE.search(log_text):
        return "error", "missing_module", m.group(1)
    if GPU_ERROR_RE.search(log_text):
        return "not_runnable", "gpu", None
    if NETWORK_ERROR_RE.search(log_text) and not network:
        return "error", "network", None
    return "error", None, None


def package_for_module(module: str) -> str:
    if module in IMPORT_TO_PACKAGE:
        return IMPORT_TO_PACKAGE[module]
    top = module.split(".")[0]
    return IMPORT_TO_PACKAGE.get(top, top.replace("_", "-"))


def cleanup_run_containers() -> int:
    """Remove leftover ReproScope run containers (e.g. after a crash)."""
    try:
        c = client()
    except SandboxUnavailable:
        return 0
    removed = 0
    for container in c.containers.list(all=True, filters={"label": "reproscope=run"}):
        try:
            container.remove(force=True)
            removed += 1
        except Exception:
            pass
    return removed


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*TAR_SKIP), dirs_exist_ok=True)
