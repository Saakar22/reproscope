"""Stage 5 sandbox: environment spec, safe tar I/O, failure classification, and real
Docker isolation checks (skipped automatically when Docker isn't running)."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from reproscope import sandbox

# ----------------------------------------------------------- pure logic

@pytest.mark.parametrize("hints,expected", [
    (["3.10"], "3.10"), (["==3.9.*"], "3.9"), ([">=3.9"], "3.11"), ([">=3.12"], "3.12"),
    (["<3.10"], "3.9"), (["3.6"], "3.8"), ([], "3.11"), (["Python 3.13"], "3.12"),
])
def test_choose_python(hints, expected):
    assert sandbox.choose_python(hints)[0] == expected


def test_requirement_lines_keep_repo_pins():
    facts = [{"kind": "requirement", "name": "scikit-learn", "value": None, "extra": {"pin": "none"}},
             {"kind": "requirement", "name": "numpy", "value": ">=1.24", "extra": {"pin": "range"}},
             {"kind": "requirement", "name": "scipy", "value": "1.11.0", "extra": {"pin": "exact"}},
             {"kind": "requirement", "name": "git+https://x/y", "value": "vcs-or-option", "extra": {"pin": "other"}},
             {"kind": "requirement", "name": "numpy", "value": "==2.0", "extra": {"pin": "exact"}}]
    assert sandbox.requirement_lines(facts) == ["scikit-learn", "numpy>=1.24", "scipy==1.11.0"]


def test_spec_tag_is_stable_and_torch_uses_cpu_index():
    facts = [{"kind": "requirement", "name": "torch", "value": "==2.3.0", "extra": {"pin": "exact"}},
             {"kind": "python_version", "name": "x", "value": "3.10"}]
    a, b = sandbox.make_spec(facts), sandbox.make_spec(list(reversed(facts)))
    assert a.tag == b.tag and a.needs_torch_cpu
    assert sandbox.CPU_TORCH_INDEX in sandbox.dockerfile(a)
    assert "USER runner" in sandbox.dockerfile(a)
    assert sandbox.make_spec(facts, ["pyyaml"]).tag != a.tag


@pytest.mark.parametrize("module,package", [("cv2", "opencv-python-headless"), ("sklearn.metrics", "scikit-learn"),
                                            ("yaml", "pyyaml"), ("einops", "einops"), ("my_pkg", "my-pkg")])
def test_package_for_module(module, package):
    assert sandbox.package_for_module(module) == package


@pytest.mark.parametrize("code,log,net,expected", [
    (0, "ok", False, ("ok", None, None)),
    (1, "ModuleNotFoundError: No module named 'einops'", False, ("error", "missing_module", "einops")),
    (1, "AssertionError: Torch not compiled with CUDA enabled", False, ("not_runnable", "gpu", None)),
    (1, "urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>", False,
     ("error", "network", None)),
    (1, "urlopen error ... Temporary failure in name resolution", True, ("error", None, None)),
    (1, "ValueError: shapes do not match", False, ("error", None, None)),
])
def test_classify(code, log, net, expected):
    assert sandbox.classify(code, log, net) == expected


def _tar(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for info, data in members:
            if data is not None:
                info.size = len(data)
            t.addfile(info, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def test_extract_outputs_only_changed_files_and_rejects_unsafe(tmp_path):
    pristine = tmp_path / "repo"
    pristine.mkdir()
    (pristine / "train.py").write_text("print(1)")
    link = tarfile.TarInfo("work/evil_link")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    data = _tar([
        (tarfile.TarInfo("work/train.py"), b"print(1)"),                  # unchanged -> not copied
        (tarfile.TarInfo("work/results.json"), b'{"accuracy": 0.97}'),    # new output
        (tarfile.TarInfo("work/../escape.txt"), b"x"),                     # traversal -> skipped
        (link, None),                                                      # symlink -> skipped
        (tarfile.TarInfo("work/__pycache__/x.pyc"), b"\0"),                 # skipped dir
    ])
    out = tmp_path / "out"
    changed = sandbox.extract_outputs([data], out, pristine)
    assert changed == [{"path": "results.json", "size": 18, "new": True}]
    assert json.loads((out / "results.json").read_text()) == {"accuracy": 0.97}
    assert not (tmp_path / "escape.txt").exists() and not (out / "evil_link").exists()


def test_tar_directory_roundtrip(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.py").write_text("x = 1")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret")
    with tarfile.open(fileobj=io.BytesIO(sandbox.tar_directory(tmp_path))) as t:
        names = set(t.getnames())
        assert "a/b.py" in names and not any(n.startswith(".git") for n in names)
        assert all(m.uid == 1000 for m in t.getmembers())


# ------------------------------------------------------ real Docker checks

def _docker_ok() -> bool:
    try:
        sandbox.client()
        return True
    except sandbox.SandboxUnavailable:
        return False


docker = pytest.mark.skipif(not _docker_ok(), reason="Docker is not running")


@pytest.fixture(scope="module")
def bare_image():
    spec = sandbox.EnvSpec(python="3.10", requirements=[], needs_torch_cpu=False, python_reason="test")
    tag, _ = sandbox.ensure_image(spec, lambda line: None)
    return tag


def probe_repo(tmp_path: Path, code: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "probe.py").write_text(code)
    return repo


@docker
def test_runs_as_non_root_without_network_and_returns_outputs(tmp_path, bare_image):
    repo = probe_repo(tmp_path, (
        "import os, socket, json\n"
        "info = {'uid': os.getuid(), 'cwd': os.getcwd()}\n"
        "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=3); info['net'] = 'open'\n"
        "except OSError as e:\n    info['net'] = 'blocked'\n"
        "for target in ('/etc/reproscope_test', '/usr/local/x', '/work/ok.txt'):\n"
        "    try:\n        open(target, 'w').write('x'); info[target] = 'written'\n"
        "    except OSError:\n        info[target] = 'denied'\n"
        "info['host_users_dir'] = os.path.exists('/c/Users') or os.path.exists('/mnt/c')\n"
        "json.dump(info, open('probe.json', 'w')); print(json.dumps(info))\n"))
    res = sandbox.run_in_sandbox(bare_image, ["python", "probe.py"], repo, tmp_path / "out", timeout_s=60)
    assert res.status == "ok", res.log
    info = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert info["uid"] == 1000 and info["cwd"] == "/work"
    assert info["net"] == "blocked"
    assert info["/etc/reproscope_test"] == "denied" and info["/usr/local/x"] == "denied"
    assert info["/work/ok.txt"] == "written"
    assert info["host_users_dir"] is False                      # no host mounts
    assert {o["path"] for o in res.outputs} == {"probe.json", "ok.txt"}


@docker
def test_timeout_kills_the_run(tmp_path, bare_image):
    repo = probe_repo(tmp_path, "import time\nprint('start', flush=True)\ntime.sleep(60)\n")
    lines = []
    res = sandbox.run_in_sandbox(bare_image, ["python", "probe.py"], repo, tmp_path / "out", timeout_s=3,
                                 on_line=lines.append)
    assert res.status == "timeout" and res.failure == "timeout" and res.seconds < 20
    assert lines == ["start"]


@docker
def test_failures_are_reported_not_hidden(tmp_path, bare_image):
    repo = probe_repo(tmp_path, "import einops_not_installed_pkg\n")
    res = sandbox.run_in_sandbox(bare_image, ["python", "probe.py"], repo, tmp_path / "out", timeout_s=60)
    assert res.status == "error" and res.exit_code == 1
    assert res.failure == "missing_module" and res.missing_module == "einops_not_installed_pkg"
    assert "ModuleNotFoundError" in res.log


@docker
def test_argv_is_not_interpreted_by_a_shell(tmp_path, bare_image):
    repo = probe_repo(tmp_path, "import sys\nprint(sys.argv[1:])\n")
    res = sandbox.run_in_sandbox(bare_image, ["python", "probe.py", "$(id)", ";", "echo", "pwned"], repo,
                                 tmp_path / "out", timeout_s=60)
    assert res.status == "ok" and "['$(id)', ';', 'echo', 'pwned']" in res.log
