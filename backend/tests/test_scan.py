"""Stage 3: deterministic repository scan, checked against the planted repo's answer key."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from reproscope.stages.scan import scan_repo

FIXTURES = Path(__file__).parent / "fixtures"
TRUTH = json.loads((FIXTURES / "planted_truth.json").read_text())["scan"]


@pytest.fixture(scope="module")
def planted():
    card, facts = scan_repo(FIXTURES / "planted_repo")
    return card, facts


def of(facts, kind):
    return [f for f in facts if f.kind == kind]


def test_entrypoints(planted):
    card, _ = planted
    assert sorted(e["file"] for e in card["entrypoints"]) == TRUTH["entrypoints"]


def test_cli_defaults_with_lines(planted):
    _, facts = planted
    for script, expected in TRUTH["args"].items():
        got = {f.name: f.extra for f in of(facts, "arg") if f.file == script}
        assert set(got) == set(expected)
        for flag, default in expected.items():
            fact = next(f for f in of(facts, "arg") if f.file == script and f.name == flag)
            assert fact.value == (None if default is None else str(default))
            assert fact.line and fact.line > 1
    lr = next(f for f in of(facts, "arg") if f.name == "--lr")
    src = (FIXTURES / "planted_repo" / "train.py").read_text().splitlines()
    assert "--lr" in src[lr.line - 1]                                  # line number is exact


def test_readme_commands(planted):
    card, _ = planted
    cmds = [c["command"] for c in card["readme_commands"]]
    assert cmds == TRUTH["readme_commands"]
    assert all(c["script_exists"] for c in card["readme_commands"])
    assert "pip install -r requirements.txt" not in cmds


def test_requirements_and_pins(planted):
    card, facts = planted
    pins = {f.name: f.extra["pin"] for f in of(facts, "requirement")}
    assert pins == TRUTH["requirements"]
    assert card["summary"]["deps_all_pinned"] is False


def test_python_version_from_readme(planted):
    card, _ = planted
    assert card["summary"]["python_version"] == TRUTH["python_version"]


def test_seeds_and_reachability(planted):
    card, facts = planted
    assert sorted({f.file for f in of(facts, "seed_call")}) == TRUTH["seed_files"]
    # baseline.py is seeded, but that must not hide the unseeded main script (planted P3)
    assert card["summary"]["seed_set_in_entrypoints"] is False
    assert "train.py" in card["summary"]["unseeded_entrypoints"]
    seeded = {e["file"]: e["seeded"] for e in card["entrypoints"]}
    assert seeded["baseline.py"] is True and seeded["train.py"] is False


def test_split_and_metric_calls(planted):
    _, facts = planted
    splits = {f.file: f for f in of(facts, "split_call")}
    assert splits["train.py"].extra["kwargs"]["test_size"] == TRUTH["unseeded_split"]["test_size"]
    assert splits["train.py"].extra["seeded"] is False
    assert splits["baseline.py"].extra["seeded"] is True
    f1 = next(f for f in of(facts, "metric_call") if f.name == "f1_score")
    assert f1.file == "train.py" and f1.value == TRUTH["metric_average"]["average"]


def test_gpu_only_and_frameworks(planted):
    card, _ = planted
    assert card["summary"]["gpu_only_files"] == TRUTH["gpu_only_files"]
    assert {"scikit-learn", "pytorch"} <= set(card["summary"]["frameworks"])


def test_yaml_config_values_with_lines(planted):
    _, facts = planted
    cfg = {f.name: (f.value, f.line) for f in of(facts, "config_value") if f.file == "configs/mlp128.yaml"}
    assert cfg == {"hidden": ("128", 2), "lr": ("0.01", 3), "epochs": ("200", 4)}


def test_data_loaders_and_outputs(planted):
    _, facts = planted
    assert any("load_digits" in f.name for f in of(facts, "data_loader"))
    assert any(f.file == "train.py" and "json.dump" in f.name for f in of(facts, "output_write"))


# -------------------------------------------------------------- edge cases

def write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def test_packaging_formats_and_click(tmp_path):
    root = write(tmp_path, {
        "setup.py": "from setuptools import setup\nsetup(name='x', install_requires=['torch==2.1.0', 'tqdm'],\n"
                    "      python_requires='>=3.9')\n",
        "pyproject.toml": "[project]\nname='x'\nrequires-python='>=3.10'\ndependencies=['numpy==1.26.4']\n",
        "environment.yml": "name: e\ndependencies:\n  - python=3.9\n  - scipy=1.11.0\n  - pip:\n    - einops\n",
        "run_exp.py": "import click\n@click.command()\n@click.option('--lr', default=0.1)\ndef main(lr):\n    pass\n"
                      "if __name__ == '__main__':\n    main()\n",
        "conf/params.json": '{"optim": {"lr": 0.003, "batch_size": 32}}',
    })
    card, facts = scan_repo(root)
    pins = {f.name: f.extra["pin"] for f in of(facts, "requirement")}
    assert pins == {"torch": "exact", "tqdm": "none", "numpy": "exact", "scipy": "exact", "einops": "none"}
    versions = {f.name: f.value for f in of(facts, "python_version")}
    assert versions["python_requires"] == ">=3.9" and versions["requires-python"] == ">=3.10"
    lr = next(f for f in of(facts, "arg") if f.name == "--lr")
    assert lr.value == "0.1" and lr.file == "run_exp.py"
    cfg = {f.name: f.value for f in of(facts, "config_value") if f.file == "conf/params.json"}
    assert cfg == {"optim.lr": "0.003", "optim.batch_size": "32"}
    assert card["summary"]["cli_frameworks"] == ["click"]


def test_seed_in_imported_module_counts(tmp_path):
    root = write(tmp_path, {
        "train.py": "import argparse\nfrom utils.seeding import fix\nap = argparse.ArgumentParser()\n"
                    "ap.add_argument('--x', default=1)\nif __name__ == '__main__':\n    fix()\n",
        "utils/__init__.py": "",
        "utils/seeding.py": "import torch\ndef fix():\n    torch.manual_seed(0)\n",
        "other/unused.py": "import random\nrandom.seed(1)\n",
    })
    card, _ = scan_repo(root)
    assert card["summary"]["seed_set_in_entrypoints"] is True
    assert "utils/seeding.py" in card["reachable_from_entrypoints"]
    assert "other/unused.py" not in card["reachable_from_entrypoints"]


def test_unseeded_entrypoint(tmp_path):
    root = write(tmp_path, {"main.py": "import random\nprint(random.random())\nif __name__ == '__main__':\n    pass\n",
                            "lib/seed.py": "import numpy as np\nnp.random.seed(0)\n"})
    card, _ = scan_repo(root)
    assert card["summary"]["seed_set_in_entrypoints"] is False


def test_broken_python_missing_checkpoint_and_multiline_readme(tmp_path):
    root = write(tmp_path, {
        "legacy.py": "print 'python 2 code'\n",
        "eval.py": "import torch\nm = torch.load('checkpoints/best.pt')\nif __name__ == '__main__':\n    pass\n",
        "README.md": "```\n$ python eval.py \\\n    --split test\n```\n",
    })
    card, facts = scan_repo(root)
    assert any("legacy.py is not valid Python 3" in w for w in card["warnings"])
    ck = of(facts, "checkpoint")[0]
    assert ck.name == "checkpoints/best.pt" and ck.value == "missing" and ck.line == 2
    assert [c["command"] for c in card["readme_commands"]] == ["python eval.py --split test"]


def test_gpu_with_fallback_is_not_flagged(tmp_path):
    root = write(tmp_path, {"main.py": "import torch\ndev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
                                       "x = torch.zeros(1).cuda()\nif __name__ == '__main__':\n    pass\n"})
    card, _ = scan_repo(root)
    assert card["summary"]["gpu_only_files"] == []


def test_scan_never_executes_repo_code(tmp_path):
    marker = tmp_path / "EXECUTED"
    root = write(tmp_path / "repo", {
        "setup.py": f"open({str(marker)!r}, 'w').write('x')\n",
        "train.py": f"open({str(marker)!r}, 'w').write('x')\nif __name__ == '__main__':\n    pass\n",
    })
    scan_repo(root)
    assert not marker.exists()
