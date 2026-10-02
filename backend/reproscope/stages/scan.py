"""Stage 3 — Repository scan (deterministic; no LLM, never imports or runs repo code).

Builds a RepoCard: entry points, README commands, CLI arguments with defaults,
config values, dependencies and pins, Python version hints, seed calls, dataset
loaders, checkpoint loads, output writes, metric/split calls and GPU-only code.
Every fact carries its file and line so later findings can cite it.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import yaml
from sqlmodel import delete

from .. import storage
from ..db import RepoFact, session

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env", ".mypy_cache",
             ".pytest_cache", ".idea", ".vscode", "site-packages", "dist", "build", ".tox", "wandb", "mlruns"}
MAX_PY_BYTES = 1_000_000
MAX_CONFIG_BYTES = 300_000
MAX_FILES = 5_000
CONFIG_EXT = {".yaml", ".yml", ".json", ".toml"}
CONFIG_SKIP = {"package.json", "package-lock.json", "tsconfig.json", "pyproject.toml", ".pre-commit-config.yaml",
               "environment.yml", "environment.yaml", "codecov.yml", "mkdocs.yml", ".readthedocs.yaml"}

ENTRY_NAME_RE = re.compile(r"^(train|main|run|eval|evaluate|test_model|experiment|reproduce|benchmark)\w*\.py$", re.I)
SEED_RE = re.compile(r"(torch\.manual_seed|torch\.cuda\.manual_seed(?:_all)?|np\.random\.seed|numpy\.random\.seed|"
                     r"(?<![\w.])random\.seed|tf\.random\.set_seed|tf\.set_random_seed|set_seed\(|seed_everything\(|"
                     r"np\.random\.default_rng\(\s*\w|random_state\s*=\s*(?!None)\w|PYTHONHASHSEED|"
                     r"use_deterministic_algorithms|cudnn\.deterministic\s*=\s*True)")
DATA_RE = re.compile(r"(torchvision\.datasets\.\w+|datasets\.\w+\(|load_dataset\(\s*['\"][\w/.-]+|"
                     r"sklearn\.datasets\.\w+|(?<![\w.])(?:fetch|load)_(?:digits|iris|wine|breast_cancer|diabetes|"
                     r"california_housing|openml|20newsgroups\w*|mnist|lfw_\w+|covtype|kddcup99|rcv1|olivetti_faces)\(|"
                     r"read_csv\(\s*['\"][^'\"]+|np\.load\(\s*['\"][^'\"]+|tfds\.load\(\s*['\"][\w/]+)")
CHECKPOINT_RE = re.compile(r"(torch\.load|load_state_dict|joblib\.load|pickle\.load|from_pretrained|"
                           r"keras\.models\.load_model|load_weights)\s*\(\s*(?:open\()?\s*['\"]([^'\"]+)['\"]")
OUTPUT_RE = re.compile(r"(json\.dump|\.to_csv\(|np\.save\w*\(|\.savefig\(|torch\.save\(|add_scalar\(|"
                       r"wandb\.log\(|mlflow\.log_metric|csv\.writer)")
METRIC_PRINT_RE = re.compile(r"(print|log\w*|info)\s*\(.*\b(acc|accuracy|f1|auc|rmse|mae|loss|bleu|precision|recall|"
                             r"score|ppl|perplexity)\b", re.I)
GPU_RE = re.compile(r"(\.cuda\(\)|device\s*=\s*['\"]cuda|torch\.device\(\s*['\"]cuda['\"]\s*\)|\.to\(\s*['\"]cuda['\"])")
GPU_FALLBACK_RE = re.compile(r"(cuda\.is_available\(\)|--no[-_]cuda|--device|--cpu|mps\.is_available)")
FRAMEWORKS = {"torch": "pytorch", "tensorflow": "tensorflow", "keras": "keras", "jax": "jax",
              "sklearn": "scikit-learn", "xgboost": "xgboost", "lightgbm": "lightgbm",
              "transformers": "transformers", "lightning": "lightning", "pytorch_lightning": "lightning"}
CONFIG_FRAMEWORKS = {"hydra": "hydra", "click": "click", "typer": "typer", "fire": "fire", "absl": "absl",
                     "gin": "gin", "omegaconf": "omegaconf", "argparse": "argparse"}


@dataclass
class Fact:
    kind: str
    name: str
    value: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


class Scanner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.facts: list[Fact] = []
        self.warnings: list[str] = []
        self.py_files: list[Path] = []
        self.trees: dict[str, ast.Module] = {}
        self.sources: dict[str, str] = {}

    # ------------------------------------------------------------ helpers
    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def add(self, kind: str, name: str, value: Any = None, file: Optional[str] = None,
            line: Optional[int] = None, **extra: Any) -> None:
        self.facts.append(Fact(kind, name, None if value is None else str(value), file, line, extra))

    def walk(self) -> list[Path]:
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
                                 or d == ".github")
            for name in sorted(filenames):
                files.append(Path(dirpath) / name)
                if len(files) >= MAX_FILES:
                    self.warnings.append(f"Repository has more than {MAX_FILES} files; scan truncated")
                    return files
        return files

    # --------------------------------------------------------------- scan
    def scan(self) -> dict[str, Any]:
        files = self.walk()
        self.py_files = [f for f in files if f.suffix == ".py"]
        notebooks = [self.rel(f) for f in files if f.suffix == ".ipynb"]
        for f in self.py_files:
            self._parse_python(f)
        entrypoints = self._entrypoints()
        readme_cmds = self._readme_commands(files)
        self._requirements(files)
        self._python_version(files)
        for f in files:
            if f.suffix in CONFIG_EXT and f.name not in CONFIG_SKIP and ".github" not in f.parts:
                self._config(f)
        for f in self.py_files:
            self._python_facts(f)
        reachable = self._reachable_from(entrypoints)
        frameworks = sorted({fw for f in self.trees.values() for fw in _imports_frameworks(f)})
        cli_frameworks = sorted({fw for f in self.trees.values() for fw in _imports_cli(f)})
        for fw in frameworks:
            self.add("framework", fw)

        # Seeding is judged per entry point: a seeded baseline must not hide an unseeded main script.
        seed_files = {x.file for x in self.facts if x.kind == "seed_call"}
        for e in entrypoints:
            reach = self._reachable_from([e])
            e["reach"] = sorted(reach)
            e["seed_calls"] = [{"file": x.file, "line": x.line, "call": x.name}
                               for x in self.facts if x.kind == "seed_call" and x.file in reach]
            e["seeded"] = bool(e["seed_calls"])
        unseeded = [e["file"] for e in entrypoints if not e["seeded"]]
        gpu = [x for x in self.facts if x.kind == "gpu_only"]
        pins = [x for x in self.facts if x.kind == "requirement"]
        return {
            "summary": {
                "python_files": len(self.py_files), "files": len(files), "notebooks": notebooks,
                "frameworks": frameworks, "cli_frameworks": cli_frameworks,
                "entrypoints": [e["file"] for e in entrypoints],
                "readme_commands": len(readme_cmds),
                "seed_set_in_entrypoints": (not unseeded) if entrypoints else None,
                "unseeded_entrypoints": unseeded,
                "any_seed_call": bool(seed_files),
                "deps_total": len(pins),
                "deps_pinned": sum(1 for p in pins if p.extra.get("pin") == "exact"),
                "deps_all_pinned": (all(p.extra.get("pin") == "exact" for p in pins) if pins else None),
                "gpu_only_files": sorted({g.file for g in gpu if not g.extra.get("has_fallback")}),
                "python_version": next((x.value for x in self.facts if x.kind == "python_version"), None),
            },
            "entrypoints": entrypoints,
            "readme_commands": readme_cmds,
            "reachable_from_entrypoints": sorted(reachable),
            "warnings": self.warnings,
        }

    # ------------------------------------------------------------ python
    def _parse_python(self, path: Path) -> None:
        rel = self.rel(path)
        try:
            if path.stat().st_size > MAX_PY_BYTES:
                self.warnings.append(f"Skipped large Python file {rel}")
                return
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self.warnings.append(f"Could not read {rel}: {exc}")
            return
        self.sources[rel] = src
        try:
            self.trees[rel] = ast.parse(src, filename=rel)
        except SyntaxError as exc:
            self.warnings.append(f"{rel} is not valid Python 3 (line {exc.lineno}); only text patterns were scanned")

    def _entrypoints(self) -> list[dict[str, Any]]:
        out = []
        for rel, tree in self.trees.items():
            has_main = any(isinstance(n, ast.If) and _is_main_guard(n.test) for n in tree.body)
            args = _argparse_args(tree)
            click_opts = _click_options(tree)
            name_hint = bool(ENTRY_NAME_RE.match(Path(rel).name))
            if not (has_main or (name_hint and (args or click_opts))):
                continue
            if "/tests/" in f"/{rel}" or Path(rel).name.startswith("test_"):
                continue
            score = (2 if has_main else 0) + (2 if name_hint else 0) + (1 if args or click_opts else 0)
            out.append({"file": rel, "has_main_guard": has_main, "name_hint": name_hint, "score": score,
                        "args": args + click_opts})
            self.add("entrypoint", rel, None, rel, 1, has_main_guard=has_main, score=score)
            for a in args + click_opts:
                self.add("arg", a["name"], a["default"], rel, a["line"], flags=a["flags"], type=a["type"],
                         action=a["action"], required=a["required"], choices=a["choices"],
                         help=a["help"], script=rel)
        out.sort(key=lambda e: (-e["score"], e["file"]))
        return out

    def _python_facts(self, path: Path) -> None:
        rel = self.rel(path)
        src = self.sources.get(rel)
        if src is None:
            return
        lines = src.splitlines()
        has_gpu_fallback = bool(GPU_FALLBACK_RE.search(src))
        for no, line in enumerate(lines, 1):
            code = line.split("#", 1)[0]
            if not code.strip():
                continue
            if m := SEED_RE.search(code):
                self.add("seed_call", m.group(1).rstrip("("), code.strip()[:160], rel, no)
            if m := DATA_RE.search(code):
                self.add("data_loader", m.group(1)[:120], code.strip()[:160], rel, no)
            if m := CHECKPOINT_RE.search(code):
                target = m.group(2)
                exists = (self.root / target).exists()
                self.add("checkpoint", target, "present" if exists else "missing", rel, no,
                         call=m.group(1), exists=exists)
            if OUTPUT_RE.search(code) or METRIC_PRINT_RE.search(code):
                self.add("output_write", code.strip()[:160], None, rel, no)
            if m := GPU_RE.search(code):
                self.add("gpu_only", m.group(1), code.strip()[:160], rel, no, has_fallback=has_gpu_fallback)
        tree = self.trees.get(rel)
        if tree is not None:
            seen_imports: set[str] = set()
            for node in ast.walk(tree):
                mods: list[str] = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mods = [node.module]
                for mod in mods:
                    top = mod.split(".")[0]
                    if top in seen_imports or top in sys.stdlib_module_names or top in self._local_modules():
                        continue
                    seen_imports.add(top)
                    self.add("import", top, mod, rel, node.lineno)
            for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
                fname = _call_name(call)
                if fname.endswith(("f1_score", "precision_score", "recall_score", "roc_auc_score",
                                   "accuracy_score", "mean_squared_error", "mean_absolute_error",
                                   "balanced_accuracy_score", "r2_score")):
                    self.add("metric_call", fname.split(".")[-1], _kw(call, "average"), rel, call.lineno,
                             kwargs=_literal_kwargs(call))
                elif fname.endswith("train_test_split"):
                    kw = _literal_kwargs(call)
                    self.add("split_call", "train_test_split", kw.get("test_size", kw.get("train_size")),
                             rel, call.lineno, kwargs=kw, seeded="random_state" in kw and kw["random_state"] is not None)

    def _local_modules(self) -> set[str]:
        """Top-level names importable from the repo itself (scripts, packages, src/ layout)."""
        if not hasattr(self, "_local_cache"):
            names: set[str] = set()
            for rel in self.trees:
                parts = Path(rel).with_suffix("").parts
                names.add(parts[0])
                if parts[0] in ("src", "lib") and len(parts) > 1:
                    names.add(parts[1])
                names.add(parts[-1] if parts[-1] != "__init__" else (parts[-2] if len(parts) > 1 else parts[0]))
            self._local_cache = names
        return self._local_cache

    def _reachable_from(self, entrypoints: list[dict[str, Any]]) -> set[str]:
        """Local import graph: which repo files can an entry point reach?"""
        modules: dict[str, str] = {}
        for rel in self.trees:
            p = Path(rel)
            dotted = ".".join(p.with_suffix("").parts)
            modules[dotted] = rel
            if p.name == "__init__.py":
                modules[".".join(p.parent.parts)] = rel
            modules.setdefault(p.stem, rel)        # scripts often add their folder to sys.path
        edges: dict[str, set[str]] = {}
        for rel, tree in self.trees.items():
            here = Path(rel).parent.parts
            deps = set()
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        parent = here[: len(here) - (node.level - 1)] if node.level > 1 else here
                        base = ".".join([*parent, base] if base else parent)
                    names = [base] + [f"{base}.{a.name}" if base else a.name for a in node.names]
                for name in names:
                    parts = name.split(".")
                    for k in range(len(parts), 0, -1):
                        target = modules.get(".".join(parts[:k]))
                        if target:
                            deps.add(target)
                            break
            edges[rel] = deps
        seen: set[str] = set()
        stack = [e["file"] for e in entrypoints]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(edges.get(cur, ()))
        return seen

    # ------------------------------------------------------------ README
    def _readme_commands(self, files: list[Path]) -> list[dict[str, Any]]:
        cmds: list[dict[str, Any]] = []
        readmes = [f for f in files if f.name.lower().startswith("readme") and f.suffix.lower() in (".md", ".rst", ".txt", "")]
        for f in readmes:
            rel = self.rel(f)
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            in_block = False
            buffer, start = "", 0
            for no, raw in enumerate(lines, 1):
                stripped = raw.strip()
                if stripped.startswith("```"):
                    in_block = not in_block
                    continue
                candidate = stripped
                if not in_block:
                    # also catch indented code and inline `python foo.py`
                    m = re.search(r"`((?:python3?|bash|sh)\s+[^`]+)`", raw)
                    if not (raw.startswith(("    ", "\t")) or m):
                        continue
                    candidate = m.group(1) if m else stripped
                candidate = re.sub(r"^\$\s*", "", candidate)
                if buffer:
                    buffer += " " + candidate.rstrip("\\").strip()
                elif re.match(r"^(python3?|bash|sh|CUDA_VISIBLE_DEVICES=\S+\s+python3?)\s", candidate):
                    buffer, start = candidate.rstrip("\\").strip(), no
                else:
                    continue
                if not candidate.endswith("\\"):
                    cmd = re.sub(r"\s+", " ", buffer).strip()
                    if cmd and not any(c["command"] == cmd for c in cmds):
                        script = _script_of(cmd)
                        cmds.append({"command": cmd, "file": rel, "line": start, "script": script,
                                     "script_exists": bool(script and (self.root / script).is_file())})
                        self.add("readme_cmd", cmd, script, rel, start, script_exists=cmds[-1]["script_exists"])
                    buffer = ""
        return cmds

    # ------------------------------------------------------ dependencies
    def _requirements(self, files: list[Path]) -> None:
        for f in files:
            rel = self.rel(f)
            name = f.name.lower()
            try:
                if re.match(r"^requirements.*\.(txt|in)$", name):
                    for no, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        self._requirement(line, rel, no)
                elif name == "setup.py" and rel in self.trees:
                    for call in (n for n in ast.walk(self.trees[rel]) if isinstance(n, ast.Call)):
                        for kw in call.keywords:
                            if kw.arg == "install_requires" and isinstance(kw.value, (ast.List, ast.Tuple)):
                                for elt in kw.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        self._requirement(elt.value, rel, elt.lineno)
                            if kw.arg == "python_requires" and isinstance(kw.value, ast.Constant):
                                self.add("python_version", "python_requires", kw.value.value, rel, kw.value.lineno)
                elif name == "pyproject.toml":
                    data = tomllib.loads(f.read_text(encoding="utf-8", errors="replace"))
                    text = f.read_text(encoding="utf-8", errors="replace").splitlines()
                    project = data.get("project", {})
                    for dep in project.get("dependencies", []):
                        self._requirement(dep, rel, _find_line(text, dep))
                    if rp := project.get("requires-python"):
                        self.add("python_version", "requires-python", rp, rel, _find_line(text, "requires-python"))
                    poetry = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
                    for pkg, spec in poetry.items():
                        if pkg.lower() == "python":
                            self.add("python_version", "poetry python", spec, rel, _find_line(text, "python ="))
                        else:
                            ver = spec if isinstance(spec, str) else spec.get("version", "") if isinstance(spec, dict) else ""
                            self._requirement(f"{pkg}{'==' + ver if ver and ver[0].isdigit() else ver}", rel,
                                              _find_line(text, pkg))
                elif name in ("environment.yml", "environment.yaml"):
                    text = f.read_text(encoding="utf-8", errors="replace")
                    data = yaml.safe_load(text) or {}
                    lines = text.splitlines()
                    for dep in data.get("dependencies", []) or []:
                        if isinstance(dep, str):
                            conda = dep.replace("=", "==", 1) if re.match(r"^[\w.-]+=[\d]", dep) and "==" not in dep else dep
                            if conda.lower().startswith("python"):
                                self.add("python_version", "conda python", dep, rel, _find_line(lines, dep))
                            else:
                                self._requirement(conda, rel, _find_line(lines, dep), source="conda")
                        elif isinstance(dep, dict):
                            for pip_dep in dep.get("pip", []) or []:
                                self._requirement(pip_dep, rel, _find_line(lines, pip_dep))
                elif name == ".python-version":
                    self.add("python_version", ".python-version", f.read_text().strip(), rel, 1)
            except (OSError, ValueError, yaml.YAMLError, tomllib.TOMLDecodeError) as exc:
                self.warnings.append(f"Could not parse {rel}: {type(exc).__name__}")

    def _requirement(self, line: str, rel: str, no: Optional[int], source: str = "pip") -> None:
        spec = line.split("#", 1)[0].strip()
        if not spec or spec.startswith(("-", "--")):
            if spec.startswith(("-e", "git+", "--extra-index-url", "-f", "--find-links")):
                self.add("requirement", spec, "vcs-or-option", rel, no, pin="other", source=source)
            return
        spec = spec.split(";", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$", spec)
        if not m:
            return
        name, version = m.group(1), m.group(3).strip()
        if "git+" in spec or "@" in version:
            pin = "other"
        elif re.match(r"^===?\s*[\w.*+!-]+$", version) and "*" not in version:
            pin = "exact"
        elif version:
            pin = "range"
        else:
            pin = "none"
        self.add("requirement", name, version or None, rel, no, pin=pin, source=source)

    def _python_version(self, files: list[Path]) -> None:
        if any(x.kind == "python_version" for x in self.facts):
            return
        for f in files:
            if f.name.lower().startswith("readme"):
                try:
                    for no, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if m := re.search(r"[Pp]ython\s*(?:version\s*)?(>=|==|≥)?\s*(3\.\d{1,2})", line):
                            self.add("python_version", "README mention", (m.group(1) or "") + m.group(2),
                                     self.rel(f), no)
                            return
                except OSError:
                    continue

    # ------------------------------------------------------------ configs
    def _config(self, path: Path) -> None:
        rel = self.rel(path)
        try:
            if path.stat().st_size > MAX_CONFIG_BYTES:
                return
            text = path.read_text(encoding="utf-8", errors="replace")
            if path.suffix in (".yaml", ".yml"):
                for key, value, line in _yaml_leaves(text):
                    self.add("config_value", key, value, rel, line)
            elif path.suffix == ".json":
                data = json.loads(text)
                lines = text.splitlines()
                for key, value in _flatten(data):
                    self.add("config_value", key, value, rel, _find_line(lines, f'"{key.split(".")[-1]}"'))
            elif path.suffix == ".toml":
                data = tomllib.loads(text)
                lines = text.splitlines()
                for key, value in _flatten(data):
                    self.add("config_value", key, value, rel,
                             _find_line(lines, key.split(".")[-1], regex=rf"^\s*{re.escape(key.split('.')[-1])}\s*="))
        except (ValueError, yaml.YAMLError, tomllib.TOMLDecodeError, OSError) as exc:
            self.warnings.append(f"Could not parse config {rel}: {type(exc).__name__}")


# --------------------------------------------------------------- AST helpers

def _is_main_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators))


def _call_name(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func)
    except Exception:
        return ""


def _literal(node: Optional[ast.AST]) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        try:
            return ast.unparse(node)
        except Exception:
            return None


def _kw(call: ast.Call, name: str) -> Any:
    for kw in call.keywords:
        if kw.arg == name:
            return _literal(kw.value)
    return None


def _literal_kwargs(call: ast.Call) -> dict[str, Any]:
    return {kw.arg: _literal(kw.value) for kw in call.keywords if kw.arg}


def _argparse_args(tree: ast.Module) -> list[dict[str, Any]]:
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not flags:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        action = _literal(kw.get("action"))
        default = _literal(kw["default"]) if "default" in kw else (False if action == "store_true"
                                                                    else True if action == "store_false" else None)
        long_flags = [f for f in flags if f.startswith("--")]
        name = (long_flags or flags)[0]
        out.append({"name": name, "flags": flags, "default": default,
                    "type": ast.unparse(kw["type"]) if "type" in kw else None,
                    "action": action, "required": bool(_literal(kw.get("required"))),
                    "choices": _literal(kw.get("choices")), "help": _literal(kw.get("help")),
                    "positional": not flags[0].startswith("-"), "line": node.lineno})
    return out


def _click_options(tree: ast.Module) -> list[dict[str, Any]]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and _call_name(dec).endswith(("click.option", "typer.Option")):
                flags = [a.value for a in dec.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                         and a.value.startswith("-")]
                if flags:
                    out.append({"name": next((f for f in flags if f.startswith("--")), flags[0]), "flags": flags,
                                "default": _kw(dec, "default"), "type": None, "action": None,
                                "required": bool(_kw(dec, "required")), "choices": None,
                                "help": _kw(dec, "help"), "positional": False, "line": dec.lineno})
    return out


def _imports(tree: ast.Module) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module.split(".")[0]


def _imports_frameworks(tree: ast.Module) -> set[str]:
    return {FRAMEWORKS[m] for m in _imports(tree) if m in FRAMEWORKS}


def _imports_cli(tree: ast.Module) -> set[str]:
    return {CONFIG_FRAMEWORKS[m] for m in _imports(tree) if m in CONFIG_FRAMEWORKS}


def _script_of(cmd: str) -> Optional[str]:
    tokens = cmd.split()
    for i, tok in enumerate(tokens):
        if re.match(r"^python3?$", tok) or tok.endswith("/python"):
            rest = tokens[i + 1:]
            if rest[:1] == ["-m"] and len(rest) > 1:
                return rest[1].replace(".", "/") + ".py"
            return next((t for t in rest if t.endswith(".py")), None)
        if tok in ("bash", "sh") and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def _flatten(data: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(data, dict):
        for k, v in data.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(data, list) and data and all(isinstance(x, (dict, list)) for x in data):
        for i, v in enumerate(data[:20]):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, data if not isinstance(data, list) else json.dumps(data)[:200]


def _yaml_leaves(text: str) -> Iterable[tuple[str, Any, int]]:
    """Leaf key -> scalar value with 1-based line numbers, via the YAML node graph."""
    root = yaml.compose(text)
    if root is None:
        return

    def visit(node: yaml.Node, prefix: str) -> Iterable[tuple[str, Any, int]]:
        if isinstance(node, yaml.MappingNode):
            for k, v in node.value:
                key = f"{prefix}.{k.value}" if prefix else str(k.value)
                if isinstance(v, yaml.ScalarNode):
                    yield key, yaml.safe_load(yaml.serialize(v)), v.start_mark.line + 1
                else:
                    yield from visit(v, key)
        elif isinstance(node, yaml.SequenceNode) and prefix:
            if all(isinstance(x, yaml.ScalarNode) for x in node.value):
                yield prefix, [x.value for x in node.value], node.start_mark.line + 1
            else:
                for i, x in enumerate(node.value[:20]):
                    yield from visit(x, f"{prefix}[{i}]")
    yield from visit(root, "")


def _find_line(lines: list[str], needle: str, regex: Optional[str] = None) -> Optional[int]:
    for no, line in enumerate(lines, 1):
        if (regex and re.search(regex, line)) or (not regex and needle in line):
            return no
    return None


# --------------------------------------------------------------------- stage

def scan_repo(root: Path) -> tuple[dict[str, Any], list[Fact]]:
    scanner = Scanner(root)
    card = scanner.scan()
    return card, scanner.facts


def run(job_id: str, job: dict[str, Any], log: Callable[..., None]) -> dict[str, Any]:
    root = storage.repo_dir(job_id)
    card, facts = scan_repo(root)
    s = card["summary"]
    log(f"Scanned {s['python_files']} Python file(s); frameworks: {', '.join(s['frameworks']) or 'none detected'}")
    if card["entrypoints"]:
        log("Entry points: " + ", ".join(
            f"{e['file']} ({len(e['args'])} CLI args)" for e in card["entrypoints"][:6]),
            data={"entrypoints": [e["file"] for e in card["entrypoints"]]})
    else:
        log("No runnable entry point found (no __main__ guard or train/main/eval script)", level="warning")
    if card["readme_commands"]:
        log(f"{len(card['readme_commands'])} command(s) found in the README")
    if s["deps_total"]:
        log(f"Dependencies: {s['deps_total']} listed, {s['deps_pinned']} pinned exactly")
    else:
        log("No dependency file found (requirements.txt / setup.py / pyproject.toml / environment.yml)", level="warning")
    if s["unseeded_entrypoints"]:
        log(f"No random-seed call reachable from: {', '.join(s['unseeded_entrypoints'])}", level="warning")
    if s["gpu_only_files"]:
        log(f"GPU-only code without a CPU fallback in: {', '.join(s['gpu_only_files'])}", level="warning")
    if s["notebooks"]:
        log(f"{len(s['notebooks'])} notebook(s) found; notebooks are not executed", level="warning")
    for w in card["warnings"]:
        log(w, level="warning")

    with session() as sess:
        sess.exec(delete(RepoFact).where(RepoFact.job_id == job_id))
        for i, f in enumerate(facts, 1):
            sess.add(RepoFact(job_id=job_id, id=f"F{i}", kind=f.kind, name=f.name[:500], value=f.value,
                              file=f.file, line=f.line, extra=json.dumps(f.extra, default=str)))
        sess.commit()
    card["facts"] = [{"id": f"F{i}", **asdict(f)} for i, f in enumerate(facts, 1)]
    counts: dict[str, int] = {}
    for f in facts:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    card["fact_counts"] = counts
    log(f"Recorded {len(facts)} repository facts with file and line", data={"counts": counts})
    return card
