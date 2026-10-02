"""Matching paper hyperparameters to repository flags / config keys, and resolving the
value the repository will *actually* use for a given command.

Precedence for the effective repo value (highest first):
  1. a value passed on the command line (e.g. from the README command)
  2. a value from a config file the command selects (--config path)
  3. the argparse / click default in the script
"""
from __future__ import annotations

import re
from typing import Any, Optional

# canonical name -> aliases (normalised: lower-case, no '-', '_' or spaces)
SYNONYMS: dict[str, set[str]] = {
    "lr": {"lr", "learningrate", "learningrateinit", "eta", "initlr", "baselr"},
    "batch_size": {"batchsize", "bs", "batch", "trainbatchsize", "minibatchsize"},
    "epochs": {"epochs", "nepochs", "numepochs", "maxepochs", "epoch", "maxiter", "numiterations", "iterations"},
    "weight_decay": {"weightdecay", "wd", "l2", "l2reg"},
    "dropout": {"dropout", "pdrop", "droprate", "dropoutrate"},
    "hidden": {"hidden", "hiddendim", "hiddensize", "hiddenunits", "nhidden", "hiddenlayersizes", "units"},
    "layers": {"layers", "numlayers", "nlayers", "depth"},
    "seed": {"seed", "randomseed", "randomstate"},
    "optimizer": {"optimizer", "optim", "opt"},
    "momentum": {"momentum", "beta"},
    "activation": {"activation", "act", "activationfunction"},
    "n_estimators": {"nestimators", "ntrees", "numtrees"},
    "max_depth": {"maxdepth", "depth_max"},
    "C": {"c", "regularization", "invreg"},
    "solver": {"solver"},
    "heads": {"heads", "numheads", "nheads"},
    "test_size": {"testsize", "testsplit", "testfraction", "valsplit"},
}
# Training hyperparameters may never be changed in the primary run (paper values go to
# hypothesis reruns instead). Identity/architecture parameters may select the experiment.
TRAINING_PARAMS = {"lr", "batch_size", "epochs", "weight_decay", "dropout", "seed", "optimizer",
                   "momentum", "solver", "test_size"}
OUTPUT_FLAG_RE = re.compile(r"^--?(out|output|outdir|out_dir|output_dir|output-dir|save|save_dir|save-dir|"
                            r"log|logdir|log_dir|log-dir|results|result_file|results_file|metrics_file)$", re.I)
CONFIG_FLAG_RE = re.compile(r"^--?(config|cfg|config[_-]file|config[_-]path|conf|c)$", re.I)


def norm(name: str) -> str:
    return re.sub(r"[\s_\-.]+", "", name.strip().lower().lstrip("-"))


def canonical(name: str) -> Optional[str]:
    """Canonical parameter name for a hyperparameter name, CLI flag or config key."""
    n = norm(name.split(".")[-1])
    for canon, aliases in SYNONYMS.items():
        if n == norm(canon) or n in aliases:
            return canon
    return None


def parse_value(raw: Any) -> Any:
    """Turn '0.001', '1e-3', '200', 'True', 'adam' into comparable Python values."""
    if raw is None or isinstance(raw, (int, float, bool)):
        return raw
    s = str(raw).strip().strip("'\"")
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null"):
        return None
    s_num = s.replace(",", "")
    m = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*%?", s_num)
    if m:
        v = float(m.group(1))
        return int(v) if v.is_integer() and "." not in s_num and "e" not in s_num.lower() else v
    if re.fullmatch(r"\(\s*\d+\s*,?\s*\)", s):            # "(64,)" -> 64
        return int(re.sub(r"\D", "", s))
    return s.lower()


def values_equal(a: Any, b: Any) -> bool:
    pa, pb = parse_value(a), parse_value(b)
    if isinstance(pa, (int, float)) and isinstance(pb, (int, float)) and not isinstance(pa, bool):
        return abs(float(pa) - float(pb)) <= 1e-9 * max(1.0, abs(float(pa)), abs(float(pb)))
    return pa == pb


def effective_values(script: str, argv_args: list[str], facts: list[dict[str, Any]],
                     cmd_source: Optional[str]) -> dict[str, dict[str, Any]]:
    """canonical param -> {value, flag, source, file, line} for running `script` with `argv_args`."""
    out: dict[str, dict[str, Any]] = {}
    script_args = [f for f in facts if f["kind"] == "arg" and f["file"] == script]
    flag_to_fact: dict[str, dict[str, Any]] = {}
    for f in script_args:
        for flag in f.get("extra", {}).get("flags", [f["name"]]):
            flag_to_fact[flag] = f

    # 3. defaults
    for f in script_args:
        canon = canonical(f["name"])
        if canon:
            out[canon] = {"value": f["value"], "flag": f["name"], "source": "default",
                          "file": f["file"], "line": f["line"]}

    # 2. config file selected on the command line
    passed = _pairs(argv_args)
    for flag, value in passed:
        if CONFIG_FLAG_RE.match(flag) and value:
            for c in facts:
                if c["kind"] == "config_value" and c["file"] == value.lstrip("./"):
                    canon = canonical(c["name"])
                    if canon:
                        out[canon] = {"value": c["value"], "flag": out.get(canon, {}).get("flag"),
                                      "source": "config", "file": c["file"], "line": c["line"]}

    # 1. explicit command-line values
    for flag, value in passed:
        fact = flag_to_fact.get(flag)
        canon = canonical(fact["name"] if fact else flag)
        if canon and value is not None:
            out[canon] = {"value": value, "flag": flag, "source": "command",
                          "file": (cmd_source or "").split(":")[0] or None,
                          "line": _line_of(cmd_source)}
    return out


def _pairs(args: list[str]) -> list[tuple[str, Optional[str]]]:
    pairs, i = [], 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("-"):
            if "=" in tok:
                k, v = tok.split("=", 1)
                pairs.append((k, v))
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                pairs.append((tok, args[i + 1]))
                i += 1
            else:
                pairs.append((tok, None))
        i += 1
    return pairs


def _line_of(ref: Optional[str]) -> Optional[int]:
    if ref and ":" in ref:
        tail = ref.rsplit(":", 1)[1]
        return int(tail) if tail.isdigit() else None
    return None


def compare_hyperparameters(paper_hps: list[dict[str, Any]], effective: dict[str, dict[str, Any]]
                            ) -> list[dict[str, Any]]:
    """Paper-stated (verified) hyperparameters vs what the repo will use for this command."""
    out = []
    for hp in paper_hps:
        if not hp.get("verified"):
            continue
        canon = canonical(hp["name"])
        if canon is None:
            continue
        repo = effective.get(canon)
        entry = {"param": canon, "paper_name": hp["name"], "paper_value": hp["value"], "paper_quote": hp["quote"],
                 "flag": repo.get("flag") if repo else None,
                 "repo_value": repo.get("value") if repo else None,
                 "repo_source": repo.get("source") if repo else None,
                 "repo_file": repo.get("file") if repo else None, "repo_line": repo.get("line") if repo else None}
        if repo is None or repo.get("value") is None:
            entry["status"] = "not_in_repo"
        elif values_equal(hp["value"], repo["value"]):
            entry["status"] = "match"
        else:
            entry["status"] = "mismatch"
        entry["testable"] = entry["status"] == "mismatch" and bool(entry["flag"])
        out.append(entry)
    return out
