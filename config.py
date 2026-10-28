"""Configuration loading.

The configuration is kept as a plain nested ``dict`` (YAML) so that it can be
snapshotted verbatim into the provenance log. Paths in ``paths`` are resolved
relative to the directory of the YAML file.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_REQUIRED = ["project", "paths", "roi", "normalization", "thresholds", "postprocessing",
             "statistics", "lab", "spatial", "sensitivity", "perturbation", "synthetic", "figures"]


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise KeyError(f"config {path} is missing sections: {missing}")
    cfg = copy.deepcopy(cfg)
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(path.parent)
    cfg["_paths_raw"] = dict(cfg["paths"])          # as written in the YAML, for the provenance record
    cfg["paths"] = {k: str((path.parent / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """``get(cfg, "thresholds.sauvola.window")`` with a default."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
