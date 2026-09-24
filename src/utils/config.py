"""Loads configs/config.yaml and resolves paths relative to the project root."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """`overrides` is a nested dict merged over the file values."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return cfg


def resolve(path: str | Path) -> Path:
    """Relative paths are taken from the project root, absolute paths are kept."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
