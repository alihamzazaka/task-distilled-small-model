"""Config loading + project-root resolution.

Everything path-like in configs/default.yaml is relative to the project
root (the directory containing pyproject.toml). Scripts call
``load_config()`` and ``resolve_path()`` so they work regardless of cwd.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_CONFIG_RELPATH = Path("configs") / "default.yaml"


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (default: this file, then cwd) looking for
    pyproject.toml. Falls back to the package's grandparent directory."""
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd().resolve())
    for base in candidates:
        for p in [base, *base.parents]:
            if (p / "pyproject.toml").exists():
                return p
    # src/distil_task/config.py -> parents[2] == project root in a checkout
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. `path` overrides the default configs/default.yaml."""
    root = find_project_root()
    cfg_path = Path(path) if path else root / _CONFIG_RELPATH
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_root"] = str(root)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve cfg['paths'][key] against the project root and return a Path."""
    root = Path(cfg.get("_root") or find_project_root())
    rel = cfg["paths"][key]
    p = Path(rel)
    return p if p.is_absolute() else root / p


def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def merged_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge a small override dict (e.g. from CLI flags) into cfg."""
    out = copy.deepcopy(cfg)

    def _merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v

    _merge(out, overrides)
    return out