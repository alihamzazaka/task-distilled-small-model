"""Put the repo's ``serve/`` directory on ``sys.path``.

``serve/infer.py`` is a deployment module, not part of the installed
``distil_task`` package, so scripts that need it (``scripts/evaluate.py``)
call :func:`ensure_serve_on_path` before ``from infer import ...``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def ensure_serve_on_path() -> Path:
    """Prepend ``<project_root>/serve`` to ``sys.path`` (idempotent).

    Returns the serve directory path.
    """
    serve_dir = _ROOT / "serve"
    if str(serve_dir) not in sys.path:
        sys.path.insert(0, str(serve_dir))
    # infer.py also needs distil_task; make src importable too.
    src_dir = _ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return serve_dir
