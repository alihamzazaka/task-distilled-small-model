"""Small shared helpers: JSONL I/O and robust JSON extraction from raw
model text (fences, prefixes, trailing chatter)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import ensure_parent


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSONL line: {e}") from e
    return out


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    p = ensure_parent(Path(path))
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Robust JSON extraction from model text
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_first_json(text: str) -> Any:
    """Extract the first JSON value (object or array) from raw model output.

    Handles: clean JSON, ```json fenced``` blocks, leading/trailing prose.
    Raises ValueError if nothing parseable is found.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model output")
    s = text.strip()

    # 1. whole string
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2. fenced block
    m = _FENCE_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            s = m.group(1)  # keep scanning inside the fence

    # 3. first balanced {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = s[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = s.find(opener, start + 1)

    raise ValueError(f"no parseable JSON found in model output ({len(text)} chars)")
