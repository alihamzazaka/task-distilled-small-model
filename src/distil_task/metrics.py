"""Task-native metrics for invoice extraction. Pure Python — runs anywhere.

Definitions (committed in Phase 0):
- **field_f1** (metric M): flatten prediction and gold into (path, value)
  fields; a predicted field is correct if the same path exists in gold and
  the values match under normalization (money fields: numeric tolerance).
  F1 over all fields, averaged per document then across documents.
- **exact_match**: 1.0 iff every gold field is matched and no extra fields.
- **schema_valid_rate**: fraction of raw outputs that pass the strict
  pydantic gate in `schema.validate_invoice`.
- **teacher_agreement**: mean field-level agreement between student and
  teacher outputs on a fresh unseen pool.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from .schema import MONEY_FIELDS, try_validate_invoice

DEFAULT_MONEY_ABS_TOL = 0.01
QTY_ABS_TOL = 1e-4


# ---------------------------------------------------------------------------
# Field flattening
# ---------------------------------------------------------------------------

def flatten_invoice(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten an invoice dict to {field_path: value}.

    line_items are indexed by position: line_items[0].description, ...
    (Order matters on an invoice; the teacher and student both see the
    same document order, so positional comparison is the honest one.)
    """
    flat: dict[str, Any] = {}
    for key in ("vendor", "date", "currency", "subtotal", "tax", "grand_total", "payment_terms"):
        if key in doc:
            flat[key] = doc.get(key)
    for i, item in enumerate(doc.get("line_items") or []):
        if isinstance(item, Mapping):
            for sub in ("description", "qty", "unit_price", "total"):
                if sub in item:
                    flat[f"line_items[{i}].{sub}"] = item.get(sub)
    return flat


def _leaf_name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def values_match(
    path: str,
    a: Any,
    b: Any,
    money_abs_tol: float = DEFAULT_MONEY_ABS_TOL,
) -> bool:
    """Field-aware value comparison."""
    if a is None or b is None:
        return a is None and b is None
    leaf = _leaf_name(path)
    if leaf in MONEY_FIELDS:
        try:
            return abs(float(a) - float(b)) <= money_abs_tol
        except (TypeError, ValueError):
            return False
    if leaf == "qty":
        try:
            return abs(float(a) - float(b)) <= QTY_ABS_TOL
        except (TypeError, ValueError):
            return False
    if isinstance(a, str) and isinstance(b, str):
        return _norm_text(a) == _norm_text(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= money_abs_tol
    return a == b


# ---------------------------------------------------------------------------
# Per-document scores
# ---------------------------------------------------------------------------

@dataclass
class FieldScore:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def field_score(
    pred: Optional[Mapping[str, Any]],
    gold: Mapping[str, Any],
    money_abs_tol: float = DEFAULT_MONEY_ABS_TOL,
) -> FieldScore:
    """TP/FP/FN over flattened fields for one document.
    A `pred` of None (unparseable/invalid output) scores 0 recall."""
    gold_flat = flatten_invoice(gold)
    if pred is None:
        return FieldScore(tp=0, fp=0, fn=len(gold_flat))
    pred_flat = flatten_invoice(pred)
    tp = fp = 0
    for path, pv in pred_flat.items():
        if path in gold_flat and values_match(path, pv, gold_flat[path], money_abs_tol):
            tp += 1
        else:
            fp += 1
    fn = sum(
        1
        for path, gv in gold_flat.items()
        if path not in pred_flat or not values_match(path, pred_flat[path], gv, money_abs_tol)
    )
    return FieldScore(tp=tp, fp=fp, fn=fn)


def document_exact_match(
    pred: Optional[Mapping[str, Any]],
    gold: Mapping[str, Any],
    money_abs_tol: float = DEFAULT_MONEY_ABS_TOL,
) -> bool:
    """True iff pred and gold have identical field paths and every value matches."""
    if pred is None:
        return False
    pred_flat, gold_flat = flatten_invoice(pred), flatten_invoice(gold)
    if set(pred_flat) != set(gold_flat):
        return False
    return all(values_match(p, pred_flat[p], gold_flat[p], money_abs_tol) for p in gold_flat)


def field_agreement(
    a: Optional[Mapping[str, Any]],
    b: Optional[Mapping[str, Any]],
    money_abs_tol: float = DEFAULT_MONEY_ABS_TOL,
) -> float:
    """Symmetric field-level agreement in [0, 1] between two outputs
    (used both for teacher self-consistency and student-vs-teacher)."""
    if a is None or b is None:
        return 0.0
    fa, fb = flatten_invoice(a), flatten_invoice(b)
    paths = set(fa) | set(fb)
    if not paths:
        return 0.0
    agree = sum(
        1
        for p in paths
        if p in fa and p in fb and values_match(p, fa[p], fb[p], money_abs_tol)
    )
    return agree / len(paths)


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------

def schema_valid_rate(raw_outputs: Sequence[Any]) -> float:
    """Fraction of raw model outputs (str or dict) that pass the strict gate."""
    if not raw_outputs:
        return 0.0
    ok = sum(1 for o in raw_outputs if try_validate_invoice(o)[0] is not None)
    return ok / len(raw_outputs)


def teacher_agreement(
    student_preds: Sequence[Optional[Mapping[str, Any]]],
    teacher_preds: Sequence[Optional[Mapping[str, Any]]],
    money_abs_tol: float = DEFAULT_MONEY_ABS_TOL,
) -> float:
    """Mean field agreement between paired student/teacher outputs."""
    if not student_preds or len(student_preds) != len(teacher_preds):
        raise ValueError("student and teacher prediction lists must be equal-length and non-empty")
    return sum(
        field_agreement(s, t, money_abs_tol) for s, t in zip(student_preds, teacher_preds)
    ) / len(student_preds)


def evaluate_batch(
    preds: Sequence[Optional[Mapping[str, Any]]],
    golds: Sequence[Mapping[str, Any]],
    raw_outputs: Optional[Sequence[Any]] = None,
    money_abs_tol: float = DEFAULT_MONEY_ABS_TOL,
) -> dict[str, float]:
    """Aggregate metrics for one eval set. `preds[i]` may be None (invalid)."""
    if len(preds) != len(golds) or not golds:
        raise ValueError("preds and golds must be equal-length and non-empty")
    scores = [field_score(p, g, money_abs_tol) for p, g in zip(preds, golds)]
    n = len(scores)
    micro = FieldScore(
        tp=sum(s.tp for s in scores),
        fp=sum(s.fp for s in scores),
        fn=sum(s.fn for s in scores),
    )
    result = {
        "n": float(n),
        "field_f1": sum(s.f1 for s in scores) / n,          # macro (per-doc avg)
        "field_f1_micro": micro.f1,
        "field_precision": sum(s.precision for s in scores) / n,
        "field_recall": sum(s.recall for s in scores) / n,
        "exact_match": sum(document_exact_match(p, g, money_abs_tol) for p, g in zip(preds, golds)) / n,
    }
    if raw_outputs is not None:
        result["schema_valid_rate"] = schema_valid_rate(list(raw_outputs))
    else:
        result["schema_valid_rate"] = sum(1 for p in preds if p is not None) / n
    return result


def percentile(values: Iterable[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0,100]); pure Python, no numpy."""
    vals = sorted(values)
    if not vals:
        raise ValueError("no values")
    if len(vals) == 1:
        return vals[0]
    k = (pct / 100.0) * (len(vals) - 1)
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    frac = k - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac
