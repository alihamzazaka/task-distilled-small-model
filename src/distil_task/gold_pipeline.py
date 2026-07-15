"""Human-gold rating pipeline — active-learning sampler + Cohen's κ scorer.

Pure Python (stdlib only): no network, no heavy deps, fully unit-testable
offline. This is the machinery behind the "how do I run a κ ≥ 0.70 gold slice"
answer for the invoice-extraction student.

Domain
------
A human rater judges **invoice-JSON field correctness vs the source document**:
given the raw receipt/invoice text and the student's extracted JSON, is every
required field correct? Binary label (chosen for a stable κ at this slice size):

    correct    — every required field matches the source document
    incorrect  — at least one field is wrong / missing / hallucinated

Active-learning sampler
-----------------------
Human time is the scarce resource, so the sampler picks the pairs most worth
rating, scored by an *informativeness* signal:

    * model uncertainty       — low field-F1 vs the silver reference
    * schema-repair triggered — the raw output needed constrained repair to parse
    * exact-match miss         — the parsed prediction != the silver reference

…subject to **stratified coverage** (by currency here) so the slice is not all
one kind of document. See :func:`stratified_sample`.

The three stages this module powers:

    sample   → data/gold/rating_sheet.csv  (+ rating_instructions.md)
    rate     → two humans fill label/notes → rater_a.csv, rater_b.csv
    score    → Cohen's κ, adjudicate disagreements, model-vs-human agreement
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Label vocabulary + rating-sheet schema
# --------------------------------------------------------------------------- #

LABELS: tuple[str, ...] = ("correct", "incorrect")
POSITIVE_LABEL = "incorrect"  # the class we most care about catching

# Exact column order of the hand-rating sheet. `label` and `notes` start blank.
RATING_SHEET_FIELDS: tuple[str, ...] = (
    "id",
    "source_document",
    "model_extraction",
    "label",
    "notes",
)

# Aliases a human might type; folded to the canonical label.
_LABEL_ALIASES = {
    "correct": "correct",
    "c": "correct",
    "ok": "correct",
    "good": "correct",
    "1": "correct",
    "yes": "correct",
    "incorrect": "incorrect",
    "i": "incorrect",
    "wrong": "incorrect",
    "bad": "incorrect",
    "0": "incorrect",
    "no": "incorrect",
}


def normalize_label(value: Any) -> str | None:
    """Fold a raw cell to a canonical label, or ``None`` if blank/unknown."""
    if value is None:
        return None
    s = str(value).strip().casefold()
    if not s:
        return None
    return _LABEL_ALIASES.get(s)


# --------------------------------------------------------------------------- #
# Agreement statistics (pure Python — no numpy/sklearn)
# --------------------------------------------------------------------------- #

def _check_pair(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("empty label sequences")


def raw_agreement(a: Sequence, b: Sequence) -> float:
    """Fraction of items where the two raters gave the same label."""
    _check_pair(a, b)
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's κ between two raters over a shared item set.

    κ = (p_o − p_e) / (1 − p_e), where p_e is the chance agreement implied by
    each rater's marginal label frequencies. Works for any hashable label set.
    Returns 1.0 when both raters are constant *and* identical; 0.0 when chance
    agreement is total but they differ, matching sklearn's convention.
    """
    _check_pair(a, b)
    n = len(a)
    labels = sorted(set(a) | set(b), key=repr)
    p_o = raw_agreement(a, b)
    ca = {lab: 0 for lab in labels}
    cb = {lab: 0 for lab in labels}
    for x in a:
        ca[x] += 1
    for y in b:
        cb[y] += 1
    p_e = sum(ca[lab] * cb[lab] for lab in labels) / (n * n)
    if math.isclose(p_e, 1.0):
        return 1.0 if math.isclose(p_o, 1.0) else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def confusion_counts(a: Sequence, b: Sequence,
                     labels: Sequence | None = None) -> dict[tuple, int]:
    """Confusion dict keyed by (rater_a_label, rater_b_label)."""
    _check_pair(a, b)
    labs = list(labels) if labels is not None else sorted(set(a) | set(b), key=repr)
    counts = {(x, y): 0 for x in labs for y in labs}
    for x, y in zip(a, b):
        counts[(x, y)] = counts.get((x, y), 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Model-vs-human agreement (the defensible headline number)
# --------------------------------------------------------------------------- #

def binary_report(y_true: Sequence[str], y_pred: Sequence[str],
                  positive: str = POSITIVE_LABEL) -> dict[str, float]:
    """Accuracy / precision / recall / F1 / κ of predictions vs human gold."""
    _check_pair(y_true, y_pred)
    n = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": float(n),
        "accuracy": raw_agreement(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohen_kappa": cohens_kappa(y_true, y_pred),
    }


def human_verified_accuracy(gold_labels: Sequence[str]) -> dict[str, float]:
    """Model-vs-human number when the model has no independent label axis.

    The student emitted an extraction it considers correct for *every* item, so
    its implicit prediction is the positive-quality class ("correct") on all of
    them. The defensible number is then the fraction of items humans adjudicated
    as ``correct`` — the human-verified extraction accuracy — with recall on the
    "correct" class trivially 1.0.
    """
    if not gold_labels:
        raise ValueError("no gold labels")
    n = len(gold_labels)
    n_correct = sum(1 for g in gold_labels if g == "correct")
    acc = n_correct / n
    return {
        "n": float(n),
        "human_verified_accuracy": acc,
        "n_correct": float(n_correct),
        "n_incorrect": float(n - n_correct),
        # model asserts "correct" for all → precision == accuracy, recall == 1.0
        "precision_correct": acc,
        "recall_correct": 1.0,
    }


# --------------------------------------------------------------------------- #
# Active-learning informativeness + stratified sampling
# --------------------------------------------------------------------------- #

@dataclass
class SampleItem:
    """One candidate pair for the rating sheet, with its informativeness inputs."""

    id: str
    stratum: str
    informativeness: float
    payload: dict[str, Any] = field(default_factory=dict)


def informativeness(*, field_f1: float, schema_repaired: bool,
                    exact_match: bool,
                    w_uncertainty: float = 1.0,
                    w_repair: float = 0.5,
                    w_exact_miss: float = 0.5) -> float:
    """Higher = more worth a human's time.

    Combines model uncertainty (``1 - field_f1``), a schema-repair flag, and an
    exact-match miss flag. All inputs are in [0, 1]; the result is a weighted
    sum (not normalised — only the *ordering* matters to the sampler).
    """
    uncertainty = 1.0 - max(0.0, min(1.0, field_f1))
    return (w_uncertainty * uncertainty
            + w_repair * (1.0 if schema_repaired else 0.0)
            + w_exact_miss * (0.0 if exact_match else 1.0))


def _largest_remainder_alloc(sizes: Mapping[Any, int], n: int) -> dict[Any, int]:
    """Allocate ``n`` slots across strata ∝ size, capped by each stratum's size.

    Uses the largest-remainder method, then greedily fills any shortfall (caused
    by caps) into strata that still have room. Deterministic given ``sizes``.
    """
    total = sum(sizes.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in sizes}
    n = min(n, total)
    quotas = {k: n * sz / total for k, sz in sizes.items()}
    alloc = {k: min(sizes[k], int(math.floor(q))) for k, q in quotas.items()}
    assigned = sum(alloc.values())
    # Distribute the remaining slots by largest fractional remainder.
    remainders = sorted(
        sizes.keys(),
        key=lambda k: (quotas[k] - math.floor(quotas[k]), sizes[k], repr(k)),
        reverse=True,
    )
    i = 0
    while assigned < n and remainders:
        progressed = False
        for k in remainders:
            if assigned >= n:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
                assigned += 1
                progressed = True
        if not progressed:
            break
        i += 1
    return alloc


def stratified_sample(items: Sequence[SampleItem], n: int,
                      seed: int = 0) -> list[SampleItem]:
    """Pick ``n`` items: stratified coverage first, informativeness within.

    1. Group by ``item.stratum``.
    2. Allocate ``n`` across strata proportional to stratum size (largest
       remainder), so every populated stratum with room gets representation.
    3. Within each stratum, take the highest-informativeness items (ties broken
       by ``id`` for determinism).
    4. Return the union sorted by informativeness desc (id tie-break) so the
       sheet leads with the most-informative rows.
    """
    if n <= 0 or not items:
        return []
    strata: dict[str, list[SampleItem]] = {}
    for it in items:
        strata.setdefault(it.stratum, []).append(it)
    sizes = {k: len(v) for k, v in strata.items()}
    alloc = _largest_remainder_alloc(sizes, n)
    picked: list[SampleItem] = []
    for k, group in strata.items():
        take = alloc.get(k, 0)
        if take <= 0:
            continue
        ranked = sorted(group, key=lambda it: (-it.informativeness, it.id))
        picked.extend(ranked[:take])
    picked.sort(key=lambda it: (-it.informativeness, it.id))
    return picked


# --------------------------------------------------------------------------- #
# CSV I/O
# --------------------------------------------------------------------------- #

def write_rating_sheet(path: str | Path, rows: Sequence[Mapping[str, Any]],
                       fieldnames: Sequence[str] = RATING_SHEET_FIELDS) -> Path:
    """Write the blank-label rating sheet (UTF-8, standard CSV quoting)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def read_rater_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a filled rater CSV → list of {id, label, notes} (label normalised).

    Rows with an unrecognised/blank label keep ``label=None`` so the caller can
    decide whether to treat them as unrated.
    """
    path = Path(path)
    out: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"{path}: CSV must have an 'id' column")
        for row in reader:
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            out.append({
                "id": rid,
                "label": normalize_label(row.get("label")),
                "notes": (row.get("notes") or "").strip(),
                "raw_label": (row.get("label") or "").strip(),
            })
    return out


def align_raters(rows_a: Sequence[Mapping[str, Any]],
                 rows_b: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    """Inner-join two rater tables on ``id`` where both gave a valid label.

    Returns ``(ids, labels_a, labels_b)`` over the shared, fully-rated items.
    """
    by_a = {r["id"]: r["label"] for r in rows_a if r.get("label")}
    by_b = {r["id"]: r["label"] for r in rows_b if r.get("label")}
    ids = sorted(set(by_a) & set(by_b))
    return ids, [by_a[i] for i in ids], [by_b[i] for i in ids]


def score_raters(rows_a: Sequence[Mapping[str, Any]],
                 rows_b: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Inter-rater report: κ, raw agreement, confusion, and disagreement ids."""
    ids, la, lb = align_raters(rows_a, rows_b)
    if not ids:
        raise ValueError("no shared, fully-rated items between the two raters")
    disagreements = [
        {"id": i, "rater_a": a, "rater_b": b}
        for i, a, b in zip(ids, la, lb) if a != b
    ]
    return {
        "n": len(ids),
        "cohen_kappa": cohens_kappa(la, lb),
        "raw_agreement": raw_agreement(la, lb),
        "n_disagreements": len(disagreements),
        "confusion": {f"{x}|{y}": c
                      for (x, y), c in confusion_counts(la, lb, LABELS).items()},
        "disagreements": disagreements,
        "ids": ids,
    }
