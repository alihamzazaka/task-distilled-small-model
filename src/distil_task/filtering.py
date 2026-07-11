"""Quality-filtering gates (SPEC §4.3): schema validation, near-duplicate
detection, and teacher self-consistency.

Dedup strategy is pluggable:
- **embedding** cosine similarity via sentence-transformers (GPU box), or
- **jaccard** over character n-grams — pure Python, dependency-free, used
  as the automatic fallback and in the local test suite.

All functions return *decisions with reasons* so scripts can log a
data-quality record of every drop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from .metrics import field_agreement
from .schema import try_validate_invoice

# ---------------------------------------------------------------------------
# 1. Schema gate
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    kept: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)  # each has "_drop_reason"


def schema_gate(records: Sequence[Mapping[str, Any]], output_key: str = "output") -> GateResult:
    """Keep records whose `output` validates; normalize kept outputs to the
    canonical form. Dropped records carry `_drop_reason`."""
    res = GateResult()
    for rec in records:
        inv, err = try_validate_invoice(rec.get(output_key))
        r = dict(rec)
        if inv is None:
            r["_drop_reason"] = f"schema: {err}"
            res.dropped.append(r)
        else:
            r[output_key] = inv.model_dump(mode="json")
            res.kept.append(r)
    return res


# ---------------------------------------------------------------------------
# 2. Near-duplicate detection
# ---------------------------------------------------------------------------

def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Lowercased, whitespace-collapsed character n-grams."""
    s = " ".join(text.lower().split())
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard_similarity(a: str, b: str, n: int = 3) -> float:
    ga, gb = char_ngrams(a, n), char_ngrams(b, n)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def dedup_jaccard(
    texts: Sequence[str],
    threshold: float = 0.90,
    n: int = 3,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Greedy near-dup removal, pure Python.

    Returns (kept_indices, dropped) where dropped = [(idx, dup_of_idx, sim)].
    First occurrence wins. Uses an inverted n-gram index so it is
    O(candidates) per item rather than all-pairs in practice.
    """
    kept: list[int] = []
    dropped: list[tuple[int, int, float]] = []
    grams_of: dict[int, set[str]] = {}
    index: dict[str, list[int]] = {}
    for i, text in enumerate(texts):
        g = char_ngrams(text, n)
        grams_of[i] = g
        # candidate kept items sharing at least one n-gram
        cand_counts: dict[int, int] = {}
        for gram in g:
            for j in index.get(gram, ()):
                cand_counts[j] = cand_counts.get(j, 0) + 1
        dup_of, best_sim = -1, 0.0
        # check candidates in order of shared grams (most similar first)
        for j, shared in sorted(cand_counts.items(), key=lambda kv: -kv[1]):
            gj = grams_of[j]
            union = len(g) + len(gj) - shared
            sim = shared / union if union else 1.0
            if sim >= threshold and sim > best_sim:
                dup_of, best_sim = j, sim
                break
        if dup_of >= 0:
            dropped.append((i, dup_of, round(best_sim, 4)))
        else:
            kept.append(i)
            for gram in g:
                index.setdefault(gram, []).append(i)
    return kept, dropped


def dedup_embeddings(
    texts: Sequence[str],
    threshold: float = 0.90,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Cosine-similarity dedup via sentence-transformers (GPU box only).
    Raises ImportError if the package is unavailable — callers should use
    `dedup_texts` which falls back to Jaccard."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(model_name)
    emb = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    kept: list[int] = []
    dropped: list[tuple[int, int, float]] = []
    for i in range(len(texts)):
        dup_of, best = -1, 0.0
        for j in kept:
            sim = float(emb[i] @ emb[j])  # normalized -> dot == cosine
            if sim >= threshold and sim > best:
                dup_of, best = j, sim
        if dup_of >= 0:
            dropped.append((i, dup_of, round(best, 4)))
        else:
            kept.append(i)
    return kept, dropped


def dedup_texts(
    texts: Sequence[str],
    threshold: float = 0.90,
    method: str = "auto",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[list[int], list[tuple[int, int, float]], str]:
    """Dispatch: 'embedding' | 'jaccard' | 'auto' (embedding if available).
    Returns (kept_indices, dropped, method_used)."""
    method = method.lower()
    if method not in {"auto", "embedding", "embeddings", "jaccard"}:
        raise ValueError(f"unknown dedup method {method!r}")
    if method in {"embedding", "embeddings", "auto"}:
        try:
            kept, dropped = dedup_embeddings(texts, threshold, embedding_model)
            return kept, dropped, "embedding"
        except ImportError:
            if method != "auto":
                raise
    kept, dropped = dedup_jaccard(texts, threshold)
    return kept, dropped, "jaccard"


def cross_split_leak_check(
    train_texts: Sequence[str],
    heldout_texts: Sequence[str],
    threshold: float = 0.90,
    n: int = 3,
) -> list[tuple[int, int, float]]:
    """Return [(heldout_idx, train_idx, sim)] pairs whose similarity exceeds
    the threshold — any hit is train/test leakage and must be removed."""
    leaks: list[tuple[int, int, float]] = []
    train_grams = [char_ngrams(t, n) for t in train_texts]
    for i, h in enumerate(heldout_texts):
        gh = char_ngrams(h, n)
        for j, gt in enumerate(train_grams):
            union = len(gh | gt)
            sim = len(gh & gt) / union if union else 1.0
            if sim >= threshold:
                leaks.append((i, j, round(sim, 4)))
    return leaks


# ---------------------------------------------------------------------------
# 3. Self-consistency (teacher twice, field-level agreement)
# ---------------------------------------------------------------------------

def self_consistency_check(
    first: Optional[Mapping[str, Any]],
    second: Optional[Mapping[str, Any]],
    threshold: float = 0.85,
    money_abs_tol: float = 0.01,
) -> tuple[bool, float]:
    """Compare two teacher outputs for the same input. Returns
    (passes, agreement). An unparseable second pass fails the check."""
    agreement = field_agreement(first, second, money_abs_tol)
    return agreement >= threshold, agreement


def run_consistency_filter(
    records: Sequence[Mapping[str, Any]],
    second_outputs: Mapping[str, Any],
    threshold: float = 0.85,
    money_abs_tol: float = 0.01,
    id_key: str = "id",
    output_key: str = "output",
) -> GateResult:
    """Filter `records` by agreement with a second teacher pass keyed by id.
    Records with no second output are kept (check is best-effort) but tagged."""
    res = GateResult()
    for rec in records:
        r = dict(rec)
        rid = str(rec.get(id_key))
        second = second_outputs.get(rid)
        if second is None:
            r["_consistency"] = None
            res.kept.append(r)
            continue
        ok, agreement = self_consistency_check(
            rec.get(output_key), second, threshold, money_abs_tol
        )
        r["_consistency"] = round(agreement, 4)
        if ok:
            res.kept.append(r)
        else:
            r["_drop_reason"] = f"consistency: agreement {agreement:.3f} < {threshold}"
            res.dropped.append(r)
    return res
