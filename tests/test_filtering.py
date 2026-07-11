"""Filtering tests: char-ngram Jaccard dedup fallback, cross-split leak check,
schema gate, and teacher self-consistency."""
from __future__ import annotations

import pytest

from distil_task.filtering import (
    char_ngrams,
    cross_split_leak_check,
    dedup_jaccard,
    dedup_texts,
    jaccard_similarity,
    schema_gate,
    self_consistency_check,
)

VALID_OUTPUT = {
    "vendor": "Acme Corp",
    "date": "2024-03-14",
    "currency": "USD",
    "line_items": [{"description": "Widget A", "qty": 2, "unit_price": 6.49, "total": 12.98}],
    "subtotal": 12.98,
    "tax": 0.0,
    "grand_total": 12.98,
    "payment_terms": None,
}


# ---------------------------------------------------------------------------
# char n-grams + Jaccard
# ---------------------------------------------------------------------------

def test_char_ngrams_basic():
    grams = char_ngrams("abcabc", n=3)
    assert grams == {"abc", "bca", "cab"}


def test_char_ngrams_collapses_whitespace_and_case():
    assert char_ngrams("A  B", n=3) == char_ngrams("a b", n=3)


def test_jaccard_identical_is_one():
    assert jaccard_similarity("hello world", "hello world") == pytest.approx(1.0)


def test_jaccard_disjoint_is_zero():
    assert jaccard_similarity("aaaa", "zzzz", n=3) == pytest.approx(0.0)


def test_jaccard_partial_between_zero_and_one():
    sim = jaccard_similarity("the quick brown fox", "the quick brown cat", n=3)
    assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# dedup_jaccard fallback
# ---------------------------------------------------------------------------

def test_dedup_drops_exact_duplicate():
    texts = [
        "Invoice ALPHA — total 100.00 USD, vendor Acme",
        "Invoice ALPHA — total 100.00 USD, vendor Acme",   # identical dup
        "Completely different document about zebras 9999",
    ]
    kept, dropped = dedup_jaccard(texts, threshold=0.90)
    assert kept == [0, 2]
    assert len(dropped) == 1
    idx, dup_of, sim = dropped[0]
    assert idx == 1 and dup_of == 0
    assert sim == pytest.approx(1.0)


def test_dedup_drops_near_duplicate():
    texts = [
        "the quick brown fox jumps over the lazy dog every morning",
        "the quick brown fox jumps over the lazy dog every morning!",  # +1 char
        "unrelated content mentioning umbrellas and rainstorms daily",
    ]
    kept, dropped = dedup_jaccard(texts, threshold=0.90)
    assert 0 in kept and 2 in kept
    assert [d[0] for d in dropped] == [1]


def test_dedup_keeps_distinct_documents():
    texts = ["alpha one two three", "beta four five six", "gamma seven eight nine"]
    kept, dropped = dedup_jaccard(texts, threshold=0.90)
    assert kept == [0, 1, 2]
    assert dropped == []


def test_dedup_texts_jaccard_method_label():
    texts = ["a document here", "a document here", "another thing entirely"]
    kept, dropped, method = dedup_texts(texts, threshold=0.90, method="jaccard")
    assert method == "jaccard"
    assert kept == [0, 2]


def test_dedup_texts_auto_falls_back_to_jaccard_without_embeddings():
    # sentence-transformers is not installed in the laptop/test env, so auto
    # must degrade to the pure-Python Jaccard path.
    texts = ["x y z repeated", "x y z repeated", "distinct payload word"]
    _, _, method = dedup_texts(texts, threshold=0.90, method="auto")
    assert method == "jaccard"


def test_dedup_texts_unknown_method_raises():
    with pytest.raises(ValueError):
        dedup_texts(["a", "b"], method="bogus")


# ---------------------------------------------------------------------------
# cross-split leak check
# ---------------------------------------------------------------------------

def test_cross_split_leak_detected():
    train = ["shared invoice body text about widgets and totals here"]
    heldout = ["shared invoice body text about widgets and totals here"]
    leaks = cross_split_leak_check(train, heldout, threshold=0.90)
    assert len(leaks) == 1
    held_idx, train_idx, sim = leaks[0]
    assert held_idx == 0 and train_idx == 0 and sim == pytest.approx(1.0)


def test_cross_split_no_false_positive():
    train = ["alpha document one about cats"]
    heldout = ["totally separate beta text on turbines"]
    assert cross_split_leak_check(train, heldout, threshold=0.90) == []


# ---------------------------------------------------------------------------
# schema gate
# ---------------------------------------------------------------------------

def test_schema_gate_splits_valid_and_invalid():
    records = [
        {"id": "1", "output": VALID_OUTPUT},
        {"id": "2", "output": {"vendor": "missing the rest"}},
        {"id": "3", "output": "not even json"},
    ]
    res = schema_gate(records)
    assert [r["id"] for r in res.kept] == ["1"]
    assert {r["id"] for r in res.dropped} == {"2", "3"}
    for r in res.dropped:
        assert r["_drop_reason"].startswith("schema:")


def test_schema_gate_normalizes_kept_output():
    messy = dict(VALID_OUTPUT, currency="$", date="03/14/2024")
    res = schema_gate([{"id": "1", "output": messy}])
    assert len(res.kept) == 1
    kept = res.kept[0]["output"]
    assert kept["currency"] == "USD"
    assert kept["date"] == "2024-03-14"


# ---------------------------------------------------------------------------
# self-consistency
# ---------------------------------------------------------------------------

def test_self_consistency_pass_on_identical():
    ok, agreement = self_consistency_check(VALID_OUTPUT, dict(VALID_OUTPUT), threshold=0.85)
    assert ok is True
    assert agreement == pytest.approx(1.0)


def test_self_consistency_fail_on_divergent():
    second = dict(VALID_OUTPUT, vendor="Other", grand_total=999.0, subtotal=999.0, date="2020-01-01")
    ok, agreement = self_consistency_check(VALID_OUTPUT, second, threshold=0.85)
    assert ok is False
    assert agreement < 0.85


def test_self_consistency_fail_on_none_second_pass():
    ok, agreement = self_consistency_check(VALID_OUTPUT, None, threshold=0.85)
    assert ok is False
    assert agreement == 0.0
