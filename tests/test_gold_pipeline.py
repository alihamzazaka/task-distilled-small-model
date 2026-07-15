"""Unit tests for the human-gold pipeline: sampler + Cohen's κ scorer.

Pure-Python, no network, no model. Covers informativeness ordering, stratified
coverage, the rating-sheet schema, and known-κ fixtures (perfect / chance / a
hand-worked textbook example).
"""
from __future__ import annotations

import csv

import pytest

from distil_task.gold_pipeline import (
    LABELS,
    RATING_SHEET_FIELDS,
    SampleItem,
    align_raters,
    binary_report,
    cohens_kappa,
    human_verified_accuracy,
    informativeness,
    normalize_label,
    raw_agreement,
    read_rater_csv,
    score_raters,
    stratified_sample,
    write_rating_sheet,
)


# --------------------------------------------------------------------------- #
# Cohen's κ — known values
# --------------------------------------------------------------------------- #

def test_kappa_perfect_agreement():
    a = ["correct", "incorrect", "correct", "incorrect"]
    assert cohens_kappa(a, list(a)) == pytest.approx(1.0)


def test_kappa_chance_is_zero():
    # p_o == p_e exactly → κ == 0 (matched marginals, no better than chance).
    a = ["correct", "correct", "incorrect", "incorrect"]
    b = ["correct", "incorrect", "correct", "incorrect"]
    assert cohens_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_kappa_hand_worked_example():
    # Textbook 2x2: agree=35/50, p_e=0.50 → κ = (0.70-0.50)/(1-0.50) = 0.40.
    #        B=yes B=no
    # A=yes    20    5
    # A=no     10   15
    a = ["yes"] * 25 + ["no"] * 25
    b = (["yes"] * 20 + ["no"] * 5) + (["yes"] * 10 + ["no"] * 15)
    assert raw_agreement(a, b) == pytest.approx(0.70)
    assert cohens_kappa(a, b) == pytest.approx(0.40, abs=1e-9)


def test_kappa_both_constant_and_identical():
    assert cohens_kappa(["correct"] * 5, ["correct"] * 5) == pytest.approx(1.0)


def test_kappa_constant_but_opposite():
    assert cohens_kappa(["correct"] * 4, ["incorrect"] * 4) == pytest.approx(0.0)


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["correct"], ["correct", "incorrect"])


def test_kappa_symmetric():
    a = ["correct", "incorrect", "correct", "correct", "incorrect"]
    b = ["correct", "correct", "correct", "incorrect", "incorrect"]
    assert cohens_kappa(a, b) == pytest.approx(cohens_kappa(b, a))


# --------------------------------------------------------------------------- #
# Informativeness ordering
# --------------------------------------------------------------------------- #

def test_informativeness_low_f1_ranks_higher():
    high = informativeness(field_f1=0.4, schema_repaired=False, exact_match=False)
    low = informativeness(field_f1=1.0, schema_repaired=False, exact_match=True)
    assert high > low


def test_informativeness_schema_repair_adds_weight():
    base = informativeness(field_f1=0.9, schema_repaired=False, exact_match=True)
    repaired = informativeness(field_f1=0.9, schema_repaired=True, exact_match=True)
    assert repaired > base


def test_informativeness_perfect_item_is_minimal():
    perfect = informativeness(field_f1=1.0, schema_repaired=False, exact_match=True)
    assert perfect == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Stratified sampling
# --------------------------------------------------------------------------- #

def _mk(id_, stratum, info):
    return SampleItem(id=id_, stratum=stratum, informativeness=info)


def test_stratified_sample_orders_by_informativeness():
    items = [_mk(f"i{i}", "USD", info) for i, info in enumerate([0.1, 0.9, 0.5, 0.3])]
    picked = stratified_sample(items, n=4)
    infos = [p.informativeness for p in picked]
    assert infos == sorted(infos, reverse=True)
    assert picked[0].informativeness == pytest.approx(0.9)


def test_stratified_sample_covers_all_strata():
    items = (
        [_mk(f"u{i}", "USD", 0.9 - i * 0.01) for i in range(10)]
        + [_mk(f"e{i}", "EUR", 0.2) for i in range(4)]
        + [_mk(f"p{i}", "PKR", 0.1) for i in range(2)]
    )
    picked = stratified_sample(items, n=8)
    strata = {p.stratum for p in picked}
    # every populated stratum is represented despite EUR/PKR being low-info
    assert strata == {"USD", "EUR", "PKR"}
    assert len(picked) == 8


def test_stratified_sample_proportional_allocation():
    items = (
        [_mk(f"u{i}", "USD", 0.5) for i in range(80)]
        + [_mk(f"e{i}", "EUR", 0.5) for i in range(20)]
    )
    picked = stratified_sample(items, n=10)
    counts = {}
    for p in picked:
        counts[p.stratum] = counts.get(p.stratum, 0) + 1
    # ~80/20 split → 8 USD, 2 EUR
    assert counts == {"USD": 8, "EUR": 2}


def test_stratified_sample_within_stratum_takes_top_info():
    items = [_mk(f"u{i}", "USD", info) for i, info in enumerate([0.1, 0.2, 0.9, 0.8])]
    picked = stratified_sample(items, n=2)
    assert {p.id for p in picked} == {"u2", "u3"}


def test_stratified_sample_n_exceeds_pool():
    items = [_mk("a", "USD", 0.5), _mk("b", "EUR", 0.5)]
    picked = stratified_sample(items, n=99)
    assert len(picked) == 2


def test_stratified_sample_deterministic():
    items = [_mk(f"i{i}", "USD" if i % 2 else "EUR", (i * 7) % 5 / 5.0)
             for i in range(20)]
    assert [p.id for p in stratified_sample(items, 6)] == \
           [p.id for p in stratified_sample(items, 6)]


def test_stratified_sample_empty():
    assert stratified_sample([], n=5) == []


# --------------------------------------------------------------------------- #
# Rating sheet schema + rater CSV round-trip
# --------------------------------------------------------------------------- #

def test_rating_sheet_schema(tmp_path):
    rows = [{"id": "x1", "source_document": "Store: A\nTotal: 5",
             "model_extraction": '{"vendor": "A"}', "label": "", "notes": ""}]
    path = write_rating_sheet(tmp_path / "sheet.csv", rows)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == RATING_SHEET_FIELDS
        got = list(reader)
    assert got[0]["id"] == "x1"
    # blank label + notes columns for humans to fill
    assert got[0]["label"] == ""
    assert got[0]["notes"] == ""


def test_read_rater_csv_normalizes_labels(tmp_path):
    path = tmp_path / "rater.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label", "notes"])
        w.writeheader()
        w.writerow({"id": "x1", "label": "Correct", "notes": ""})
        w.writerow({"id": "x2", "label": "WRONG", "notes": "date off"})
        w.writerow({"id": "x3", "label": "", "notes": "unsure"})
    rows = read_rater_csv(path)
    by_id = {r["id"]: r for r in rows}
    assert by_id["x1"]["label"] == "correct"
    assert by_id["x2"]["label"] == "incorrect"
    assert by_id["x3"]["label"] is None  # blank stays unrated


def test_normalize_label_aliases():
    assert normalize_label("ok") == "correct"
    assert normalize_label("bad") == "incorrect"
    assert normalize_label("  ") is None
    assert normalize_label("banana") is None


# --------------------------------------------------------------------------- #
# score_raters + model-vs-human
# --------------------------------------------------------------------------- #

def _rows(pairs):
    return [{"id": i, "label": lab} for i, lab in pairs]


def test_score_raters_reports_kappa_and_disagreements():
    a = _rows([("i1", "correct"), ("i2", "incorrect"), ("i3", "correct")])
    b = _rows([("i1", "correct"), ("i2", "correct"), ("i3", "correct")])
    rep = score_raters(a, b)
    assert rep["n"] == 3
    assert rep["n_disagreements"] == 1
    assert rep["disagreements"][0]["id"] == "i2"
    assert 0.0 <= rep["raw_agreement"] <= 1.0


def test_align_raters_inner_joins_on_rated_items():
    a = _rows([("i1", "correct"), ("i2", None), ("i3", "incorrect")])
    b = _rows([("i1", "correct"), ("i3", "correct"), ("i4", "correct")])
    ids, la, lb = align_raters(a, b)
    assert ids == ["i1", "i3"]  # i2 unrated by A, i4 absent from A
    assert la == ["correct", "incorrect"]
    assert lb == ["correct", "correct"]


def test_human_verified_accuracy():
    gold = ["correct", "correct", "incorrect", "correct"]
    mvh = human_verified_accuracy(gold)
    assert mvh["human_verified_accuracy"] == pytest.approx(0.75)
    assert mvh["recall_correct"] == pytest.approx(1.0)


def test_binary_report_f1():
    y_true = ["incorrect", "incorrect", "correct", "correct"]
    y_pred = ["incorrect", "correct", "correct", "correct"]
    rep = binary_report(y_true, y_pred, positive="incorrect")
    # tp=1, fp=0, fn=1 → precision 1.0, recall 0.5, f1 0.667
    assert rep["precision"] == pytest.approx(1.0)
    assert rep["recall"] == pytest.approx(0.5)
    assert rep["f1"] == pytest.approx(2 / 3, abs=1e-6)


def test_labels_are_binary():
    assert set(LABELS) == {"correct", "incorrect"}
