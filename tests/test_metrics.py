"""Metrics tests: field exact-match, field F1, money numeric tolerance,
schema-valid rate, teacher-agreement."""
from __future__ import annotations

import copy

import pytest

from distil_task.metrics import (
    document_exact_match,
    evaluate_batch,
    field_agreement,
    field_score,
    flatten_invoice,
    percentile,
    schema_valid_rate,
    teacher_agreement,
    values_match,
)

GOLD = {
    "vendor": "Acme Corp",
    "date": "2024-03-14",
    "currency": "USD",
    "line_items": [
        {"description": "Widget A", "qty": 2.0, "unit_price": 6.49, "total": 12.98},
        {"description": "Widget B", "qty": 1.0, "unit_price": 5.00, "total": 5.00},
    ],
    "subtotal": 17.98,
    "tax": 1.44,
    "grand_total": 19.42,
    "payment_terms": None,
}

# GOLD has 7 top-level fields + 2 line items x 4 subfields = 15 flat fields.
N_FLAT_FIELDS = 15


def test_flatten_counts_all_fields():
    assert len(flatten_invoice(GOLD)) == N_FLAT_FIELDS


# ---------------------------------------------------------------------------
# Money numeric tolerance
# ---------------------------------------------------------------------------

def test_money_match_within_tolerance():
    assert values_match("grand_total", 19.42, 19.43, money_abs_tol=0.01) is True
    assert values_match("grand_total", 19.42, 19.42, money_abs_tol=0.01) is True


def test_money_match_outside_tolerance():
    assert values_match("grand_total", 19.42, 19.44, money_abs_tol=0.01) is False


def test_text_match_is_case_and_whitespace_insensitive():
    assert values_match("vendor", "Acme  Corp", "acme corp") is True
    assert values_match("vendor", "Acme", "Beta") is False


def test_none_only_matches_none():
    assert values_match("payment_terms", None, None) is True
    assert values_match("payment_terms", None, "Net 30") is False


# ---------------------------------------------------------------------------
# Field score / exact match / F1
# ---------------------------------------------------------------------------

def test_perfect_prediction_scores_one():
    pred = copy.deepcopy(GOLD)
    fs = field_score(pred, GOLD)
    assert (fs.tp, fs.fp, fs.fn) == (N_FLAT_FIELDS, 0, 0)
    assert fs.f1 == pytest.approx(1.0)
    assert document_exact_match(pred, GOLD) is True


def test_one_wrong_field_lowers_f1_and_breaks_exact_match():
    pred = copy.deepcopy(GOLD)
    pred["vendor"] = "Wrong Vendor"
    fs = field_score(pred, GOLD)
    assert fs.tp == N_FLAT_FIELDS - 1
    assert fs.fp == 1 and fs.fn == 1
    expected_f1 = 2 * (14 / 15) * (14 / 15) / (2 * (14 / 15))
    assert fs.f1 == pytest.approx(expected_f1)
    assert document_exact_match(pred, GOLD) is False


def test_money_tolerance_flows_through_scoring():
    pred = copy.deepcopy(GOLD)
    pred["grand_total"] = 19.42 + 0.009   # within 0.01 tolerance
    assert document_exact_match(pred, GOLD) is True
    pred["grand_total"] = 19.42 + 0.5     # clearly wrong
    assert document_exact_match(pred, GOLD) is False


def test_none_prediction_scores_zero_recall():
    fs = field_score(None, GOLD)
    assert fs.tp == 0 and fs.fn == N_FLAT_FIELDS
    assert fs.recall == 0.0
    assert document_exact_match(None, GOLD) is False


def test_missing_line_item_penalized():
    pred = copy.deepcopy(GOLD)
    pred["line_items"] = pred["line_items"][:1]   # drop the 2nd item (4 fields)
    fs = field_score(pred, GOLD)
    assert fs.fn == 4          # 4 missing subfields
    assert document_exact_match(pred, GOLD) is False


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------

def test_evaluate_batch_all_correct():
    preds = [copy.deepcopy(GOLD), copy.deepcopy(GOLD)]
    golds = [GOLD, GOLD]
    res = evaluate_batch(preds, golds)
    assert res["field_f1"] == pytest.approx(1.0)
    assert res["exact_match"] == pytest.approx(1.0)
    assert res["schema_valid_rate"] == pytest.approx(1.0)
    assert res["n"] == 2.0


def test_evaluate_batch_mixed_with_none():
    preds = [copy.deepcopy(GOLD), None]
    golds = [GOLD, GOLD]
    res = evaluate_batch(preds, golds)
    assert res["exact_match"] == pytest.approx(0.5)
    assert 0.0 < res["field_f1"] < 1.0
    assert res["schema_valid_rate"] == pytest.approx(0.5)  # 1 of 2 preds non-None


def test_evaluate_batch_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_batch([GOLD], [GOLD, GOLD])


# ---------------------------------------------------------------------------
# schema_valid_rate on raw outputs
# ---------------------------------------------------------------------------

def test_schema_valid_rate_on_raw_outputs():
    valid_json = (
        '{"vendor":"X","date":"2024-01-01","currency":"USD",'
        '"line_items":[{"description":"a","qty":1,"unit_price":1,"total":1}],'
        '"subtotal":1,"tax":0,"grand_total":1,"payment_terms":null}'
    )
    raws = [valid_json, "not json at all", {"vendor": "incomplete"}]
    assert schema_valid_rate(raws) == pytest.approx(1 / 3)


def test_schema_valid_rate_empty_is_zero():
    assert schema_valid_rate([]) == 0.0


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def test_field_agreement_identical_is_one():
    assert field_agreement(GOLD, copy.deepcopy(GOLD)) == pytest.approx(1.0)


def test_field_agreement_none_is_zero():
    assert field_agreement(GOLD, None) == 0.0


def test_teacher_agreement_pairs():
    students = [copy.deepcopy(GOLD), copy.deepcopy(GOLD)]
    teachers = [GOLD, GOLD]
    assert teacher_agreement(students, teachers) == pytest.approx(1.0)


def test_teacher_agreement_length_mismatch_raises():
    with pytest.raises(ValueError):
        teacher_agreement([GOLD], [GOLD, GOLD])


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------

def test_percentile():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert percentile(vals, 50) == pytest.approx(0.3)
    assert percentile(vals, 0) == pytest.approx(0.1)
    assert percentile(vals, 100) == pytest.approx(0.5)


def test_percentile_single_value():
    assert percentile([0.42], 95) == pytest.approx(0.42)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)
