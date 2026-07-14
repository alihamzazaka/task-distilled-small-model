"""Unit tests for the categorized failure analysis (scripts/05_failure_analysis.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "failure_analysis", _SCRIPTS / "05_failure_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FA = _load()


def test_correct_fields_are_not_categorized():
    assert FA.categorize("vendor", "Costco", "costco ") is None          # text norm
    assert FA.categorize("grand_total", 10.00, 10.005) is None           # money tol


def test_categories_map_to_buckets():
    assert FA.categorize("line_items[0].description", "Milk", None) == "missing_line_item"
    assert FA.categorize("line_items[2].total", None, 9.99) == "hallucinated_line_item"
    assert FA.categorize("grand_total", 10.0, 22.5) == "wrong_money"
    assert FA.categorize("line_items[0].qty", 2, 5) == "wrong_qty"
    assert FA.categorize("date", "2024-03-17", "2024-07-13") == "wrong_date"
    assert FA.categorize("vendor", "Costco", "Walmart") == "wrong_text"
    assert FA.categorize("payment_terms", "Net 30", None) == "missing_field"
    assert FA.categorize("tax", None, 1.5) == "hallucinated_field"


def test_analyse_counts_and_perfect_docs():
    rows = [
        {"id": "ok", "input": "x",
         "gold": {"vendor": "A", "line_items": [{"description": "d", "total": 1.0}]},
         "pred": {"vendor": "A", "line_items": [{"description": "d", "total": 1.0}]}},
        {"id": "bad", "input": "y",
         "gold": {"vendor": "A", "grand_total": 10.0},
         "pred": {"vendor": "B", "grand_total": 99.0}},
    ]
    cats, leaves, docs = FA.analyse(rows)
    assert sum(cats.values()) == 2
    assert cats["wrong_text"] == 1 and cats["wrong_money"] == 1
    assert leaves["vendor"] == 1 and leaves["grand_total"] == 1
    assert [d["n_errors"] for d in docs] == [0, 2]
