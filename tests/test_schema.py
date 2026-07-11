"""Schema tests: pydantic validation + normalization of money, dates, currency."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from distil_task.schema import (
    Invoice,
    SchemaValidationError,
    export_json_schema,
    invoice_to_canonical_json,
    normalize_currency,
    parse_date_iso,
    parse_money,
    parse_quantity,
    try_validate_invoice,
    validate_invoice,
)


# ---------------------------------------------------------------------------
# Money normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$1,234.56", 1234.56),   # US thousands + decimal
        ("1.234,56", 1234.56),    # EU thousands + decimal
        ("1 234,56", 1234.56),    # space thousands, comma decimal
        ("12,50", 12.50),         # comma decimal only
        ("EUR 99.90", 99.90),     # currency prefix stripped
        ("1234", 1234.0),         # bare integer
        ("(45.00)", -45.00),      # accounting negative
        ("-7.5", -7.5),
        (1234.5, 1234.5),         # float passthrough
        (1000, 1000.0),           # int passthrough
        (Decimal("3.14"), 3.14),  # Decimal passthrough
    ],
)
def test_parse_money(raw, expected):
    assert parse_money(raw) == pytest.approx(expected)


def test_parse_money_rounds_to_cents():
    assert parse_money("1.239") == pytest.approx(1.24)


@pytest.mark.parametrize("bad", [True, False, "", "   ", "abc", None])
def test_parse_money_rejects_garbage(bad):
    with pytest.raises((ValueError, TypeError)):
        parse_money(bad)


def test_parse_quantity_strips_units():
    assert parse_quantity("2 pcs") == pytest.approx(2.0)
    assert parse_quantity("2.5 hrs") == pytest.approx(2.5)
    assert parse_quantity("0.75kg") == pytest.approx(0.75)
    assert parse_quantity(3) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Date normalization -> ISO-8601
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2024-01-15", "2024-01-15"),
        ("2024/01/15", "2024-01-15"),
        ("03/14/2024", "2024-03-14"),   # US month-first
        ("15.01.2024", "2024-01-15"),   # EU day-first, unambiguous (15 > 12)
        ("01/15/2024", "2024-01-15"),   # day > 12 disambiguates to MDY
        ("Jan 15, 2024", "2024-01-15"),
        ("15 January 2024", "2024-01-15"),
        ("15-Jan-2024", "2024-01-15"),
        ("20240115", "2024-01-15"),     # compact
    ],
)
def test_parse_date_iso(raw, expected):
    assert parse_date_iso(raw) == expected


def test_parse_date_from_objects():
    assert parse_date_iso(date(2024, 3, 14)) == "2024-03-14"
    assert parse_date_iso(datetime(2024, 3, 14, 9, 30)) == "2024-03-14"


def test_parse_date_ambiguous_defaults_to_month_first():
    # both <= 12 -> assume MDY (US style, dominant in the corpus)
    assert parse_date_iso("02/03/2024") == "2024-02-03"


@pytest.mark.parametrize("bad", ["", "not a date", "2024-13-40", "99/99/99"])
def test_parse_date_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_date_iso(bad)


# ---------------------------------------------------------------------------
# Currency normalization -> ISO-4217
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$", "USD"),
        ("US$", "USD"),
        ("€", "EUR"),
        ("£", "GBP"),
        ("usd", "USD"),
        ("EUR", "EUR"),
        ("euro", "EUR"),
        ("US Dollars", "USD"),
        ("pakistani rupee", "PKR"),
        ("Rs.", "PKR"),
        ("ZZZ", "ZZZ"),   # unknown but plausible 3-letter code accepted
    ],
)
def test_normalize_currency(raw, expected):
    assert normalize_currency(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "not-a-currency", "dollarz"])
def test_normalize_currency_rejects_garbage(bad):
    with pytest.raises(ValueError):
        normalize_currency(bad)


# ---------------------------------------------------------------------------
# Full Invoice validation + normalization
# ---------------------------------------------------------------------------

MESSY_INVOICE = {
    "vendor": "  Acme   Corp  ",
    "date": "03/14/2024",
    "currency": "$",
    "line_items": [
        {"description": "Widget  A", "qty": "2 pcs", "unit_price": "$6.49", "total": "12,98"},
    ],
    "subtotal": "12.98",
    "tax": "0",
    "grand_total": "$12.98",
    "payment_terms": "n/a",
}


def test_invoice_normalizes_all_fields():
    inv = Invoice.model_validate(MESSY_INVOICE)
    assert inv.vendor == "Acme Corp"          # whitespace collapsed
    assert inv.date == "2024-03-14"           # ISO date
    assert inv.currency == "USD"              # symbol -> code
    assert inv.subtotal == pytest.approx(12.98)
    assert inv.tax == pytest.approx(0.0)
    assert inv.grand_total == pytest.approx(12.98)
    assert inv.payment_terms is None          # "n/a" -> None
    li = inv.line_items[0]
    assert li.description == "Widget A"
    assert li.qty == pytest.approx(2.0)
    assert li.unit_price == pytest.approx(6.49)
    assert li.total == pytest.approx(12.98)


def test_invoice_keeps_real_payment_terms():
    data = dict(MESSY_INVOICE, payment_terms="Net 30")
    assert Invoice.model_validate(data).payment_terms == "Net 30"


def test_invoice_rejects_extra_fields():
    data = dict(MESSY_INVOICE, surprise="nope")
    with pytest.raises(Exception):
        Invoice.model_validate(data)


def test_invoice_requires_at_least_one_line_item():
    data = dict(MESSY_INVOICE, line_items=[])
    with pytest.raises(Exception):
        Invoice.model_validate(data)


def test_validate_invoice_accepts_json_string():
    inv = validate_invoice(invoice_to_canonical_json(Invoice.model_validate(MESSY_INVOICE)))
    assert inv.currency == "USD"


def test_validate_invoice_raises_structured_error_on_bad_json():
    with pytest.raises(SchemaValidationError):
        validate_invoice("{not json")


def test_try_validate_invoice_nonraising():
    good, err = try_validate_invoice(MESSY_INVOICE)
    assert good is not None and err is None
    bad, err2 = try_validate_invoice({"vendor": "x"})  # missing required fields
    assert bad is None and err2 is not None


def test_canonical_json_roundtrip_is_stable():
    inv = Invoice.model_validate(MESSY_INVOICE)
    once = invoice_to_canonical_json(inv)
    twice = invoice_to_canonical_json(validate_invoice(once))
    assert once == twice


def test_export_json_schema_shape():
    schema = export_json_schema()
    assert schema["title"] == "Invoice"
    assert "properties" in schema
    for key in ("vendor", "date", "currency", "line_items", "subtotal", "tax", "grand_total"):
        assert key in schema["properties"]
