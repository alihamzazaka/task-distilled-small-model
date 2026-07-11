"""Invoice/receipt output schema (Phase 0 contract).

Pydantic v2 models with aggressive-but-safe normalization so that the
teacher's (and later the student's) slightly-varied surface forms coerce
into one canonical representation:

- money fields: "$1,234.56", "1.234,56", "1 234,56 EUR" -> float 1234.56
- dates: many common formats -> ISO-8601 "YYYY-MM-DD"
- currency: symbols/names ("$", "euro", "US Dollars") -> ISO-4217 code

Also exports the JSON Schema used in the teacher prompt contract and by
`jsonschema` consumers.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Currency normalization
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS = {
    "$": "USD", "US$": "USD", "USD$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₨": "PKR", "RS": "PKR", "RS.": "PKR",
    "A$": "AUD", "AU$": "AUD",
    "C$": "CAD", "CA$": "CAD",
    "CHF": "CHF",
    "R$": "BRL",
    "KR": "SEK",
    "ZŁ": "PLN", "ZL": "PLN",
    "฿": "THB",
    "₩": "KRW",
    "₺": "TRY",
    "د.إ": "AED", "DH": "AED", "DHS": "AED",
}

_CURRENCY_NAMES = {
    "dollar": "USD", "dollars": "USD", "us dollar": "USD", "us dollars": "USD",
    "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP", "pound sterling": "GBP", "gbp sterling": "GBP",
    "yen": "JPY",
    "rupee": "INR", "rupees": "INR", "indian rupee": "INR", "indian rupees": "INR",
    "pakistani rupee": "PKR", "pakistani rupees": "PKR",
    "yuan": "CNY", "renminbi": "CNY", "rmb": "CNY",
    "swiss franc": "CHF", "swiss francs": "CHF",
    "canadian dollar": "CAD", "canadian dollars": "CAD",
    "australian dollar": "AUD", "australian dollars": "AUD",
    "dirham": "AED", "dirhams": "AED",
    "riyal": "SAR", "riyals": "SAR", "saudi riyal": "SAR",
    "real": "BRL", "reais": "BRL",
    "won": "KRW",
    "krona": "SEK", "kronor": "SEK",
    "zloty": "PLN",
    "lira": "TRY",
    "peso": "MXN", "pesos": "MXN", "mexican peso": "MXN",
    "baht": "THB",
}

# ISO-4217 codes we accept without complaint (extend as needed; unknown
# 3-letter uppercase codes are still accepted to avoid over-rejecting).
KNOWN_CURRENCY_CODES = {
    "USD", "EUR", "GBP", "JPY", "CNY", "INR", "PKR", "AUD", "CAD", "CHF",
    "SEK", "NOK", "DKK", "PLN", "TRY", "AED", "SAR", "BRL", "MXN", "KRW",
    "THB", "SGD", "HKD", "NZD", "ZAR", "EGP", "NGN", "KES", "IDR", "MYR",
    "PHP", "VND", "BDT", "LKR", "QAR", "KWD", "BHD", "OMR", "JOD", "ILS",
    "CZK", "HUF", "RON", "BGN", "HRK", "RSD", "UAH", "RUB", "ARS", "CLP",
    "COP", "PEN", "TWD",
}


def normalize_currency(value: str) -> str:
    """Coerce a currency symbol / name / code into an ISO-4217 code."""
    if not isinstance(value, str):
        raise ValueError(f"currency must be a string, got {type(value).__name__}")
    raw = value.strip()
    if not raw:
        raise ValueError("currency is empty")
    upper = raw.upper()
    if upper in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[upper]
    if raw in _CURRENCY_SYMBOLS:  # symbols like € are case-insensitive anyway
        return _CURRENCY_SYMBOLS[raw]
    lower = raw.lower()
    if lower in _CURRENCY_NAMES:
        return _CURRENCY_NAMES[lower]
    if re.fullmatch(r"[A-Z]{3}", upper):
        return upper  # accept any 3-letter code (known or plausible)
    # last chance: a single non-alnum char that is a known symbol
    for sym, code in _CURRENCY_SYMBOLS.items():
        if raw == sym:
            return code
    raise ValueError(f"unrecognized currency: {value!r}")


# ---------------------------------------------------------------------------
# Money / decimal normalization
# ---------------------------------------------------------------------------

_MONEY_STRIP_RE = re.compile(r"[^\d,.\-()]")


def parse_money(value: Any) -> float:
    """Parse a money amount from float/int/str with locale-tolerant handling.

    Handles: "$1,234.56"  "1.234,56"  "1 234,56"  "(45.00)" (negative)
             "1234"       "12,50"     "EUR 99.90"
    Returns a float rounded to 2 decimal places.
    """
    if isinstance(value, bool):
        raise ValueError("money field cannot be a boolean")
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, Decimal):
        return round(float(value), 2)
    if not isinstance(value, str):
        raise ValueError(f"cannot parse money from {type(value).__name__}")

    s = value.strip()
    if not s:
        raise ValueError("money field is empty")
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    s = _MONEY_STRIP_RE.sub("", s.replace(" ", " ")).strip()
    s = s.replace("(", "").replace(")", "")
    if s.startswith("-"):
        negative, s = True, s[1:]
    s = s.strip()
    if not s or not re.search(r"\d", s):
        raise ValueError(f"no digits in money value {value!r}")

    has_comma, has_dot = "," in s, "." in s
    if has_comma and has_dot:
        # Rightmost separator is the decimal point.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")   # 1.234,56 -> 1234.56
        else:
            s = s.replace(",", "")                     # 1,234.56 -> 1234.56
    elif has_comma:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) != 3:
            s = s.replace(",", ".")                    # 12,50 -> 12.50
        elif len(parts) == 2 and len(parts[0]) <= 3 and len(parts[1]) == 3:
            s = s.replace(",", "")                     # 1,234 -> 1234 (thousands)
        else:
            s = s.replace(",", "")                     # 1,234,567
    elif has_dot:
        parts = s.split(".")
        if len(parts) > 2:
            s = s.replace(".", "")                     # 1.234.567 -> 1234567
        elif len(parts) == 2 and len(parts[0]) > 3 and len(parts[1]) == 3:
            # e.g. "12345.678" is ambiguous; keep as decimal (safer for totals)
            pass
    try:
        d = Decimal(s)
    except InvalidOperation as e:
        raise ValueError(f"cannot parse money value {value!r}") from e
    if negative:
        d = -d
    return round(float(d), 2)


def parse_quantity(value: Any) -> float:
    """Quantities may be fractional (hours, kg). Same parsing as money but
    rounded to 4 dp."""
    if isinstance(value, bool):
        raise ValueError("qty cannot be a boolean")
    if isinstance(value, (int, float, Decimal)):
        return round(float(value), 4)
    if isinstance(value, str):
        s = value.strip().lower()
        s = re.sub(r"(pcs|pc|units?|x|hrs?|hours?|kg|ea)\.?$", "", s).strip()
        return round(parse_money(s), 4)  # reuse locale-tolerant number parsing
    raise ValueError(f"cannot parse qty from {type(value).__name__}")


# ---------------------------------------------------------------------------
# Date normalization -> ISO-8601
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_date_iso(value: Any) -> str:
    """Coerce a date in common invoice formats to ISO-8601 'YYYY-MM-DD'.

    Supported: 2024-01-15 / 2024/01/15 / 15.01.2024 / 01/15/2024 /
    15/01/2024 / Jan 15, 2024 / 15 January 2024 / 20240115.
    Ambiguous numeric d/m vs m/d: if one part > 12 it disambiguates;
    otherwise assumes month-first (US style, most common in the corpus).
    """
    if isinstance(value, (datetime, date)):
        return (value.date() if isinstance(value, datetime) else value).isoformat()
    if not isinstance(value, str):
        raise ValueError(f"cannot parse date from {type(value).__name__}")
    s = value.strip()
    if not s:
        raise ValueError("date is empty")

    # ISO first (with optional time part)
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return _build_iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Compact yyyymmdd
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s)
    if m:
        return _build_iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Textual month: "Jan 15, 2024" / "15 January 2024" / "15-Jan-2024"
    m = re.match(
        r"^(?:(\d{1,2})(?:st|nd|rd|th)?[\s\-/,.]+)?([A-Za-z]{3,9})[\s\-/,.]+(\d{1,2})?(?:st|nd|rd|th)?[\s,]*?(\d{4})$",
        s,
    )
    if m:
        d1, mon_name, d2, year = m.groups()
        mon = _MONTHS.get(mon_name.lower())
        if mon:
            day = d1 or d2
            if day:
                return _build_iso(int(year), mon, int(day))

    # Numeric d/m/y or m/d/y (2- or 4-digit year)
    m = re.match(r"^(\d{1,2})[\s./-](\d{1,2})[\s./-](\d{2,4})$", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y < 70 else 1900
        if a > 12 and b <= 12:
            day, mon = a, b          # unambiguous DMY
        elif b > 12 and a <= 12:
            mon, day = a, b          # unambiguous MDY
        else:
            mon, day = a, b          # ambiguous -> assume MDY
        return _build_iso(y, mon, day)

    raise ValueError(f"unrecognized date format: {value!r}")


def _build_iso(year: int, month: int, day: int) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError as e:
        raise ValueError(f"invalid calendar date {year}-{month}-{day}") from e


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

MONEY_FIELDS = {"unit_price", "total", "subtotal", "tax", "grand_total"}


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(..., min_length=1, description="What was purchased.")
    qty: float = Field(..., description="Quantity (may be fractional).")
    unit_price: float = Field(..., description="Price per unit.")
    total: float = Field(..., description="Line total (qty x unit_price, minus line discounts).")

    @field_validator("description", mode="before")
    @classmethod
    def _clean_desc(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = re.sub(r"\s+", " ", v).strip()
        return v

    @field_validator("qty", mode="before")
    @classmethod
    def _qty(cls, v: Any) -> float:
        return parse_quantity(v)

    @field_validator("unit_price", "total", mode="before")
    @classmethod
    def _money(cls, v: Any) -> float:
        return parse_money(v)


class Invoice(BaseModel):
    """Canonical extraction target for one invoice/receipt."""

    model_config = ConfigDict(extra="forbid")

    vendor: str = Field(..., min_length=1, description="Vendor/merchant name as printed.")
    date: str = Field(..., description="Invoice date, ISO-8601 YYYY-MM-DD.")
    currency: str = Field(..., description="ISO-4217 currency code, e.g. USD.")
    line_items: list[LineItem] = Field(..., min_length=1)
    subtotal: float = Field(..., description="Sum of line totals before tax/discounts.")
    tax: float = Field(..., description="Total tax amount (0 if none shown).")
    grand_total: float = Field(..., description="Final amount due/paid.")
    payment_terms: Optional[str] = Field(
        None, description="Payment terms as printed (e.g. 'Net 30'), null if absent."
    )

    @field_validator("vendor", mode="before")
    @classmethod
    def _clean_vendor(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = re.sub(r"\s+", " ", v).strip()
        return v

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> str:
        return parse_date_iso(v)

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, v: Any) -> str:
        return normalize_currency(v)

    @field_validator("subtotal", "tax", "grand_total", mode="before")
    @classmethod
    def _money(cls, v: Any) -> float:
        if v is None:
            raise ValueError("money field is required (use 0 for absent tax)")
        return parse_money(v)

    @field_validator("payment_terms", mode="before")
    @classmethod
    def _terms(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = re.sub(r"\s+", " ", v).strip()
            if not v or v.lower() in {"none", "n/a", "na", "-", "null"}:
                return None
        return v


# ---------------------------------------------------------------------------
# Validation entry points
# ---------------------------------------------------------------------------

class SchemaValidationError(ValueError):
    """Raised when a candidate output does not satisfy the invoice schema."""


def validate_invoice(obj: Any) -> Invoice:
    """Strict gate: dict/JSON-string -> validated+normalized Invoice.
    Raises SchemaValidationError with a readable message on failure."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"not valid JSON: {e}") from e
    if isinstance(obj, Invoice):
        return obj
    if not isinstance(obj, dict):
        raise SchemaValidationError(
            f"expected a JSON object, got {type(obj).__name__}"
        )
    try:
        return Invoice.model_validate(obj)
    except Exception as e:  # pydantic.ValidationError -> readable message
        raise SchemaValidationError(str(e)) from e


def try_validate_invoice(obj: Any) -> tuple[Optional[Invoice], Optional[str]]:
    """Non-raising variant: returns (Invoice, None) or (None, error_message)."""
    try:
        return validate_invoice(obj), None
    except SchemaValidationError as e:
        return None, str(e)


def invoice_to_canonical_json(inv: Invoice, indent: int | None = None) -> str:
    """Deterministic JSON serialization (stable key order, 2dp floats stay floats)."""
    return json.dumps(inv.model_dump(mode="json"), ensure_ascii=False, indent=indent)


def export_json_schema() -> dict[str, Any]:
    """JSON Schema of the Invoice contract (for prompts, docs, jsonschema)."""
    schema = Invoice.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Invoice"
    return schema


if __name__ == "__main__":
    print(json.dumps(export_json_schema(), indent=2))
