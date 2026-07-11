"""Fixed prompts for the distillation pipeline.

Two prompts live here:
1. The teacher *extraction* instruction (fixed for the whole run so the
   dataset is internally consistent): strict schema contract + 3 real
   few-shot examples.
2. The *seed generation* prompt that asks the teacher to invent diverse,
   realistic invoice/receipt texts.

The same extraction system prompt is reused verbatim when formatting the
student's SFT chat template, so teacher and student see one contract.
"""
from __future__ import annotations

import json

from .schema import export_json_schema

# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA_JSON = json.dumps(export_json_schema(), ensure_ascii=False, indent=2)

SCHEMA_CONTRACT = """\
Return ONLY a single JSON object — no markdown fences, no commentary, no
trailing text. The object MUST have exactly these keys:

  vendor        string  — merchant/vendor name as printed
  date          string  — invoice/receipt date in ISO-8601 (YYYY-MM-DD)
  currency      string  — ISO-4217 code (USD, EUR, GBP, ...). Infer from
                          symbols ($ -> USD, € -> EUR, £ -> GBP) or context.
  line_items    array   — one object per purchased line, each with exactly:
                            description (string), qty (number),
                            unit_price (number), total (number)
  subtotal      number  — pre-tax total of line items
  tax           number  — total tax amount; 0 if no tax is shown
  grand_total   number  — final amount due/paid
  payment_terms string|null — terms as printed (e.g. "Net 30", "Due on
                          receipt"); null if the document shows none

Rules:
- All monetary values are plain JSON numbers (1234.56), never strings,
  never with currency symbols or thousands separators.
- qty may be fractional (hours, kg).
- If a line shows only qty and total, compute unit_price = total / qty.
- If subtotal is not printed, compute it as the sum of line totals.
- Do not invent fields. Do not add fields. Do not omit fields.
- If the document is ambiguous, choose the most plausible reading; never
  leave a required field empty."""

# ---------------------------------------------------------------------------
# Few-shot examples (3 real, layout-diverse examples)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "input": """\
NORTHWIND OFFICE SUPPLY CO.
1401 Elm Street, Suite 300, Dallas TX 75201
INVOICE #INV-2024-0387          Date: 03/14/2024

Bill To: Acme Analytics LLC

QTY   DESCRIPTION                      UNIT PRICE      AMOUNT
 5    A4 Copy Paper, 500-sheet ream       $6.49         $32.45
 2    Laser Toner Cartridge HP 26X       $89.99        $179.98
12    Ballpoint Pens (box of 10)          $3.25         $39.00

                                   SUBTOTAL:           $251.43
                                   SALES TAX (8.25%):   $20.74
                                   TOTAL DUE:          $272.17

Terms: Net 30. A 1.5% monthly late fee applies to overdue balances.""",
        "output": {
            "vendor": "Northwind Office Supply Co.",
            "date": "2024-03-14",
            "currency": "USD",
            "line_items": [
                {"description": "A4 Copy Paper, 500-sheet ream", "qty": 5, "unit_price": 6.49, "total": 32.45},
                {"description": "Laser Toner Cartridge HP 26X", "qty": 2, "unit_price": 89.99, "total": 179.98},
                {"description": "Ballpoint Pens (box of 10)", "qty": 12, "unit_price": 3.25, "total": 39.0},
            ],
            "subtotal": 251.43,
            "tax": 20.74,
            "grand_total": 272.17,
            "payment_terms": "Net 30",
        },
    },
    {
        "input": """\
*** CAFÉ LUMIÈRE ***
14 Rue des Abbesses, 75018 Paris
TVA FR-88-512-334-991
--------------------------------
15.11.2024   19:42   Table 7
--------------------------------
2x Croque Monsieur     9,50   19,00
1x Soupe à l'oignon           8,50
3x Café allongé        3,20    9,60
--------------------------------
Sous-total                    37,10
TVA 10%                        3,71
TOTAL EUR                     40,81
--------------------------------
CB **** 4412 — payé
Merci de votre visite !""",
        "output": {
            "vendor": "Café Lumière",
            "date": "2024-11-15",
            "currency": "EUR",
            "line_items": [
                {"description": "Croque Monsieur", "qty": 2, "unit_price": 9.5, "total": 19.0},
                {"description": "Soupe à l'oignon", "qty": 1, "unit_price": 8.5, "total": 8.5},
                {"description": "Café allongé", "qty": 3, "unit_price": 3.2, "total": 9.6},
            ],
            "subtotal": 37.10,
            "tax": 3.71,
            "grand_total": 40.81,
            "payment_terms": None,
        },
    },
    {
        "input": """\
TAX INVOICE
Khan Brothers Hardware & Electric
Shop 12, Saddar Bazaar, Karachi
NTN 4521178-3                        Invoice No: KB-7734
Dated: 7 January 2025

Item                              Qty    Rate (PKR)    Amount
PVC Pipe 1.5in (per meter)        24        185        4,440
Circuit Breaker 32A Schneider      6      1,250        7,500
Insulation Tape                   10         60          600

                              Sub Total:              12,540
                              GST 18%:                 2,257.20
                              Grand Total (PKR):      14,797.20

Payment due within 15 days of invoice date.""",
        "output": {
            "vendor": "Khan Brothers Hardware & Electric",
            "date": "2025-01-07",
            "currency": "PKR",
            "line_items": [
                {"description": "PVC Pipe 1.5in (per meter)", "qty": 24, "unit_price": 185.0, "total": 4440.0},
                {"description": "Circuit Breaker 32A Schneider", "qty": 6, "unit_price": 1250.0, "total": 7500.0},
                {"description": "Insulation Tape", "qty": 10, "unit_price": 60.0, "total": 600.0},
            ],
            "subtotal": 12540.0,
            "tax": 2257.20,
            "grand_total": 14797.20,
            "payment_terms": "Payment due within 15 days of invoice date",
        },
    },
]


def _render_few_shots() -> str:
    blocks = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        blocks.append(
            f"### Example {i}\n"
            f"<document>\n{ex['input']}\n</document>\n"
            f"Output:\n{json.dumps(ex['output'], ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Teacher extraction prompt (FIXED for the entire run)
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = f"""\
You are a precise invoice/receipt data-extraction engine. Given the raw
text of one invoice or receipt, extract its structured data.

{SCHEMA_CONTRACT}

JSON Schema (authoritative):
{OUTPUT_SCHEMA_JSON}

{_render_few_shots()}"""

# The student is trained with a SHORT system prompt (cheaper tokens at
# inference; the few-shot behavior is distilled into the weights).
STUDENT_SYSTEM_PROMPT = f"""\
You are a precise invoice/receipt data-extraction engine. Given the raw
text of one invoice or receipt, extract its structured data.

{SCHEMA_CONTRACT}"""


def build_extraction_user_prompt(document_text: str) -> str:
    """User turn for both teacher labeling and student SFT/inference."""
    return f"<document>\n{document_text}\n</document>\nOutput:"


# ---------------------------------------------------------------------------
# Seed-input diversity prompt (synthetic invoice generation)
# ---------------------------------------------------------------------------

SEED_GENERATION_SYSTEM_PROMPT = """\
You generate REALISTIC synthetic invoice and receipt texts for training a
data-extraction model. Output plain-text documents exactly as they would
appear after OCR or copy-paste — imperfect alignment, real-world noise.

Diversity requirements (vary aggressively across documents):
- Document type: formal B2B invoices, retail receipts, restaurant bills,
  utility bills, freelance invoices, medical bills, hotel folios, e-commerce
  order confirmations, repair-shop work orders.
- Layout: column tables, dashed-line receipts, inline "2x Item ... 9.50"
  styles, right-aligned totals, headers/footers, VAT/GST/sales-tax blocks.
- Locale: vendor names and addresses from many countries (USA, France,
  Germany, Pakistan, India, Japan, Brazil, UK, UAE, Mexico, Poland ...);
  vendor names may be in local languages/scripts, but keep the body mostly
  Latin-script so OCR text is plausible.
- Currency: many ISO currencies AND their symbols/local notation
  ($1,234.56 / 1.234,56 € / ¥12,300 / Rs. 14,500 / £45.20).
- Date formats: 2024-03-14, 03/14/2024, 14.03.2024, 14 March 2024, etc.
- Amount scale: from a $3.75 coffee receipt to a $250,000 equipment invoice.
- Line items: between 1 and 12 lines; fractional quantities (2.5 hrs,
  0.75 kg); occasional discounts printed as separate lines.
- Payment terms: sometimes present ("Net 30", "Due on receipt", "50%
  advance"), often absent (receipts usually have none).
- Noise: occasional OCR-style artifacts (stray asterisks, doubled spaces,
  broken column alignment) in ~20% of documents — but every document must
  still contain enough information for a human to extract vendor, date,
  currency, all line items (description, qty, unit price, total), subtotal,
  tax and grand total. Numbers must be arithmetically consistent (line
  totals sum to subtotal; subtotal + tax = grand total, within rounding).

Return ONLY a JSON array of strings, each string being one complete
document text. No markdown fences, no commentary."""


def build_seed_generation_user_prompt(n: int, batch_index: int, hint: str | None = None) -> str:
    """Ask for `n` documents. `batch_index` is echoed so different batches
    aren't near-identical even at the same temperature; `hint` can steer a
    batch toward a domain (e.g. 'restaurant receipts in Japan')."""
    extra = f"\nDomain steering hint for this batch: {hint}" if hint else ""
    return (
        f"Generate {n} diverse synthetic invoice/receipt documents "
        f"(batch #{batch_index}; make this batch clearly different from "
        f"typical previous batches — pick uncommon combinations of document "
        f"type, country, currency and layout).{extra}\n"
        f"Return a JSON array of {n} strings."
    )
