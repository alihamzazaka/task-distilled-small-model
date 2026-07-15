# Invoice-Extraction Gold — Rating Instructions

You are establishing the **human gold** that the student's headline accuracy is
measured against. Two raters label this sheet **independently**; we then compute
Cohen's κ (target **≥ 0.70**) and adjudicate every disagreement before the
number is trusted.

## The item you are judging
Each row has two things to compare:

- **source_document** — the raw invoice/receipt text (the ground truth).
- **model_extraction** — the student's extracted JSON.

## The decision rule (one binary label per row)
Put exactly one value in the **label** column:

- **`correct`** — *every* field in `model_extraction` matches the source
  document under the field rules below. Nothing wrong, nothing missing,
  nothing invented.
- **`incorrect`** — *at least one* field is wrong, missing, or hallucinated.

When a row is `incorrect`, briefly say which field in **notes**
(e.g. `date wrong: due date not invoice date`, `line_items[2].total missing`).
One wrong field is enough to make the whole row `incorrect` — do not average.

## Field rules (must match the extractor schema)
- **vendor** — merchant name as printed (drop addresses/slogans); keep the
  original script.
- **date** — ISO-8601 `YYYY-MM-DD`; the **invoice/issue** date, not the due date.
- **currency** — ISO-4217 code (`USD`, `EUR`, `PKR`, …), inferred from symbols
  or country when not printed.
- **line_items** — one entry per purchased line, in document order:
  `description` (as printed, whitespace-normalised), `qty` (default 1),
  `unit_price` (= `total / qty` if only a total is shown), `total` (printed
  line amount). Discounts are a line with a **negative** total; shipping/fees
  are line items too.
- **subtotal** — pre-tax sum. **tax** — total tax (`0` if none). **grand_total**
  — final amount due/paid. **payment_terms** — verbatim (`Net 30`) or `null`.

## Money & arithmetic
Money fields match within **±0.01**. Expect `sum(line totals) ≈ subtotal` and
`subtotal + tax ≈ grand_total` within ±0.02. If the document itself is
inconsistent, the **printed grand_total wins** — do not "fix" the document.

## Worked examples
- Extraction is byte-for-byte faithful to the receipt → **`correct`**.
- Every field right except a truncated `description`
  ("Frozen Pizza" vs "Frozen Pizza, Pepperoni") → **`incorrect`**
  (note: `line_items[2].description truncated`).
- `date` filled with the *due* date instead of the invoice date →
  **`incorrect`** (note: `date = due date`).
- A tax of `0.00` on a receipt that prints no tax → **`correct`**.

## What NOT to do
- Do not consult the other rater while labelling — independence is what makes κ
  meaningful.
- Do not leave a row blank; if the source text is not a real invoice, mark
  `incorrect` and note `not an invoice`.
