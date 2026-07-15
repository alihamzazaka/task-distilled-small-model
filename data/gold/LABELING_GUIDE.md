# Gold Test Set — Human Labeling Guide

The student's headline quality claim is measured on THIS set, so it must be
human-verified. **The student never trains on these items.**

## Your job
For every record in `gold_test.jsonl`:
1. Read `input` (the raw invoice/receipt text).
2. Check every field of `output` (the teacher's proposal) against the text.
3. Correct any wrong field **in place**.
4. Set `"human_verified": true` once the whole record is correct.

Records left with `human_verified: false` are EXCLUDED from the eval.

## Field rules (must match `distil_task/schema.py`)
- **vendor** — merchant name as printed (trim addresses/slogans). Keep the
  original language/script.
- **date** — ISO-8601 `YYYY-MM-DD`. If several dates appear, use the
  invoice/issue date, not the due date.
- **currency** — ISO-4217 code (`USD`, `EUR`, `PKR`, ...). Infer from
  symbols or country context when no code is printed.
- **line_items** — one entry per purchased line, in document order:
  - `description` — as printed, whitespace-normalized.
  - `qty` — number; fractional allowed (2.5 hours). If no qty printed, 1.
  - `unit_price` — per-unit price. If only a line total is shown,
    `unit_price = total / qty`.
  - `total` — the printed line amount.
  - Discounts printed as their own line are a line item with a negative
    total; shipping/fees lines are line items too.
- **subtotal** — pre-tax sum. If not printed, sum of line totals.
- **tax** — total tax; `0` when none is shown (typical for receipts).
- **grand_total** — the final amount due/paid.
- **payment_terms** — verbatim terms ("Net 30", "Due on receipt") or
  `null` when the document shows none.

## Arithmetic sanity
`sum(line totals) ≈ subtotal` and `subtotal + tax ≈ grand_total` within
±0.02 (rounding). If the document itself is inconsistent, trust the
**printed grand_total** and note the discrepancy — do NOT "fix" the
document.

## What to do with truly broken items
If the text is not actually an invoice/receipt, or is missing so much that
a human cannot extract the required fields, delete the record entirely and
note its `id` in `gold_removed.txt` (create the file next to this guide).
