#!/usr/bin/env python
"""Active-learning sampler → the human-gold rating sheet.

Reads the existing student predictions (``reports/gold_predictions.jsonl``),
scores each pair for *informativeness* (low field-F1, schema-repair triggered,
exact-match miss), draws a stratified-by-currency slice of the most-informative
pairs, and writes a self-contained rating job:

    data/gold/rating_sheet.csv          (id, source_document, model_extraction,
                                          blank label, blank notes)
    data/gold/rating_instructions.md    (rubric + decision rule + examples)
    reports/gold_sample_predictions.jsonl  (per-id model signals, for the scorer)

    python scripts/gold_sample.py --n 24

Pure stdlib + the project's own metrics/schema helpers — no GPU, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from distil_task.gold_pipeline import (                       # noqa: E402
    RATING_SHEET_FIELDS,
    SampleItem,
    informativeness,
    stratified_sample,
    write_rating_sheet,
)
from distil_task.metrics import document_exact_match, field_score  # noqa: E402
from distil_task.schema import try_validate_invoice             # noqa: E402

DEFAULT_PREDICTIONS = _ROOT / "reports" / "gold_predictions.jsonl"
DEFAULT_SHEET = _ROOT / "data" / "gold" / "rating_sheet.csv"
DEFAULT_INSTRUCTIONS = _ROOT / "data" / "gold" / "rating_instructions.md"
DEFAULT_SIGNALS = _ROOT / "reports" / "gold_sample_predictions.jsonl"

INSTRUCTIONS = """\
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
"""


def build_items(records: list[dict]) -> list[SampleItem]:
    """Score every prediction record for informativeness."""
    items: list[SampleItem] = []
    for rec in records:
        rid = rec.get("id") or ""
        gold = rec.get("gold") or {}
        pred = rec.get("pred")  # may be None for unparseable outputs
        raw = rec.get("raw")
        fs = field_score(pred, gold)
        f1 = fs.f1
        em = document_exact_match(pred, gold)
        # schema-repair triggered ≈ the raw output did not pass the strict gate
        # (so constrained-repair/normalisation was needed to get `pred`).
        schema_repaired = raw is not None and try_validate_invoice(raw)[0] is None
        stratum = str((pred or gold).get("currency") or "UNK")
        score = informativeness(
            field_f1=f1, schema_repaired=schema_repaired, exact_match=em)
        items.append(SampleItem(
            id=rid, stratum=stratum, informativeness=score,
            payload={
                "source_document": rec.get("input", ""),
                "model_extraction": json.dumps(pred, ensure_ascii=False,
                                               sort_keys=True) if pred is not None else "",
                "field_f1": round(f1, 4),
                "exact_match": em,
                "schema_repaired": schema_repaired,
                "currency": stratum,
            },
        ))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    ap.add_argument("--n", type=int, default=24,
                    help="number of pairs to hand-rate (real slice: 150-200)")
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS)
    ap.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    args = ap.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"predictions file not found: {args.predictions}\n"
                         "Run scripts/evaluate.py first to produce it.")

    records = [json.loads(line) for line in
               args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("no prediction records found")

    items = build_items(records)
    picked = stratified_sample(items, args.n)

    rows = [{
        "id": it.id,
        "source_document": it.payload["source_document"],
        "model_extraction": it.payload["model_extraction"],
        "label": "",
        "notes": "",
    } for it in picked]
    write_rating_sheet(args.sheet, rows, RATING_SHEET_FIELDS)
    args.instructions.parent.mkdir(parents=True, exist_ok=True)
    args.instructions.write_text(INSTRUCTIONS, encoding="utf-8")

    # Per-id model signals (used by the scorer for model-vs-human agreement).
    args.signals.parent.mkdir(parents=True, exist_ok=True)
    with args.signals.open("w", encoding="utf-8") as fh:
        for it in picked:
            fh.write(json.dumps({
                "id": it.id,
                "field_f1": it.payload["field_f1"],
                "exact_match": it.payload["exact_match"],
                "schema_repaired": it.payload["schema_repaired"],
                "informativeness": round(it.informativeness, 4),
            }) + "\n")

    strata: dict[str, int] = {}
    for it in picked:
        strata[it.stratum] = strata.get(it.stratum, 0) + 1
    print(f"[gold_sample] {len(records)} candidates -> picked {len(picked)}")
    print(f"[gold_sample] strata (currency): "
          f"{', '.join(f'{k}={v}' for k, v in sorted(strata.items()))}")
    print(f"[gold_sample] informativeness range: "
          f"{picked[-1].informativeness:.3f} .. {picked[0].informativeness:.3f}")
    print(f"[gold_sample] wrote {args.sheet}")
    print(f"[gold_sample] wrote {args.instructions}")
    print(f"[gold_sample] wrote {args.signals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
