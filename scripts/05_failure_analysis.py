#!/usr/bin/env python
"""Categorized failure analysis of the student on the gold set (SPEC deliverable 5).

The money table says WHAT the residual quality gap is (field-F1 0.9647); this
report says WHERE it lives and WHICH cases still need the teacher. It replays
reports/gold_predictions.jsonl (input · gold · pred, produced by evaluate.py)
through the same field-aware comparison the eval uses (metrics.flatten_invoice
+ metrics.values_match — money tolerance included, so nothing here can disagree
with the shipped F1), then buckets every field-level error into a category:

  missing_line_item     gold line item with no positional counterpart in pred
  hallucinated_line_item pred line item beyond the gold count
  wrong_money           money field present in both but outside tolerance
  wrong_qty             qty mismatch beyond tolerance
  wrong_date            date field mismatch
  wrong_text            description/vendor/terms/currency text mismatch
  missing_field         scalar field in gold, absent/None in pred
  hallucinated_field    scalar field in pred, absent/None in gold

Writes reports/failure_analysis.md: per-category counts, per-field breakdown,
the worst documents with concrete gold-vs-pred excerpts, and the honest
"when to still call the teacher" guidance derived from the buckets.

Usage:
    python scripts/05_failure_analysis.py            # reads reports/gold_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from distil_task.metrics import flatten_invoice, values_match  # noqa: E402

LINE_PREFIX = "line_items["


def _leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def categorize(path: str, gold_v, pred_v) -> str | None:
    """Category for one field-path comparison; None = correct."""
    if values_match(path, gold_v, pred_v):
        return None
    is_line = path.startswith(LINE_PREFIX)
    if gold_v is not None and pred_v is None:
        return "missing_line_item" if is_line else "missing_field"
    if gold_v is None and pred_v is not None:
        return "hallucinated_line_item" if is_line else "hallucinated_field"
    leaf = _leaf(path)
    if leaf in ("unit_price", "total", "subtotal", "tax", "grand_total"):
        return "wrong_money"
    if leaf == "qty":
        return "wrong_qty"
    if leaf == "date":
        return "wrong_date"
    return "wrong_text"


def analyse(rows: list[dict]) -> tuple[Counter, Counter, list[dict]]:
    """Bucket every field error; return (category counts, per-leaf counts, doc records)."""
    cats: Counter = Counter()
    leaves: Counter = Counter()
    docs: list[dict] = []
    for row in rows:
        gold = flatten_invoice(row.get("gold") or {})
        pred = flatten_invoice(row.get("pred") or {})
        errors: list[dict] = []
        for path in sorted(set(gold) | set(pred)):
            cat = categorize(path, gold.get(path), pred.get(path))
            if cat is None:
                continue
            cats[cat] += 1
            leaves[_leaf(path)] += 1
            errors.append({"path": path, "cat": cat,
                           "gold": gold.get(path), "pred": pred.get(path)})
        total = len(set(gold) | set(pred))
        docs.append({"id": row.get("id", "?"), "n_fields": total,
                     "n_errors": len(errors), "errors": errors,
                     "input_head": (row.get("input") or "")[:110]})
    return cats, leaves, docs


GUIDANCE = {
    "missing_line_item": "long/dense receipts — the 0.5B student drops trailing "
                         "items; route documents with many line items to the teacher",
    "hallucinated_line_item": "noisy inputs where headers/totals get read as items",
    "wrong_money": "amounts beyond the tolerance — usually OCR-ish digit slips or "
                   "tax/subtotal confusion; a cheap grand-total==sum check catches most",
    "wrong_qty": "quantity/unit confusion (e.g. '2 x 500g')",
    "wrong_date": "ambiguous or non-ISO source formats (DD/MM vs MM/DD)",
    "wrong_text": "vendor/description normalization drift",
    "missing_field": "optional scalars (payment_terms, tax) absent from sparse inputs",
    "hallucinated_field": "student fills optional scalars the gold marks absent",
}


def write_report(out: Path, cats: Counter, leaves: Counter, docs: list[dict]) -> None:
    n_docs = len(docs)
    n_perfect = sum(1 for d in docs if d["n_errors"] == 0)
    n_fields = sum(d["n_fields"] for d in docs)
    n_errors = sum(d["n_errors"] for d in docs)
    worst = sorted(docs, key=lambda d: d["n_errors"], reverse=True)[:5]

    L = [
        "# Categorized failure analysis — student vs gold (SPEC deliverable 5)",
        "",
        f"_{n_docs} silver-grade gold documents (cross-model agreement, not "
        f"human-rated — see reports/eval_report.json) · {n_fields} compared fields "
        f"(same flatten + tolerance rules as the shipped eval) · "
        f"**{n_perfect}/{n_docs} documents fully correct** · "
        f"{n_errors} field errors ({n_errors / max(n_fields, 1):.1%} of fields). "
        "Regenerate: `python scripts/05_failure_analysis.py`._",
        "",
        "## Where the residual gap lives (by category)",
        "",
        "| Category | Errors | Share | Typical cause / routing guidance |",
        "|---|---:|---:|---|",
    ]
    for cat, n in cats.most_common():
        L.append(f"| `{cat}` | {n} | {n / max(n_errors, 1):.0%} | {GUIDANCE.get(cat, '')} |")
    if not cats:
        L.append("| — | 0 | — | no field errors on the gold set |")
    L += [
        "",
        "## Per-field breakdown",
        "",
        "| Field (leaf) | Errors |",
        "|---|---:|",
    ]
    for leaf, n in leaves.most_common():
        L.append(f"| `{leaf}` | {n} |")
    L += [
        "",
        "## Worst documents (gold vs pred excerpts)",
        "",
    ]
    for d in worst:
        if d["n_errors"] == 0:
            break
        L.append(f"### `{d['id']}` — {d['n_errors']} error(s) / {d['n_fields']} fields")
        L.append("")
        L.append(f"> {d['input_head']}…")
        L.append("")
        for e in d["errors"][:6]:
            L.append(f"- **{e['path']}** ({e['cat']}): gold=`{e['gold']}` → pred=`{e['pred']}`")
        L.append("")
    L += [
        "## When to still call the teacher (the honest routing rule)",
        "",
        "The buckets above are the escalation policy: keep the student for "
        "short/medium invoices (it is at or near teacher parity there) and route "
        "to the teacher when (a) the document has an unusually high line-item "
        "count, (b) the cheap arithmetic check `grand_total ≈ Σ line totals + tax` "
        "fails on the student's output, or (c) required scalars come back null. "
        "Categories, not vibes — each rule maps to a bucket measured above.",
        "",
    ]
    out.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default="reports/gold_predictions.jsonl")
    ap.add_argument("--out", default="reports/failure_analysis.md")
    args = ap.parse_args()

    src = _ROOT / args.predictions
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    cats, leaves, docs = analyse(rows)
    write_report(_ROOT / args.out, cats, leaves, docs)
    n_err = sum(cats.values())
    print(f"[failure-analysis] {len(rows)} docs, {n_err} field errors "
          f"across {len(cats)} categories -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
