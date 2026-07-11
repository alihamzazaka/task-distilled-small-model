#!/usr/bin/env python
"""Phase 1, step 2 — filter ruthlessly, split without leakage, carve gold.

Gates (in order, drop reasons logged):
1. schema gate        — re-validate every labeled output (belt & braces)
2. consistency gate   — field agreement vs the second teacher pass
3. dedup              — near-duplicate inputs removed (embedding cosine on
                        the GPU box, char-ngram Jaccard fallback anywhere)

Then: shuffled train/dev/test split (dedup already global, so cross-split
leakage is prevented by construction; an explicit leak check runs anyway),
plus two carve-outs from the *test* pool:
- gold_test.jsonl        — gold_size items, human_verified:false template
                           for hand verification (LABELING_GUIDE.md written
                           next to it)
- agreement_pool.jsonl   — fresh unseen inputs (with teacher outputs) for
                           the teacher-agreement eval

Outputs: data/splits/{train,dev,test,agreement_pool}.jsonl,
         data/gold/{gold_test.jsonl,LABELING_GUIDE.md},
         reports/filtering_report.json
Runs on the laptop (pure Python fallback path).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from distil_task.config import ensure_parent, load_config, resolve_path
from distil_task.filtering import (
    cross_split_leak_check,
    dedup_texts,
    run_consistency_filter,
    schema_gate,
)
from distil_task.io_utils import read_jsonl, write_jsonl

LABELING_GUIDE = """\
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
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dedup-method", default=None, choices=["auto", "embedding", "jaccard"],
                    help="override filtering.dedup_method")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fcfg = cfg["filtering"]
    scfg = cfg["splits"]
    rng = random.Random(int(scfg["seed"]))

    labeled = read_jsonl(resolve_path(cfg, "labeled_file"))
    report: dict = {"input_records": len(labeled), "drops": {}}
    print(f"[load] {len(labeled)} labeled records")

    # ---- gate 1: schema --------------------------------------------------
    g1 = schema_gate(labeled)
    report["drops"]["schema"] = len(g1.dropped)
    print(f"[schema] kept {len(g1.kept)}, dropped {len(g1.dropped)}")

    # ---- gate 2: consistency --------------------------------------------
    records = g1.kept
    consistency_dropped = []
    second_path = resolve_path(cfg, "consistency_file")
    if bool(fcfg["consistency_enabled"]) and second_path.exists():
        second = {str(r["id"]): r.get("output") for r in read_jsonl(second_path)}
        g2 = run_consistency_filter(
            records,
            second,
            threshold=float(fcfg["consistency_threshold"]),
            money_abs_tol=float(fcfg["money_abs_tol"]),
        )
        records, consistency_dropped = g2.kept, g2.dropped
        print(f"[consistency] kept {len(records)}, dropped {len(consistency_dropped)}")
    else:
        print("[consistency] skipped (disabled or no second-pass file)")
    report["drops"]["consistency"] = len(consistency_dropped)

    # ---- gate 3: dedup (global, before splitting -> no cross-split dupes)
    texts = [r["input"] for r in records]
    method = args.dedup_method or str(fcfg["dedup_method"])
    kept_idx, dup_pairs, method_used = dedup_texts(
        texts,
        threshold=float(fcfg["dedup_threshold"]),
        method=method,
        embedding_model=str(fcfg["embedding_model"]),
    )
    dup_dropped = [
        {**records[i], "_drop_reason": f"near-dup of {records[j]['id']} (sim={s})"}
        for i, j, s in dup_pairs
    ]
    records = [records[i] for i in kept_idx]
    report["drops"]["near_duplicate"] = len(dup_dropped)
    report["dedup_method"] = method_used
    print(f"[dedup:{method_used}] kept {len(records)}, dropped {len(dup_dropped)}")

    # ---- split ------------------------------------------------------------
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * float(scfg["train_ratio"]))
    n_dev = int(n * float(scfg["dev_ratio"]))
    train, dev, test_pool = (
        records[:n_train],
        records[n_train : n_train + n_dev],
        records[n_train + n_dev :],
    )

    # carve gold + agreement pool out of the held-out test pool
    gold_size = min(int(scfg["gold_size"]), len(test_pool))
    agree_size = min(int(scfg["agreement_pool_size"]), len(test_pool) - gold_size)
    gold, agreement_pool, test = (
        test_pool[:gold_size],
        test_pool[gold_size : gold_size + agree_size],
        test_pool[gold_size + agree_size :],
    )

    # ---- explicit leak check (should be clean by construction) ------------
    leaks = cross_split_leak_check(
        [r["input"] for r in train],
        [r["input"] for r in gold + test + dev],
        threshold=float(fcfg["dedup_threshold"]),
    )
    report["cross_split_leaks"] = len(leaks)
    if leaks:
        print(f"[WARN] {len(leaks)} cross-split near-dup pairs survived — investigate!")

    # ---- write ------------------------------------------------------------
    splits_dir = resolve_path(cfg, "splits_dir")
    gold_dir = resolve_path(cfg, "gold_dir")

    def strip_meta(r: dict) -> dict:
        return {k: v for k, v in r.items() if not k.startswith("_")}

    write_jsonl(splits_dir / "train.jsonl", (strip_meta(r) for r in train))
    write_jsonl(splits_dir / "dev.jsonl", (strip_meta(r) for r in dev))
    write_jsonl(splits_dir / "test.jsonl", (strip_meta(r) for r in test))
    write_jsonl(splits_dir / "agreement_pool.jsonl", (strip_meta(r) for r in agreement_pool))

    gold_records = [
        {
            "id": r["id"],
            "input": r["input"],
            "output": r["output"],          # teacher proposal — humans CORRECT this
            "human_verified": False,        # flip to true after verification
            "notes": "",
        }
        for r in gold
    ]
    write_jsonl(gold_dir / "gold_test.jsonl", gold_records)
    ensure_parent(gold_dir / "LABELING_GUIDE.md").write_text(LABELING_GUIDE, encoding="utf-8")

    report["splits"] = {
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "gold_test": len(gold_records),
        "agreement_pool": len(agreement_pool),
    }
    dropped_all = g1.dropped + consistency_dropped + dup_dropped
    write_jsonl(splits_dir / "dropped.jsonl", dropped_all)
    report["total_dropped"] = len(dropped_all)

    report_path = ensure_parent(resolve_path(cfg, "reports_dir") / "filtering_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[splits] {json.dumps(report['splits'])}")
    print(f"[done] report -> {report_path}")
    print(f"[next] hand-verify {gold_dir / 'gold_test.jsonl'} per {gold_dir / 'LABELING_GUIDE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
