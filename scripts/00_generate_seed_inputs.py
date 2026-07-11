#!/usr/bin/env python
"""Phase 1, step 0 — build the seed-input pool.

1. Ingest any real samples dropped into data/raw/ (*.txt, one document per
   file; best signal when available).
2. Ask the teacher to synthesize N diverse invoice/receipt texts (batched,
   temperature-high, domain-steered round-robin).

Output: data/seeds/seeds.jsonl  — {"id", "text", "source"} per line.
Runs on the laptop with requirements-laptop.txt (API-only, no GPU).

Usage:
    python scripts/00_generate_seed_inputs.py [--n 3000] [--config configs/default.yaml] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from distil_task.config import load_config, resolve_path
from distil_task.io_utils import extract_first_json, write_jsonl
from distil_task.prompts import (
    SEED_GENERATION_SYSTEM_PROMPT,
    build_seed_generation_user_prompt,
)
from distil_task.teacher import get_teacher

# Round-robin steering hints so batches cover the diversity axes.
DOMAIN_HINTS = [
    None,
    "retail receipts (grocery, pharmacy, electronics) in the USA and Canada",
    "restaurant and café bills in France, Italy and Japan",
    "B2B service invoices (consulting, freelance dev, marketing) with hourly line items",
    "hardware/construction-supply invoices in Pakistan, India and the UAE",
    "utility and telecom bills in Germany and Poland",
    "hotel folios and travel invoices in Brazil and Mexico",
    "medical/dental bills and lab invoices in the UK",
    "e-commerce order confirmations with discounts and shipping lines",
    "auto-repair and equipment-maintenance work orders with parts + labor",
]


def seed_id(text: str) -> str:
    return "seed-" + hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def ingest_raw(raw_dir: Path) -> list[dict]:
    records = []
    if raw_dir.exists():
        for p in sorted(raw_dir.glob("*.txt")):
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                records.append({"id": seed_id(text), "text": text, "source": f"raw:{p.name}"})
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="config YAML (default configs/default.yaml)")
    ap.add_argument("--n", type=int, default=None, help="override generation.n_seeds")
    ap.add_argument("--dry-run", action="store_true", help="ingest raw only; skip teacher calls")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gcfg = cfg["generation"]
    n_target = args.n if args.n is not None else int(gcfg["n_seeds"])
    per_call = int(gcfg["seeds_per_call"])
    temperature = float(gcfg["seed_temperature"])

    seeds_path = resolve_path(cfg, "seeds_file")
    raw_dir = resolve_path(cfg, "raw_dir")
    cache_dir = resolve_path(cfg, "cache_dir")

    records = ingest_raw(raw_dir)
    print(f"[raw] ingested {len(records)} real samples from {raw_dir}")

    seen_ids = {r["id"] for r in records}

    if not args.dry_run and len(records) < n_target:
        teacher = get_teacher(cfg, cache_dir)
        batch_index = 0
        misses = 0
        while len(records) < n_target and misses < 50:
            want = min(per_call, n_target - len(records))
            hint = DOMAIN_HINTS[batch_index % len(DOMAIN_HINTS)]
            prompt = build_seed_generation_user_prompt(want, batch_index, hint)
            resp = teacher.complete(
                prompt,
                system=SEED_GENERATION_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=int(cfg["teacher"]["max_tokens"]) * 2,
                cache_salt=f"seedbatch-{batch_index}",
            )
            batch_index += 1
            try:
                docs = extract_first_json(resp.text)
            except ValueError as e:
                print(f"[warn] batch {batch_index}: unparseable response ({e}); skipping")
                misses += 1
                continue
            if not isinstance(docs, list):
                misses += 1
                continue
            added = 0
            for doc in docs:
                if not isinstance(doc, str) or len(doc.strip()) < 40:
                    continue
                sid = seed_id(doc)
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                records.append({"id": sid, "text": doc.strip(), "source": "teacher"})
                added += 1
            if added == 0:
                misses += 1
            print(
                f"[gen] batch {batch_index} (+{added}) -> {len(records)}/{n_target} "
                f"(spend so far: ${teacher.cost.usd:.2f})"
            )
        print(f"[cost] seed generation: {teacher.cost.report()}")

    n = write_jsonl(seeds_path, records)
    print(f"[done] wrote {n} seeds -> {seeds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
