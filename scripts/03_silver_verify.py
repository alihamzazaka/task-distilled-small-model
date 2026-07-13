#!/usr/bin/env python
"""Phase 2b — upgrade the gold set from DEV-GRADE to SILVER by cross-model
agreement, removing the circularity where the teacher's own labels are scored
against themselves.

An INDEPENDENT second model (default qwen2.5-coder:32b via Ollama — a different,
non-reasoning model from the qwen3:14b teacher) re-extracts each gold document
under the same extraction prompt. A gold item is marked ``silver_verified`` only
when the two independent models AGREE at the field level (field-F1 >= --agree,
default 0.95). The agreed subset is trustworthy without a human, and
``scripts/evaluate.py`` can then report the silver grade.

Usage:
    python scripts/03_silver_verify.py                       # relabel data/gold/gold_test.jsonl in place
    python scripts/03_silver_verify.py --model qwen3:32b --agree 0.9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from distil_task.config import ensure_dir, load_config, resolve_path
from distil_task.io_utils import extract_first_json, read_jsonl, write_jsonl
from distil_task.metrics import evaluate_batch
from distil_task.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from distil_task.schema import try_validate_invoice
from distil_task.teacher import LocalOpenAITeacher


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="qwen2.5-coder:32b",
                    help="independent verifier model (Ollama tag) — must differ from the teacher")
    ap.add_argument("--gold", default=None, help="gold jsonl (default paths.gold_dir/gold_test.jsonl)")
    ap.add_argument("--agree", type=float, default=0.95,
                    help="min field-F1 between the two models to count as agreement")
    ap.add_argument("--money-tol", type=float, default=0.01)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gold_path = Path(args.gold) if args.gold else resolve_path(cfg, "gold_dir") / "gold_test.jsonl"
    if not gold_path.exists():
        print(f"[silver] gold set not found: {gold_path}", file=sys.stderr)
        return 1
    rows = read_jsonl(gold_path)

    cache = ensure_dir(resolve_path(cfg, "cache_dir") / "silver")
    verifier = LocalOpenAITeacher(
        model=args.model, cache_dir=cache,
        price_table=cfg["teacher"].get("price_table", {}),
    )
    print(f"[silver] verifying {len(rows)} gold items with independent model "
          f"'{args.model}' (teacher was '{cfg['teacher']['model']}')")

    verified = 0
    invalid = 0
    for i, r in enumerate(rows):
        prompt = build_extraction_user_prompt(r["input"])
        try:
            resp = verifier.complete(prompt, system=EXTRACTION_SYSTEM_PROMPT,
                                     temperature=0.0, max_tokens=int(cfg["teacher"]["max_tokens"]))
            obj = extract_first_json(resp.text)
            inv, err = try_validate_invoice(obj)
        except Exception as e:  # noqa: BLE001 - verifier hiccup -> unverified
            inv, err = None, str(e)
        if inv is None:
            invalid += 1
            r["silver_verified"] = False
            r["silver_model"] = args.model
            r["silver_field_f1"] = 0.0
            r["silver_note"] = f"verifier-invalid: {str(err)[:60]}"
            continue
        # field-level agreement between the two independent extractions
        q = evaluate_batch([inv.model_dump(mode="json")], [r["output"]],
                           money_abs_tol=args.money_tol)
        f1 = float(q.get("field_f1", 0.0))
        ok = f1 >= args.agree
        verified += int(ok)
        r["silver_verified"] = ok
        r["silver_model"] = args.model
        r["silver_field_f1"] = round(f1, 4)
        if (i + 1) % 20 == 0:
            print(f"[silver]   {i + 1}/{len(rows)} done ({verified} agreed so far)")

    write_jsonl(gold_path, rows)
    rate = verified / max(1, len(rows))
    print(f"\n[silver] SILVER-VERIFIED {verified}/{len(rows)} ({rate:.1%}) — "
          f"{invalid} verifier-invalid. Grade: independent cross-model agreement.")
    print(f"[silver] wrote flags into {gold_path}")
    print("[silver] next: scripts/evaluate.py --silver-only for the silver-grade money table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
