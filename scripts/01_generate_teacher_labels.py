#!/usr/bin/env python
"""Phase 1, step 1 — teacher labels every seed input.

For each seed document the teacher produces the structured JSON under the
FIXED extraction prompt. Outputs are parsed and schema-validated on the
spot; invalid ones are kept in the file with valid=false so the filtering
step can log drop reasons.

Optionally (--second-pass, default on when filtering.consistency_enabled)
runs the teacher a second time at a higher temperature for the
self-consistency check consumed by scripts/02_filter_and_split.py.

Outputs:
    data/labeled/labeled.jsonl              {"id","input","output","valid","error"}
    data/labeled/labeled_second_pass.jsonl  (if second pass enabled)
    reports/cost_teacher_labeling.json      cost report

Runs on the laptop with requirements-laptop.txt (API-only, cached, resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from distil_task.config import ensure_parent, load_config, resolve_path
from distil_task.io_utils import extract_first_json, read_jsonl, write_jsonl
from distil_task.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from distil_task.schema import try_validate_invoice
from distil_task.teacher import get_teacher


def label_one(teacher, text: str, temperature: float, max_tokens: int, salt: str) -> dict:
    resp = teacher.complete(
        build_extraction_user_prompt(text),
        system=EXTRACTION_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_salt=salt,
    )
    rec: dict = {"raw": resp.text}
    try:
        obj = extract_first_json(resp.text)
    except ValueError as e:
        rec.update(output=None, valid=False, error=f"parse: {e}")
        return rec
    inv, err = try_validate_invoice(obj)
    if inv is None:
        rec.update(output=obj, valid=False, error=f"schema: {err}")
    else:
        rec.update(output=inv.model_dump(mode="json"), valid=True, error=None)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="label only the first N seeds")
    ap.add_argument("--second-pass", dest="second_pass", action="store_true", default=None,
                    help="force the consistency second pass")
    ap.add_argument("--no-second-pass", dest="second_pass", action="store_false")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["teacher"]
    max_tokens = int(tcfg["max_tokens"])
    t_label = float(tcfg["temperature"])
    t_consist = float(tcfg["consistency_temperature"])
    second_pass = (
        bool(cfg["filtering"]["consistency_enabled"])
        if args.second_pass is None
        else args.second_pass
    )

    seeds = read_jsonl(resolve_path(cfg, "seeds_file"))
    if args.limit:
        seeds = seeds[: args.limit]
    print(f"[labeling] {len(seeds)} seeds, second_pass={second_pass}")

    teacher = get_teacher(cfg, resolve_path(cfg, "cache_dir"))

    labeled, second = [], []
    n_valid = 0
    for i, seed in enumerate(seeds, 1):
        rec = label_one(teacher, seed["text"], t_label, max_tokens, salt="label-v1")
        row = {
            "id": seed["id"],
            "input": seed["text"],
            "source": seed.get("source", "unknown"),
            "output": rec["output"],
            "valid": rec["valid"],
            "error": rec["error"],
        }
        labeled.append(row)
        n_valid += int(rec["valid"])

        if second_pass:
            rec2 = label_one(teacher, seed["text"], t_consist, max_tokens, salt="consist-v1")
            second.append({"id": seed["id"], "output": rec2["output"], "valid": rec2["valid"]})

        if i % 25 == 0 or i == len(seeds):
            print(
                f"  {i}/{len(seeds)}  valid={n_valid}  "
                f"spend=${teacher.cost.usd:.2f}  cache_hits={teacher.cost.cache_hits}"
            )

    labeled_path = resolve_path(cfg, "labeled_file")
    write_jsonl(labeled_path, labeled)
    print(f"[done] {n_valid}/{len(labeled)} valid -> {labeled_path}")

    if second_pass:
        second_path = resolve_path(cfg, "consistency_file")
        write_jsonl(second_path, second)
        print(f"[done] second pass -> {second_path}")

    report = teacher.cost.report()
    report["n_seeds"] = len(seeds)
    report["n_valid"] = n_valid
    report["schema_valid_rate_teacher"] = round(n_valid / max(len(labeled), 1), 4)
    report_path = ensure_parent(resolve_path(cfg, "reports_dir") / "cost_teacher_labeling.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[cost] {json.dumps(report, indent=2)}")
    print(f"[cost] report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
