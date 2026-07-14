#!/usr/bin/env python
"""Phase 3 — evaluate the student and produce the "money table".

Measures, per SPEC §7:
- **Task metric M** (field F1 + exact match + schema-valid rate) on the
  human-verified gold test set (records with human_verified:true).
- **Teacher-agreement %** on the fresh unseen agreement pool.
- **Latency** p50/p95 from local generation timing.
- **Cost** $/1k for teacher (price table x measured token profile) vs
  student (amortized GPU + electricity x measured throughput).

Writes reports/eval_report.json (consumed by the dashboard) and
reports/money_table.md, plus per-item predictions for failure analysis.

Usage (GPU box):
    python scripts/evaluate.py [--model models/student] [--limit N]
    python scripts/evaluate.py --teacher-baseline   # also score the teacher's
                                                    # cached outputs vs gold
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from distil_task.config import ensure_parent, load_config, resolve_path
from distil_task.cost_model import (
    break_even_requests_per_day,
    cost_multiple,
    params_from_config,
    student_cost_per_1k,
    teacher_cost_per_1k,
)
from distil_task.io_utils import read_jsonl, write_jsonl
from distil_task.metrics import evaluate_batch, percentile, teacher_agreement
from distil_task.schema import try_validate_invoice
from serve_path_shim import ensure_serve_on_path  # noqa: E402

ensure_serve_on_path()
from infer import StructuredExtractor, TransformersBackend  # noqa: E402


def params_billions(model_name: str) -> float | None:
    """Best-effort parse of a parameter count, in billions, from a model id.

    E.g. 'qwen3:14b' -> 14.0, 'Qwen/Qwen2.5-0.5B-Instruct' -> 0.5. Returns None
    when the name carries no size tag, so callers can fall back to a neutral label.
    """
    hits = re.findall(r"(\d+(?:\.\d+)?)[bB](?![A-Za-z0-9])", model_name)
    return float(hits[-1]) if hits else None


def run_model(extractor: StructuredExtractor, records: list[dict], label: str):
    """Run the student on records' inputs. Returns (preds, raws, latencies_s)."""
    preds, raws, lats = [], [], []
    for i, rec in enumerate(records, 1):
        t0 = time.perf_counter()
        result = extractor.extract_safe(rec["input"])
        lats.append(time.perf_counter() - t0)
        raws.append(result.raw_text)
        preds.append(result.data)  # None when invalid after retries
        if i % 20 == 0 or i == len(records):
            print(f"  [{label}] {i}/{len(records)}  p50 so far={percentile(lats, 50)*1000:.0f}ms")
    return preds, raws, lats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="student checkpoint dir (default paths.student_dir)")
    ap.add_argument("--limit", type=int, default=None, help="cap items per set (debug)")
    ap.add_argument("--teacher-baseline", action="store_true",
                    help="also score cached teacher outputs against gold")
    ap.add_argument("--allow-unverified-gold", action="store_true",
                    help="score ALL gold records even if human_verified is false "
                         "(dev only — the headline number requires verification)")
    ap.add_argument("--silver-only", action="store_true",
                    help="score only SILVER-verified gold (cross-model agreement, see "
                         "scripts/03_silver_verify.py) — a mid grade between dev and human-gold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    money_tol = float(cfg["filtering"]["money_abs_tol"])
    model_dir = Path(args.model) if args.model else resolve_path(cfg, "student_dir")
    reports_dir = resolve_path(cfg, "reports_dir")

    # ---- load eval sets -----------------------------------------------------
    gold_all = read_jsonl(resolve_path(cfg, "gold_dir") / "gold_test.jsonl")
    if args.silver_only:
        gold = [r for r in gold_all if r.get("silver_verified")]
        if not gold:
            print("[error] no silver-verified gold — run scripts/03_silver_verify.py first "
                  "(or drop --silver-only).")
            return 1
        gold_grade = "SILVER (cross-model agreement)"
        print(f"[silver] scoring {len(gold)}/{len(gold_all)} SILVER-verified gold records "
              f"(model: {gold[0].get('silver_model', '?')})")
    else:
        gold = [r for r in gold_all if r.get("human_verified") or args.allow_unverified_gold]
        gold_grade = "human_verified" if all(r.get("human_verified") for r in gold_all) else "DEV_ONLY_teacher_labeled"
    unverified = len(gold_all) - sum(1 for r in gold_all if r.get("human_verified"))
    if not gold:
        print("[error] no usable gold records — hand-verify data/gold/gold_test.jsonl first "
              "(or pass --allow-unverified-gold for a dev run)")
        return 1
    if unverified and args.allow_unverified_gold and not args.silver_only:
        print(f"[WARN] scoring {unverified} UNVERIFIED gold records — not a headline number")

    pool = read_jsonl(resolve_path(cfg, "splits_dir") / "agreement_pool.jsonl")
    if args.limit:
        gold, pool = gold[: args.limit], pool[: args.limit]
    print(f"[eval] gold={len(gold)}  agreement_pool={len(pool)}  model={model_dir}")

    # ---- student ------------------------------------------------------------
    backend = TransformersBackend(
        model_dir=model_dir,
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
    )
    extractor = StructuredExtractor(backend, max_retries=int(cfg["inference"]["max_retries"]))

    gold_preds, gold_raws, gold_lats = run_model(extractor, gold, "gold")
    gold_targets = [r["output"] for r in gold]
    quality = evaluate_batch(gold_preds, gold_targets, raw_outputs=gold_raws, money_abs_tol=money_tol)

    # ---- teacher-agreement on the fresh pool ---------------------------------
    agreement = None
    if pool:
        pool_preds, _, pool_lats = run_model(extractor, pool, "pool")
        teacher_outputs = [r["output"] for r in pool]
        agreement = teacher_agreement(pool_preds, teacher_outputs, money_abs_tol=money_tol)
        all_lats = gold_lats + pool_lats
    else:
        all_lats = gold_lats

    # ---- latency & throughput -------------------------------------------------
    p50 = percentile(all_lats, 50)
    p95 = percentile(all_lats, 95)
    throughput_rps = 1.0 / max(sum(all_lats) / len(all_lats), 1e-9)

    # ---- cost -------------------------------------------------------------------
    pricing, params, avg_in, avg_out = params_from_config(cfg)
    # overwrite configured throughput with the measured one
    from dataclasses import replace  # noqa: PLC0415
    params = replace(params, throughput_rps=throughput_rps)
    t_1k = teacher_cost_per_1k(pricing, avg_in, avg_out)
    s_1k = student_cost_per_1k(params)
    multiple = cost_multiple(pricing, avg_in, avg_out, params)
    breakeven = break_even_requests_per_day(pricing, avg_in, avg_out, params)

    # ---- teacher baseline (optional): cached teacher outputs vs verified gold ---
    teacher_quality = None
    if args.teacher_baseline:
        t_preds = [try_validate_invoice(r["output"])[0] for r in gold]
        t_dicts = [p.model_dump(mode="json") if p else None for p in t_preds]
        teacher_quality = evaluate_batch(t_dicts, gold_targets, money_abs_tol=money_tol)
        # NOTE: if gold outputs were seeded from teacher proposals and merely
        # verified, teacher scores near 1.0 by construction on unchanged items.

    # ---- report -------------------------------------------------------------------
    metric_name = str(cfg["task"]["metric"])
    bar = float(cfg["task"]["quality_bar_ratio"])
    teacher_m = (teacher_quality or {}).get(metric_name, 1.0)
    student_m = quality[metric_name]
    ratio = student_m / teacher_m if teacher_m else 0.0

    report = {
        "model_dir": str(model_dir),
        "n_gold": len(gold),
        # Honest provenance: never erase the unverified count. When
        # --allow-unverified-gold is used the unverified records are SCORED,
        # not excluded — record both facts so a dev-grade run can't be
        # mistaken for a human-verified headline run.
        "n_unverified_excluded": 0 if args.allow_unverified_gold else unverified,
        "n_unverified_scored": unverified if args.allow_unverified_gold else 0,
        "allow_unverified_gold": bool(args.allow_unverified_gold),
        "gold_grade": gold_grade,
        "quality": quality,
        "teacher_quality": teacher_quality,
        "metric": metric_name,
        "quality_bar_ratio": bar,
        "student_over_teacher_ratio": round(ratio, 4),
        "meets_bar": bool(ratio >= bar),
        "teacher_agreement": agreement,
        "latency_s": {"p50": round(p50, 4), "p95": round(p95, 4)},
        "throughput_rps": round(throughput_rps, 3),
        "cost": {
            "teacher_usd_per_1k": round(t_1k, 4),
            "student_usd_per_1k": round(s_1k, 4),
            "cost_multiple": round(multiple, 1),
            "break_even_requests_per_day": round(breakeven, 1) if breakeven else None,
            "assumptions": {
                "teacher_avg_input_tokens": avg_in,
                "teacher_avg_output_tokens": avg_out,
                "gpu_price_usd": params.gpu_price_usd,
                "gpu_power_watts": params.gpu_power_watts,
                "electricity_usd_per_kwh": params.electricity_usd_per_kwh,
                "measured_throughput_rps": round(throughput_rps, 3),
            },
        },
    }
    report_path = ensure_parent(reports_dir / "eval_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # per-item dump for failure analysis
    write_jsonl(
        reports_dir / "gold_predictions.jsonl",
        (
            {"id": r["id"], "input": r["input"], "gold": r["output"],
             "pred": p, "raw": raw, "latency_s": round(lat, 4)}
            for r, p, raw, lat in zip(gold, gold_preds, gold_raws, gold_lats)
        ),
    )

    # ---- money table ------------------------------------------------------------
    # A local open-model teacher (priced $0) has no cost-vs-API arbitrage, so the
    # classic "Nx cheaper than a frontier API" table is misleading. Branch on it
    # and frame the local case honestly around size / latency / privacy.
    teacher_model = cfg['teacher']['model']
    student_model = cfg['training']['base_model']
    teacher_local = (t_1k <= 0.0)
    if args.silver_only:
        gold_caveat = (
            f"\n> **Silver-grade numbers:** scored on the {len(gold)} gold items where "
            f"an INDEPENDENT second model ({gold[0].get('silver_model', '?')}) agreed with the "
            f"teacher's label (cross-model agreement removes the teacher-vs-itself circularity). "
            f"This is stronger than dev-grade but not yet human-verified.\n"
        )
    elif unverified == 0:
        gold_caveat = ""
    else:
        gold_caveat = (
            f"\n> **⚠ Dev-grade numbers:** all {unverified} gold records are teacher-labeled "
            f"(`human_verified: false`) — the quality row is student-vs-teacher agreement, "
            f"not human-verified ground truth. The teacher reference of "
            f"{'{:.4f}'.format(teacher_m)} is 1.0 **by construction** on unverified gold. "
            f"Hand-verify `data/gold/gold_test.jsonl` (or use --silver-only) before quoting "
            f"these as headline numbers.\n"
        )
    # Size labels are derived from the configured model ids (not hardcoded) so a
    # teacher/student swap in configs/default.yaml keeps the table honest.
    teacher_b = params_billions(teacher_model)
    student_b = params_billions(student_model)
    teacher_class = f"{teacher_b:g}B-class" if teacher_b else "open teacher"
    student_size = f" (~{student_b:g}B)" if student_b else ""
    footprint_win = (
        f"**~{teacher_b / student_b:.0f}× smaller**"
        if teacher_b and student_b else "smaller"
    )
    teacher_forward = f"full {teacher_b:g}B forward" if teacher_b else "full teacher forward"
    student_note = f"a ~{student_b:g}B student" if student_b else "the student"
    max_retries = int(cfg["inference"]["max_retries"])
    if teacher_local:
        money = f"""# Money table — invoice extraction (local open teacher → distilled student)

_Teacher `{teacher_model}` and student `{student_model}` both run locally on the RTX 5080; gold set, {metric_name}._
{gold_caveat}

| Axis | Teacher ({teacher_model}) | Student (distilled) | Win |
|---|---|---|---|
| Model footprint | {teacher_class} | {student_model.split('/')[-1]}{student_size} | {footprint_win} |
| Quality ({metric_name}) | {teacher_m:.4f} (ref) | {student_m:.4f} ({ratio:.1%} of teacher) | {"meets" if ratio >= bar else "below"} {bar:.0%} bar |
| Schema-valid rate | — | {quality['schema_valid_rate']:.1%} (after ≤{max_retries} constrained-repair retries) | robustness |
| Exact match | — | {quality['exact_match']:.1%} | strictest view |
| p95 latency | {teacher_forward} | {p95*1000:,.0f} ms | smaller footprint |
| Data egress | stays local | stays local | on-prem / private |
| $/1k (GPU amortized) | ${t_1k:,.4f} | ${s_1k:,.4f} | both ~free locally |

**Honest note on cost:** with a *free local* teacher, the "1/40th the cost of a frontier API" pitch does **not** apply — both models run on your own GPU, so the student is not cheaper than the teacher in dollars. The real value here is **footprint** ({student_note} packs alongside other workloads and serves at {throughput_rps:.2f} req/s) and **privacy** (nothing leaves the box). To make the dollar-cost case, distill from a **paid** frontier teacher: then student ${s_1k:,.4f}/1k vs the API list price is the headline (set `teacher.provider` to a paid client in configs/default.yaml).

*Assumptions: GPU ${params.gpu_price_usd:,.0f} amortized over {params.gpu_lifetime_years:g} years, {params.gpu_power_watts:g} W at ${params.electricity_usd_per_kwh}/kWh, measured throughput {throughput_rps:.2f} req/s.*
"""
    else:
        teacher_lat = "~2.1 s (typ. API)"  # measured only if you also time API calls
        money = f"""# Money Table — invoice extraction (teacher vs distilled student)

| Metric | Teacher (API: {teacher_model}) | Student (local, RTX 5080) | Win |
|---|---|---|---|
| $/1k requests | ${t_1k:,.2f} | ${s_1k:,.4f} | **{multiple:,.0f}x cheaper** |
| p95 latency | {teacher_lat} | {p95*1000:,.0f} ms | lower |
| Data egress | leaves org | stays local | privacy |
| Quality ({metric_name}, gold set) | {teacher_m:.4f} (ref) | {student_m:.4f} ({ratio:.1%} of teacher) | {"meets" if ratio >= bar else "BELOW"} {bar:.0%} bar |
| Teacher-agreement (fresh pool) | — | {f"{agreement:.1%}" if agreement is not None else "n/a"} | generalization |
| Schema-valid rate | — | {quality['schema_valid_rate']:.1%} | robustness |
| Exact match | — | {quality['exact_match']:.1%} | strictest view |

**Break-even volume:** {"~" + format(breakeven, ",.0f") + " requests/day" if breakeven else "none (teacher marginal cost below student electricity)"} — above this, the local student is cheaper than the API.

*Cost assumptions: GPU ${params.gpu_price_usd:,.0f} amortized over {params.gpu_lifetime_years:g} years, {params.gpu_power_watts:g} W at ${params.electricity_usd_per_kwh}/kWh, measured throughput {throughput_rps:.2f} req/s; teacher token profile {avg_in:.0f} in / {avg_out:.0f} out per request.*
"""
    money_path = reports_dir / "money_table.md"
    money_path.write_text(money, encoding="utf-8")

    print(json.dumps(report["quality"], indent=2))
    print(f"[result] {metric_name}={student_m:.4f}  ratio={ratio:.1%}  meets_bar={report['meets_bar']}")
    print(f"[done] {report_path}\n[done] {money_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
