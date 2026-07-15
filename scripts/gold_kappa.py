#!/usr/bin/env python
"""Score two filled rater CSVs → Cohen's κ, disagreements, model-vs-human.

Stage 1 (always): ingest two independently-filled rater sheets, compute Cohen's
κ and raw percent agreement over the shared, fully-rated items, and list the
disagreements that need adjudication (also written to a CSV you can hand to a
third rater / consensus discussion).

Stage 2 (once an adjudicated gold column exists): report the **model's agreement
with the human gold** — the real, defensible number. For this project the
student emits an extraction it deems correct for every item, so the headline is
the human-verified extraction accuracy (fraction of adjudicated `correct`).

    # inter-rater κ + adjudication queue
    python scripts/gold_kappa.py \
        --rater-a data/gold/rater_a.csv --rater-b data/gold/rater_b.csv

    # after adjudication: add the gold to get model-vs-human
    python scripts/gold_kappa.py \
        --rater-a data/gold/rater_a.csv --rater-b data/gold/rater_b.csv \
        --adjudicated data/gold/adjudicated.csv

Pure stdlib. No network, no GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from distil_task.gold_pipeline import (             # noqa: E402
    human_verified_accuracy,
    normalize_label,
    read_rater_csv,
    score_raters,
)

DEFAULT_DISAGREEMENTS = _ROOT / "data" / "gold" / "disagreements.csv"


def _read_adjudicated(path: Path) -> dict[str, str]:
    """Read an adjudicated gold CSV (columns: id,label) → {id: label}."""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rid = (row.get("id") or "").strip()
            lab = normalize_label(row.get("label"))
            if rid and lab:
                out[rid] = lab
    return out


def _write_disagreements(path: Path, disagreements: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "rater_a", "rater_b", "adjudicated"])
        w.writeheader()
        for d in disagreements:
            w.writerow({"id": d["id"], "rater_a": d["rater_a"],
                        "rater_b": d["rater_b"], "adjudicated": ""})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rater-a", type=Path, required=True)
    ap.add_argument("--rater-b", type=Path, required=True)
    ap.add_argument("--adjudicated", type=Path, default=None,
                    help="CSV (id,label) of the final human gold after adjudication")
    ap.add_argument("--disagreements-out", type=Path, default=DEFAULT_DISAGREEMENTS)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows_a = read_rater_csv(args.rater_a)
    rows_b = read_rater_csv(args.rater_b)
    report = score_raters(rows_a, rows_b)

    kappa = report["cohen_kappa"]
    verdict = "PASS (kappa >= 0.70)" if kappa >= 0.70 else "REVISE (kappa < 0.70 - tighten rubric)"
    print("=" * 64)
    print("INTER-RATER AGREEMENT")
    print("=" * 64)
    print(f"  shared rated items : {report['n']}")
    print(f"  Cohen's kappa      : {kappa:.4f}   [{verdict}]")
    print(f"  raw agreement      : {report['raw_agreement']:.1%}")
    print(f"  disagreements      : {report['n_disagreements']}")
    print("  confusion (a|b)    : " + ", ".join(
        f"{k}={v}" for k, v in report["confusion"].items() if v))

    if report["disagreements"]:
        _write_disagreements(args.disagreements_out, report["disagreements"])
        print(f"\n  -> adjudication queue written to {args.disagreements_out}")
        print("    (fill the 'adjudicated' column, then rebuild the gold CSV)")

    out: dict = {"inter_rater": {k: report[k] for k in
                                 ("n", "cohen_kappa", "raw_agreement",
                                  "n_disagreements", "confusion")}}

    if args.adjudicated is not None:
        if not args.adjudicated.exists():
            raise SystemExit(f"adjudicated file not found: {args.adjudicated}")
        gold = _read_adjudicated(args.adjudicated)
        if not gold:
            raise SystemExit("adjudicated file has no usable (id,label) rows")
        gold_labels = [gold[i] for i in sorted(gold)]
        mvh = human_verified_accuracy(gold_labels)
        out["model_vs_human"] = mvh
        print("\n" + "=" * 64)
        print("MODEL vs HUMAN GOLD  (the defensible number)")
        print("=" * 64)
        print(f"  adjudicated gold items : {int(mvh['n'])}")
        print(f"  human-verified accuracy: {mvh['human_verified_accuracy']:.1%}"
              f"  ({int(mvh['n_correct'])}/{int(mvh['n'])} correct)")
        print("  (student asserts an extraction for every item -> precision on the")
        print("   'correct' class == accuracy, recall == 1.0)")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n  -> machine-readable report: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
