#!/usr/bin/env python
"""SYNTHETIC-rater stand-in — exercise the whole κ loop *now*, without humans.

Generates two INDEPENDENT label sets over the rating sheet using the local
Ollama model (qwen3:14b, ``think:false``) with two DIFFERENT rubric phrasings
and temperatures, so their κ is meaningful (not a trivial 1.0). A third neutral
pass adjudicates disagreements to form a synthetic gold. We then run the same κ
scorer used for humans and write ``reports/gold_pipeline_demo.md``.

    python scripts/gold_synthetic_raters.py --limit 20

>>> These are SYNTHETIC raters (LLM stand-ins) to validate the pipeline —
>>> NOT human gold. Replace with two human CSVs for the real κ ≥ 0.70 result. <<<

If Ollama is unreachable, falls back to a deterministic rule-based stand-in
(two heuristics of differing strictness) and says so in the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from distil_task.gold_pipeline import (             # noqa: E402
    RATING_SHEET_FIELDS,
    human_verified_accuracy,
    normalize_label,
    score_raters,
    write_rating_sheet,
)

DEFAULT_SHEET = _ROOT / "data" / "gold" / "rating_sheet.csv"
OUT_A = _ROOT / "data" / "gold" / "synthetic_rater_a.csv"
OUT_B = _ROOT / "data" / "gold" / "synthetic_rater_b.csv"
OUT_GOLD = _ROOT / "data" / "gold" / "synthetic_adjudicated.csv"
DEFAULT_REPORT = _ROOT / "reports" / "gold_pipeline_demo.md"

# Two deliberately different rubric phrasings → non-degenerate κ.
RUBRIC_A = (
    "You are a METICULOUS invoice auditor. You are shown the raw source "
    "document and a JSON extraction of it. Mark the extraction 'incorrect' if "
    "ANY field deviates from the document at all: a truncated description, a "
    "wrong or missing number, a due-date used as the invoice date, an invented "
    "field. Only mark 'correct' when every field is faithful."
)
RUBRIC_B = (
    "You are a pragmatic reviewer checking an invoice extraction against its "
    "source text. Decide whether the JSON captures the invoice's key facts "
    "(vendor, date, currency, the line items, and the totals). Mark 'correct' "
    "if it faithfully represents the document and 'incorrect' if it gets a "
    "material fact wrong or drops one."
)
RUBRIC_ADJ = (
    "You are the adjudicator. Two reviewers disagreed on whether this JSON "
    "extraction faithfully represents the source invoice. Judge it yourself "
    "against the field rules (vendor, ISO date, ISO currency, line items in "
    "order, subtotal/tax/grand_total within rounding). Answer 'correct' or "
    "'incorrect'."
)

_PROMPT_TMPL = (
    "{rubric}\n\n"
    "SOURCE DOCUMENT:\n---\n{doc}\n---\n\n"
    "JSON EXTRACTION:\n---\n{ext}\n---\n\n"
    "Answer with exactly one word on the first line: 'correct' or 'incorrect'.\n"
    "Then optionally a short reason on the next line prefixed 'Reason:'.\n"
    "Label:"
)


# --------------------------------------------------------------------------- #
# Ollama (stdlib urllib, think:false) with graceful fallback
# --------------------------------------------------------------------------- #

def _ollama_base() -> str:
    return os.environ.get(
        "OLLAMA_BASE_URL",
        os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434").replace("/v1", ""),
    ).rstrip("/")


def ollama_available(model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{_ollama_base()}/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        names = {m.get("name", "") for m in tags.get("models", [])}
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    except Exception:
        return False


def ollama_label(model: str, rubric: str, doc: str, ext: str,
                 temperature: float) -> str:
    prompt = _PROMPT_TMPL.format(rubric=rubric, doc=doc[:6000], ext=ext[:4000])
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 64},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{_ollama_base()}/api/chat", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 (local)
        payload = json.loads(resp.read().decode("utf-8"))
    text = (payload.get("message") or {}).get("content", "") or ""
    return _parse_label(text)


def _parse_label(text: str) -> str:
    low = text.strip().casefold()
    # Prefer a "label:" line if present.
    for line in low.splitlines():
        line = line.strip().lstrip("*-# ").replace("label:", "").strip()
        norm = normalize_label(line.split()[0]) if line.split() else None
        if norm:
            return norm
    if "incorrect" in low:
        return "incorrect"
    if "correct" in low:
        return "correct"
    return "incorrect"  # conservative default when unparaseable


# --------------------------------------------------------------------------- #
# Deterministic fallback (no network): two heuristics of differing strictness
# --------------------------------------------------------------------------- #

def _numbers_in(doc: str, ext: str) -> tuple[int, int]:
    """How many of the extraction's monetary tokens appear in the document."""
    try:
        obj = json.loads(ext) if ext else {}
    except json.JSONDecodeError:
        return 0, 1
    doc_l = doc.casefold()
    tokens: list[str] = []
    for key in ("subtotal", "tax", "grand_total"):
        if obj.get(key) is not None:
            tokens.append(f"{float(obj[key]):.2f}")
    for li in obj.get("line_items") or []:
        if isinstance(li, dict) and li.get("total") is not None:
            tokens.append(f"{float(li['total']):.2f}")
    if not tokens:
        return 0, 1
    hits = sum(1 for t in tokens if t in doc_l or t.rstrip("0").rstrip(".") in doc_l)
    return hits, len(tokens)


def rule_label(doc: str, ext: str, strict: bool) -> str:
    hits, total = _numbers_in(doc, ext)
    frac = hits / total if total else 0.0
    threshold = 1.0 if strict else 0.6
    return "correct" if frac >= threshold else "incorrect"


# --------------------------------------------------------------------------- #

def _read_sheet(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _label_rows(rows, use_ollama, model):
    a_rows, b_rows, adj = [], [], {}
    for i, row in enumerate(rows, 1):
        rid = row["id"]
        doc = row.get("source_document", "")
        ext = row.get("model_extraction", "")
        if use_ollama:
            la = ollama_label(model, RUBRIC_A, doc, ext, temperature=0.0)
            lb = ollama_label(model, RUBRIC_B, doc, ext, temperature=0.6)
        else:
            la = rule_label(doc, ext, strict=True)
            lb = rule_label(doc, ext, strict=False)
        if la == lb:
            gold = la
        elif use_ollama:
            gold = ollama_label(model, RUBRIC_ADJ, doc, ext, temperature=0.0)
        else:
            gold = rule_label(doc, ext, strict=False)  # lenient tie-break
        a_rows.append({**row, "label": la})
        b_rows.append({**row, "label": lb})
        adj[rid] = gold
        print(f"  [{i}/{len(rows)}] {rid}: a={la} b={lb} gold={gold}")
    return a_rows, b_rows, adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--force-fallback", action="store_true",
                    help="skip Ollama and use the deterministic rule-based stand-in")
    args = ap.parse_args()

    if not args.sheet.exists():
        raise SystemExit(f"rating sheet not found: {args.sheet}\n"
                         "Run scripts/gold_sample.py first.")
    rows = _read_sheet(args.sheet)
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("rating sheet is empty")

    use_ollama = (not args.force_fallback) and ollama_available(args.model)
    backend = f"Ollama {args.model} (think:false)" if use_ollama \
        else "deterministic rule-based stand-in (Ollama unreachable)"
    print(f"[synthetic] backend: {backend}")
    print(f"[synthetic] labelling {len(rows)} items with two rubrics...")

    a_rows, b_rows, adj = _label_rows(rows, use_ollama, args.model)

    write_rating_sheet(OUT_A, a_rows, RATING_SHEET_FIELDS)
    write_rating_sheet(OUT_B, b_rows, RATING_SHEET_FIELDS)
    OUT_GOLD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_GOLD.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label"])
        w.writeheader()
        for rid, lab in adj.items():
            w.writerow({"id": rid, "label": lab})

    rep = score_raters(
        [{"id": r["id"], "label": normalize_label(r["label"])} for r in a_rows],
        [{"id": r["id"], "label": normalize_label(r["label"])} for r in b_rows],
    )
    gold_labels = [adj[i] for i in sorted(adj)]
    mvh = human_verified_accuracy(gold_labels)

    _write_report(args.report, backend, use_ollama, rep, mvh, len(rows))
    print(f"\n[synthetic] measured kappa (rater A vs B): {rep['cohen_kappa']:.4f}  "
          f"(raw agreement {rep['raw_agreement']:.1%}, n={rep['n']})")
    print(f"[synthetic] model vs synthetic-gold accuracy: "
          f"{mvh['human_verified_accuracy']:.1%}")
    print(f"[synthetic] wrote {args.report}")
    return 0


def _write_report(path: Path, backend: str, use_ollama: bool, rep: dict,
                  mvh: dict, n_items: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conf = ", ".join(f"`{k}`={v}" for k, v in rep["confusion"].items() if v)
    md = f"""\
# Gold Pipeline — End-to-End Demo (SYNTHETIC raters)

> **⚠️ SYNTHETIC raters (LLM stand-ins) to validate the pipeline — NOT human
> gold. Replace with two human CSVs for the real κ ≥ 0.70 result.**

This report is produced by `scripts/gold_synthetic_raters.py`. It proves the
`sample → rate → adjudicate → score` loop runs end to end by standing in for the
two future human raters with the local model under **two different rubric
phrasings and temperatures**, so the measured κ reflects genuine rubric
sensitivity rather than a copy of one rater onto another.

## Backend
- **Raters:** {backend}
- **Rater A rubric:** meticulous auditor (temperature 0.0)
- **Rater B rubric:** pragmatic reviewer (temperature 0.6)
- **Adjudicator:** neutral field-rules pass on disagreements only{"" if use_ollama else " (lenient heuristic in fallback mode)"}
- **Items rated:** {n_items}

## Measured inter-rater agreement (synthetic)
| metric | value |
|---|---|
| Cohen's κ (A vs B) | **{rep['cohen_kappa']:.4f}** |
| raw agreement | {rep['raw_agreement']:.1%} |
| shared rated items | {rep['n']} |
| disagreements | {rep['n_disagreements']} |
| confusion (a\\|b) | {conf} |

κ here is a *pipeline-validation* number. With two humans and this rubric the
target is **κ ≥ 0.70**; if the humans land below that, the rubric/examples in
`data/gold/rating_instructions.md` need tightening before the gold can anchor
the eval.

## Model vs synthetic-gold (shape of the defensible number)
| metric | value |
|---|---|
| adjudicated items | {int(mvh['n'])} |
| human-verified accuracy | **{mvh['human_verified_accuracy']:.1%}** ({int(mvh['n_correct'])}/{int(mvh['n'])} `correct`) |

For invoice extraction the student asserts an extraction for every item, so the
model-vs-human number is the **fraction the raters adjudicated `correct`** — the
human-verified extraction accuracy. Swap the synthetic CSVs for two human sheets
and re-run `scripts/gold_kappa.py` to publish the real figure.

> **Read this slice as a conservative floor, not the headline accuracy.** The
> active-learning sampler deliberately over-weights the hardest pairs (low
> field-F1, schema-repair triggered, exact-match misses), so the student looks
> *worse* here than on a representative slice. For the real headline, rate the
> full stratified 150–200-pair gold and report on it.

## Reproduce
```bash
python scripts/gold_sample.py --n 24            # build the rating sheet
python scripts/gold_synthetic_raters.py         # this demo (Ollama or fallback)
# real path: two humans fill data/gold/rater_a.csv + rater_b.csv, then:
python scripts/gold_kappa.py \\
    --rater-a data/gold/rater_a.csv --rater-b data/gold/rater_b.csv \\
    --adjudicated data/gold/adjudicated.csv
```
"""
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
