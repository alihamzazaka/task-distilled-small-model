# Gold Pipeline — End-to-End Demo (SYNTHETIC raters)

> **⚠️ SYNTHETIC raters (LLM stand-ins) to validate the pipeline — NOT human
> gold. Replace with two human CSVs for the real κ ≥ 0.70 result.**

This report is produced by `scripts/gold_synthetic_raters.py`. It proves the
`sample → rate → adjudicate → score` loop runs end to end by standing in for the
two future human raters with the local model under **two different rubric
phrasings and temperatures**, so the measured κ reflects genuine rubric
sensitivity rather than a copy of one rater onto another.

## Backend
- **Raters:** Ollama qwen3:14b (think:false)
- **Rater A rubric:** meticulous auditor (temperature 0.0)
- **Rater B rubric:** pragmatic reviewer (temperature 0.6)
- **Adjudicator:** neutral field-rules pass on disagreements only
- **Items rated:** 16

## Measured inter-rater agreement (synthetic)
| metric | value |
|---|---|
| Cohen's κ (A vs B) | **0.6364** |
| raw agreement | 81.2% |
| shared rated items | 16 |
| disagreements | 3 |
| confusion (a\|b) | `correct|correct`=6, `incorrect|correct`=3, `incorrect|incorrect`=7 |

κ here is a *pipeline-validation* number. With two humans and this rubric the
target is **κ ≥ 0.70**; if the humans land below that, the rubric/examples in
`data/gold/rating_instructions.md` need tightening before the gold can anchor
the eval.

## Model vs synthetic-gold (shape of the defensible number)
| metric | value |
|---|---|
| adjudicated items | 16 |
| human-verified accuracy | **56.2%** (9/16 `correct`) |

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
python scripts/gold_kappa.py \
    --rater-a data/gold/rater_a.csv --rater-b data/gold/rater_b.csv \
    --adjudicated data/gold/adjudicated.csv
```
