# 05 — Evaluation & Metrics (v2.0)

> How each v2.0 feature's success is measured, the new metrics it introduces, concrete **target** numbers versus the measured v1.0 baseline, and the money table v2.0 finally fills with a real dollar win. Metric definitions from [../05-evaluation-metrics.md](../05-evaluation-metrics.md) (field-F1, exact-match, schema-valid rate, teacher-agreement, cost/latency) carry over unchanged; only the *additions* are here.

## The measured v1.0 baseline (what every target is relative to)

From [`reports/eval_report.json`](../../reports/eval_report.json) (silver, n = 37) and [`reports/money_table.md`](../../reports/money_table.md):

| Metric | v1.0 measured | Grade |
|---|---|---|
| field-F1 (M) | **0.9647** (96.5% of teacher) | silver |
| micro field-F1 | 0.9676 | silver |
| precision / recall | 0.9619 / 0.9685 | silver |
| **exact-match** | **0.5405 (54.1%)** | silver |
| schema-valid rate | **100%** | silver |
| teacher-agreement | 93.5% | on 30-item fresh pool |
| latency p50 / p95 | 4.04 s / **5.92 s** | 0.5B on RTX 5080 |
| throughput | 0.226 req/s | measured |
| teacher $/1k | **$0.00** (free local) | — |
| student $/1k | **$0.1178** (amortized) | — |
| cost multiple | **0×** (no dollar win) | — |
| break-even | **N/A** | — |

The two glaring holes v2.0 must fill: **cost multiple = 0** (F1) and **grade = silver** (F2); plus the **54% exact-match** (F3) and an unrunnable local model (F4).

## F1 — Cost: the money table (the headline)

### New/activated metrics
- **teacher $/1k (paid).** `cost_model.teacher_cost_per_1k(pricing, avg_in, avg_out)` with the paid `price_table` row and the *measured* token profile.
- **cost_multiple.** teacher $/1k ÷ student $/1k — the "N× cheaper" number, now > 1.
- **break-even requests/day.** `cost_model.break_even_requests_per_day` — fixed GPU amortization vs per-call API cost.
- **daily cost curves.** `cost_model.daily_cost_curves` feeds the dashboard's crossover chart.

### The money table (illustrative target — 0.5B student, `claude-sonnet-4-5` teacher)

Using the v1.0 student cost ($0.1178/1k) and the config token profile (900 in / 350 out):

| Axis | Teacher (`claude-sonnet-4-5`, API) | Student (distilled, local) | Win |
|---|---|---|---|
| $/1k requests | ≈ **$7.95** (0.9k in × $3 + 0.35k out × $15 /Mtok) | **$0.1178** | ≈ **67× cheaper** |
| p95 latency | network + frontier forward | 5,921 ms (0.5B) | privacy/local, tail varies |
| Data egress (inference) | — | stays local | privacy |
| Quality (field-F1) | 1.0000 (ref) | 0.9647 (96.5%) | meets 95% bar |
| Break-even | — | — | ≈ **127 requests/day** |

**Sensitivity to teacher choice** (same 0.5B student, config token profile):

| Paid teacher | teacher $/1k | cost_multiple | break-even/day |
|---|---|---|---|
| `claude-haiku-4-5` ($1/$5) | ≈ $2.65 | ≈ 22× | ≈ 385 |
| `claude-sonnet-4-5` ($3/$15) | ≈ $7.95 | ≈ 67× | ≈ 127 |
| `claude-opus-4-6` ($15/$75) | ≈ $39.75 | ≈ 337× | ≈ 25 |

> **All money-table figures above are illustrative**, computed from the config's placeholder token profile and the v1.0 amortization ($1,100 GPU / 3 yr, 360 W, $0.15/kWh, 0.226 req/s). F1's deliverable is this table filled with the *measured* paid-teacher token counts. **Note the honest coupling:** a scaled 1–3B student (F3) has lower throughput, so its $/1k rises and the multiple shrinks — the money table must use the *deployed* student's real throughput, not the 0.5B's.

### How F1 is proven
1. `reports/cost_teacher_labeling.json` shows `total_usd > 0` and a real `usd_per_1k_calls`.
2. The money table shows `cost_multiple > 1` and a non-null break-even.
3. The dashboard renders the teacher/student daily-cost crossover at the break-even volume.

## F2 — Human-gold quality & inter-annotator agreement

### New metrics
- **Inter-annotator agreement (IAA).** Field-level agreement rate across annotators + **Cohen's κ** on categorical fields (`currency`, `payment_terms` present/absent). Target: high agreement (κ ≥ 0.8 on categoricals) — a low κ means the labeling guide needs tightening before the gold is trustworthy.
- **Human field-F1 (headline).** The v1.0 metric M recomputed with `allow_unverified_gold: false` over `human_verified: true` items — the true held-out number.
- **Silver→human delta.** The gap between the silver (0.9647) and human field-F1, reported explicitly to *show* the circularity being removed.

### Targets
| Metric | v1.0 (silver) | v2.0 target (human) |
|---|---|---|
| Gold grade | SILVER (cross-model) | **HUMAN-verified** |
| Items scored | 37 | ≥ 60, growing toward a few hundred |
| field-F1 vs teacher | 0.9647 | **≥ 0.95** (bar holds on human ground truth) |
| IAA (categorical κ) | — | **≥ 0.8** |

### How F2 is proven
`eval_report.json` shows `gold_grade: HUMAN` (or equivalent), `n_gold` from human-verified items, `meets_bar: true`, and a companion IAA report; the silver row is retained, labeled, for comparison.

## F3 — Capacity × data ablation

### New deliverable: the ablation table
All cells trained on identical data, scored on the identical **human** gold set:

| Base | Data fraction | field-F1 | exact-match | p95 latency | Fits 16 GB |
|---|---|---|---|---|---|
| Qwen2.5-0.5B (v1.0) | 100% | 0.9647 | 0.5405 | 5.92 s | ✅ full FT |
| Qwen2.5-0.5B | 50% | *tbd* | *tbd* | ~5.9 s | ✅ |
| Qwen2.5-1.5B | 100% | *target ↑* | *target ↑↑* | *↑* | ✅ full FT |
| Qwen2.5-3B | 100% | *target ↑* | *target ↑↑* | *↑↑* | ✅ QLoRA |

*(Only the top row is measured; the rest are the grid to fill.)*

### The questions the ablation must answer
- **How much of the residual field-F1 gap closes with size?** field-F1 is already 96.5%, so the headroom is small — the ablation may well show *diminishing* returns, which is itself an honest, publishable finding.
- **How much of the low 54% exact-match closes with size vs data?** Exact-match is the strictest view (every field of every line item identical). This is the metric most likely to move with capacity; quantifying the split between "bigger model" and "more data" is F3's core result.
- **Is the extra size worth the latency/footprint cost?** A 3B student that adds 2 points of exact-match but triples p95 and quadruples footprint may lose to the 0.5B on the deployment tradeoff — the winner is chosen on quality/latency/footprint, not size.

## Cross-cutting — refreshed failure analysis

The v1.0 gap (96.5% field-F1, 54% exact-match) must be re-categorized on the **human** gold set, clustering exact-match misses by type (line-item boundary errors, money/date normalization edge cases, currency inference, discount/shipping lines) and stating which still need the teacher. `reports/gold_predictions.jsonl` (per-item predictions) already exists as the raw material. This turns the ablation and money numbers into a trustworthy engineering result rather than a leaderboard entry.

## How the overall v2.0 win is proven

All must hold on measured numbers:
1. **Cost (F1):** money table with `cost_multiple > 1` + break-even, from a real paid-teacher run.
2. **Quality (F2):** field-F1 ≥ 95% of teacher on the **human** gold set, with IAA reported.
3. **Scale (F3):** a 1–3B student + the capacity×data table quantifying the exact-match/field-F1 movement.
4. **Runnable (F4):** `ollama run` on a valid GGUF returns schema-valid JSON; schema-valid rate stays ~100%.
5. **Honesty (X):** refreshed failure analysis; every number labeled by grade and clearly illustrative-vs-measured.

## Related docs
- The cost-model plumbing behind the money table: [`src/distil_task/cost_model.py`](../../src/distil_task/cost_model.py)
- Where the numbers surface (dashboard): [`dashboard/app.py`](../../dashboard/app.py)
- Data/resources feeding the eval: [04-data-and-resources.md](04-data-and-resources.md)
- v1.0 metric definitions: [../05-evaluation-metrics.md](../05-evaluation-metrics.md)
- Risks to these measurements: [08-risks-pitfalls.md](08-risks-pitfalls.md)
