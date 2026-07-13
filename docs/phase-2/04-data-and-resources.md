# 04 — Data & Resources (v2.0)

> The new datasets, models, API access, hardware, and human effort each v2.0 feature needs, with sourcing and licensing notes. The v1.0 data plan ([../04-data-and-datasets.md](../04-data-and-datasets.md)) still governs *how* data is generated, filtered, and split; this doc covers only the *additional* resources v2.0 introduces.

## Resource summary

| Feature | New data | New models | New services/keys | New hardware | Human effort |
|---|---|---|---|---|---|
| **F1** paid teacher | (optional) re-labeled train set from the paid teacher | `claude-sonnet-4-5` (+ `haiku-4-5`, `opus-4-6` refs) | Anthropic API key + budget | none | none |
| **F2** human gold | expanded gold pool (harder invoices) | none | none | none | 2+ annotators |
| **F3** scale student | none (reuses the same distillation data) | Qwen2.5-1.5B, Qwen2.5-3B-Instruct (offline) | none | GPU box + more disk/VRAM budget | none |
| **F4** GGUF/demo | 1 real sample invoice for the demo | none | Ollama; pinned llama.cpp | none | short recording |

## F1 — Paid frontier teacher

### Models & pricing (already in the config)
The `teacher.price_table` in [`configs/default.yaml`](../../configs/default.yaml) already carries the paid rows, USD per million tokens:

| Model | input $/Mtok | output $/Mtok | Role in v2.0 |
|---|---|---|---|
| `claude-sonnet-4-5` | 3.00 | 15.00 | **Default paid teacher** — quality/cost balance |
| `claude-haiku-4-5` | 1.00 | 5.00 | Cheapest paid teacher (widens the cost win; may lower teacher quality) |
| `claude-opus-4-6` | 15.00 | 75.00 | Top-quality reference teacher (biggest quality ceiling; smallest cost win) |
| `qwen*` (local) | 0.00 | 0.00 | v1.0 teacher — retained for the privacy/offline path |

The price table is prefix-matched, so dated model ids (e.g. `claude-sonnet-4-5-20250929`) resolve correctly (`teacher._is`/`CostTracker.price_for`).

### Budget estimate (illustrative)
v1.0's labeling logged **~2.54M input** + **~0.18M output** tokens across 1000 calls (with 600 cache hits). Pricing those *same* tokens at `claude-sonnet-4-5` rates:

- input: 2.54M / 1e6 × $3.00 ≈ **$7.63**
- output: 0.18M / 1e6 × $15.00 ≈ **$2.73**
- **≈ $10** for a full re-label of the ~800-seed set (cache makes re-runs free).

The **price-only** mode is cheaper still: score the paid teacher on a small sample (e.g. 30–50 invoices) just to measure its real per-request token profile, then feed the money table — a few cents to a dollar. *(These are estimates from v1.0 token counts, not a quote.)*

### Sourcing & licensing
- **API access:** an Anthropic API key in `ANTHROPIC_API_KEY` (never committed). The `AnthropicTeacher` client reads it lazily.
- **ToS:** confirm the provider permits using outputs to train a **task-specific internal tool** (invoice extraction), and document the stance — this is the v1.0 `NFR-Comp1` obligation, now with a real provider. Frame the artifact as internal/task-specific, not a general competing model.
- **Data egress caveat:** with a paid teacher, seed invoices are sent to the API at *training time*. If seeds include real client documents, that egress must be consented; synthetic seeds (v1.0's default generation path) avoid the issue. *Inference-time* data still never leaves the box.

## F2 — Human-verified & expanded gold set

### The starting point
[`data/gold/gold_test.jsonl`](../../data/gold/gold_test.jsonl): **60 items**, currently `human_verified: false` for all, with `silver_verified: true` on **37** (cross-model agreement). The verification target is human ground truth for the scored items and expansion toward the SPEC's "few hundred."

### New data: the expansion pool
- **Source.** Additional diverse invoices/receipts drawn the same way as v1.0 seeds (teacher-synthesized diverse docs and/or public samples), deliberately weighted toward **hard/edge cases**: multi-currency, foreign date formats, discount/shipping lines, missing tax, fractional quantities — the exact surfaces `schema.py`'s normalizers target.
- **Leak discipline.** Every new gold item is deduped (embedding/Jaccard, threshold 0.90) against `train/dev/test` so the `cross_split_leaks: 0` property holds for the expanded set.
- **Isolation.** Frozen and never trained on, exactly as v1.0.

### Human effort & tooling
- **Annotators:** ≥ 2 people (or one plus an independent reviewer) working from [`data/gold/LABELING_GUIDE.md`](../../data/gold/LABELING_GUIDE.md), which already specifies field rules and arithmetic-sanity tie-breaks.
- **Adjudication:** disagreements resolved by the documented protocol; unrecoverable docs deleted to `data/gold/gold_removed.txt`.
- **IAA tooling:** a small script computing field-level agreement and Cohen's κ (a new, lightweight addition; no external service).
- **Licensing:** if any real invoices are used as gold inputs, respect the source's privacy/consent; prefer synthetic or clearly licensed public samples for anything shared.

## F3 — Scaling the student

### New models (offline base copies)
The blocker in v1.0 was mundane: "3B download unreliable on this link" (config comment), so a 0.5B fallback shipped. F3 stages bases **offline**:

| Base | Params | License | Fit on RTX 5080 (16 GB) | Approx. disk |
|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` (v1.0) | 0.5B | Apache-2.0 | full FT ✅ | ~1 GB |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Apache-2.0 | full FT ✅ | ~3 GB |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Qwen license | **QLoRA 4-bit** ✅ (full FT ~36 GB ✗) | ~6 GB |

- **Sourcing.** Pre-download to a local cache / staging dir and point `training.base_model` at the local path, so training never depends on the flaky link. Alternative permissive bases from the SPEC (Llama-3.2-1B/3B, Phi-3.5-mini, Gemma-2-2B) remain candidates if a Qwen size is unavailable.
- **License note.** 0.5B/1.5B are Apache-2.0 (clean redistribution); the 3B carries the Qwen license — check its redistribution terms before publishing a 3B-derived model card.

### No new training data
The ablation deliberately holds the **distillation dataset fixed** (the same train 568 / dev 78 from v1.0, plus data-fraction subsamples) so capacity and data effects are separable. Larger bases reuse the exact same corpus.

### Hardware/VRAM budget
All cells fit the single 16 GB card: 1.5B full FT with bf16 + gradient checkpointing; 3B via 4-bit QLoRA (NF4, double-quant, paged AdamW 8-bit — the reference config already in [../06-environment-setup.md](../06-environment-setup.md)). Expect longer train runtimes than v1.0's ~160 s and higher inference latency than the 0.5B baseline.

## F4 — Serving, demo & card

### New tools
- **Pinned llama.cpp checkout** for `convert_hf_to_gguf.py` + `llama-quantize` (the commands `export_ollama.py` prints). Pin a known-good revision to avoid the sampler-crash regression.
- **Ollama** (already a Phase 1 dependency) for `ollama create` / `ollama run`.

### New data (tiny)
- **One real sample invoice** (`data/gold/some_invoice.txt`, referenced by the RUNBOOK) for the demo video and the model-card quick-start — synthetic or licensed, so it can be published.

### Model card inputs
The real card ([`models/student/README.md`](../../models/student/README.md) replacement) pulls from artifacts that already exist: `training_recipe.json` (recipe), `reports/eval_report.json` (eval table), `reports/money_table.md` (cost), and the schema. No new data-gathering — just replacing the auto-generated stub.

## Resource risks (pointers)
- Paid-teacher **budget overrun** → cache + price-only mode (see [08-risks-pitfalls.md](08-risks-pitfalls.md)).
- Base **download flakiness** recurring → offline staging is the mitigation, not a retry.
- Gold **annotator scarcity** → single-annotator + independent-reviewer fallback, with IAA still reported on the overlap.

## Related docs
- How these resources are wired into the pipeline: [02-architecture.md](02-architecture.md)
- Setup steps for the new services/tools: [06-environment-setup.md](06-environment-setup.md)
- Cost math that uses the price table: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- v1.0 data plan (still authoritative for generation/filtering): [../04-data-and-datasets.md](../04-data-and-datasets.md)
