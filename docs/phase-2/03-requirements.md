# 03 — Requirements (v2.0)

> Functional and non-functional requirements for each of the four v2.0 features, explicit scope boundaries, assumptions, dependencies, and a deliverables checklist. These extend — never replace — the v1.0 requirements in [../03-requirements.md](../03-requirements.md).

## Requirement ID scheme

- **F1-** paid frontier teacher · **F2-** human-verified gold · **F3-** scale the student · **F4-** one-command local run · **X-** cross-cutting.
- Non-functional requirements are prefixed **NFR-**.

## Functional requirements

### F1 — Paid frontier teacher
- **F1-FR-1 — Provider switch.** The pipeline must run end-to-end with `teacher.provider: anthropic`, `teacher.model: claude-sonnet-4-5` set in [`configs/default.yaml`](../../configs/default.yaml), using the existing `AnthropicTeacher` client — no code change required to switch teachers.
- **F1-FR-2 — Real cost accounting.** Every billed teacher call must accumulate real token usage × the `price_table` row into `reports/cost_teacher_labeling.json` (`total_usd`, `usd_per_1k_calls` > 0), via the existing `CostTracker`.
- **F1-FR-3 — Measured token profile.** The paid teacher's measured average input/output tokens per request must overwrite the config placeholders (`teacher_avg_input_tokens: 900`, `teacher_avg_output_tokens: 350`) so the money table uses real numbers.
- **F1-FR-4 — Dollar money table.** `evaluate.py` must emit a money table where `cost_multiple > 1` and a non-null break-even requests/day, computed by `cost_model.cost_multiple` / `break_even_requests_per_day`.
- **F1-FR-5 — Two run modes.** Support both *price-only* (score the paid teacher on a sample to get its token profile; keep the v1.0 student) and *full re-distill* (paid teacher relabels seeds → retrain → re-eval).
- **F1-FR-6 — Cache honored.** Re-running an identical labeling pass must re-bill $0 (content-addressed disk cache), and the report must show `cache_hits`.

### F2 — Human-verified gold set
- **F2-FR-1 — Human verification.** Every headline-scored gold item must have `human_verified: true`, set by a human who checked all fields against the raw text per [`data/gold/LABELING_GUIDE.md`](../../data/gold/LABELING_GUIDE.md).
- **F2-FR-2 — Two independent annotators.** Each item is labeled independently by ≥ 2 annotators before adjudication.
- **F2-FR-3 — Adjudication protocol.** A written protocol resolves field-level disagreements (arithmetic-sanity tie-breaks; printed `grand_total` wins; broken docs removed to `gold_removed.txt`).
- **F2-FR-4 — Inter-annotator agreement.** Report an IAA figure: field-level agreement rate and Cohen's κ for categorical fields (`currency`, `payment_terms` present/absent).
- **F2-FR-5 — Expansion.** Grow the gold set beyond the current 60 items toward the SPEC's "few hundred," deliberately including edge/hard invoices.
- **F2-FR-6 — Circularity removed.** The headline field-F1 must be computed on the human set with `allow_unverified_gold: false`; the silver number is retained only as an explicitly labeled comparison row.
- **F2-FR-7 — Leak-free.** Expanded gold items must be deduped against train/dev/test (embedding/Jaccard) so no gold item (or near-duplicate) was trained on — extending the existing `cross_split_leaks: 0` guarantee.

### F3 — Scale the student
- **F3-FR-1 — Offline base staging.** A 1–3B base (Qwen2.5-1.5B and Qwen2.5-3B-Instruct) must be available offline on the GPU box, bypassing the download that blocked v1.0.
- **F3-FR-2 — Train in the spec band.** Produce at least one student in the SPEC's 1–3B band on the single RTX 5080 (1.5B full FT; 3B via QLoRA 4-bit).
- **F3-FR-3 — Capacity × data ablation.** Train a grid over {0.5B, 1.5B, 3B} × {≥2 data fractions}, all on identical data and scored on the identical human gold set.
- **F3-FR-4 — Gap quantification.** Report each cell's field-F1 and exact-match, and state explicitly how much of the residual field-F1 gap and the v1.0 **54% exact-match** closes with size vs data.
- **F3-FR-5 — Reproducible recipe.** Each cell writes a `training_recipe.json` (base, hyperparameters, data size, metrics), as v1.0 already does.
- **F3-FR-6 — Winner selection.** Pick the deployed student by a documented quality/latency/footprint tradeoff, not size alone.

### F4 — One-command local run
- **F4-FR-1 — Valid GGUF.** Produce a GGUF via llama.cpp's `convert_hf_to_gguf.py` (+ `llama-quantize`) that loads in llama.cpp/Ollama **without** the `Assertion 'found' … llama-sampling.cpp:660` crash.
- **F4-FR-2 — One-command run.** `ollama create distil-invoice -f Modelfile` then `ollama run distil-invoice "<invoice text>"` must return schema-valid `Invoice` JSON, using the scaffold-preserving Modelfile `export_ollama.py` already emits.
- **F4-FR-3 — Real model card.** Replace the auto-generated TRL stub [`models/student/README.md`](../../models/student/README.md) (currently a "time-machine" example) with an invoice-extraction card: base, data size, eval table, quick-start, licensing.
- **F4-FR-4 — Programmatic serving.** `serve/infer.py`'s parse → validate → retry loop must run over the GGUF/merged model for production calls.
- **F4-FR-5 — Demo video.** A short screen recording of the one-command run extracting a real invoice.
- **F4-FR-6 — Blog post.** "How I got frontier-level invoice extraction at 1/Nth the cost on a single GPU," using the measured v2.0 numbers.

### Cross-cutting
- **X-FR-1 — Refreshed failure analysis.** Re-categorize the residual gap on the human gold set (especially exact-match misses), naming which cases still need the teacher.
- **X-FR-2 — Updated dashboard.** The Streamlit cost dashboard reads the new `eval_report.json` and shows the real break-even crossover.

## Non-functional requirements

- **NFR-Q1 — Quality bar unchanged.** Student field-F1 ≥ **95%** of teacher on the **human** gold set (v1.0 achieved 96.5% on *silver*; v2.0 must hold on human).
- **NFR-Q2 — Eval integrity.** Human-made gold, deduped across splits, ecologically valid (edge cases included).
- **NFR-C1 — Real dollar win.** With a paid teacher, `cost_multiple` must be materially > 1 and the break-even volume published (student side stays $0.1178/1k unless re-measured).
- **NFR-C2 — Honest cost model.** Student cost stays amortized GPU + electricity (never "free locally"); paid-teacher cost is real API list price × measured tokens.
- **NFR-L1 — Latency reported.** p50/p95 and throughput reported per ablation cell; note that a larger student *raises* student latency above the 0.5B baseline (p95 5.92 s) — an honest tradeoff, not hidden.
- **NFR-H1 — Single GPU.** All training/inference fits the RTX 5080 (16 GB): 1.5B full FT ✅, 3B QLoRA ✅, 3B full FT ✗ (~36 GB).
- **NFR-P1 — Privacy note.** With a paid teacher, *training-time* data leaves the org (teacher API); *inference-time* data still stays local. This distinction must be stated, not blurred.
- **NFR-R1 — Format robustness.** Schema enforced at inference (constrained-output retry loop) over the GGUF.
- **NFR-R2 — Reproducibility.** Recipes, seeds (v1.0 used `20260707`), and configs tracked per run.
- **NFR-Comp1 — ToS.** Confirm the paid provider permits training a task-specific tool on its outputs; document the stance (see [04-data-and-resources.md](04-data-and-resources.md)).

## In scope

- Distilling from / pricing against a paid frontier teacher; a dollar money table + break-even.
- Human verification + expansion of the gold set with adjudication and IAA.
- A 1–3B student and a capacity×data ablation on the single GPU.
- A verified GGUF, a real model card, a demo video, and a blog.
- A refreshed failure analysis and an updated cost dashboard.

## Out of scope

- Changing the task, `Invoice` schema, or metric M (locked Phase 0).
- A new distillation algorithm or logit KD against an API teacher.
- Beating the teacher on quality; general-purpose capability.
- Multi-GPU / datacenter training; a full production MLOps platform.
- 13B+ students as the headline (tight QLoRA edge case only).
- Retiring the local-teacher or silver-comparison paths (kept, just not the headline).

## Assumptions

- A paid frontier API key (Anthropic) and budget are available; ToS permits the use.
- The 1–3B bases can be staged offline onto the GPU box.
- ≥ 2 annotators are available for gold verification.
- llama.cpp's `convert_hf_to_gguf.py` supports the fine-tuned Qwen2.5 base once the tokenizer/metadata path is correct.
- The existing cache/cost/eval plumbing works unchanged against the paid provider (it is already provider-agnostic).

## Dependencies

- **Teacher:** Anthropic API (`claude-sonnet-4-5` default; `claude-haiku-4-5` cheaper, `claude-opus-4-6` for a top-quality reference) — price rows already in the config.
- **Bases:** Qwen2.5-1.5B, Qwen2.5-3B-Instruct (offline copies).
- **Serving/convert:** a pinned llama.cpp checkout (`convert_hf_to_gguf.py`, `llama-quantize`), Ollama.
- **Annotation:** the `LABELING_GUIDE.md` + a lightweight IAA script.
- Everything else (TRL/Unsloth, datasets, pydantic, cost_model, dashboard) is unchanged from Phase 1 ([../06-environment-setup.md](../06-environment-setup.md)).

## Deliverables checklist

- [ ] Paid-teacher run: `reports/cost_teacher_labeling.json` with `total_usd > 0`; money table with `cost_multiple > 1` + break-even.
- [ ] Human-verified gold set: `human_verified: true` on the scored items + an IAA report.
- [ ] Adjudication protocol document + `gold_removed.txt` for discarded items.
- [ ] 1–3B student checkpoint(s) + `training_recipe.json` per cell.
- [ ] Capacity × data ablation table (field-F1 + exact-match per cell).
- [ ] Valid GGUF (`convert_hf_to_gguf.py`) that loads without the sampler crash.
- [ ] Working `ollama run distil-invoice "<invoice>"` returning schema-valid JSON.
- [ ] Real invoice-extraction model card (replaces the TRL stub).
- [ ] Demo video + blog post with the measured numbers.
- [ ] Refreshed failure analysis; updated cost dashboard.

## Related docs
- Why each feature exists: [01-overview.md](01-overview.md)
- How each is measured: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- Resources/licensing per requirement: [04-data-and-resources.md](04-data-and-resources.md)
- Build order and Definition of Done: [07-build-roadmap.md](07-build-roadmap.md)
- v1.0 requirements this extends: [../03-requirements.md](../03-requirements.md)
