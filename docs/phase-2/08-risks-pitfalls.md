# 08 — Risks & Pitfalls (v2.0)

> Risks specific to each v2.0 feature — expanded as risk → why it happens → mitigation — plus a risk register. The v1.0 pitfalls (distilling teacher mistakes, train/test leakage, over-narrow eval, cost hand-waving, format brittleness, ToS, wrong distillation signal, hardware over-reach, volume-over-quality) still apply and are not repeated; see [../08-risks-pitfalls.md](../08-risks-pitfalls.md).

## F1 — Paid frontier teacher

### 1. Runaway API spend
- **Risk.** A full re-distill over thousands of seeds on `claude-opus-4-6` quietly runs up a large bill.
- **Why.** Frontier tokens are cheap per call but add up across a dataset; loops/retries multiply it.
- **Mitigation.** The **content-addressed disk cache** re-bills $0 on identical passes (v1.0 already logged 600/1000 cache hits). Do **M1 price-only first** (30–50 items) to prove the cost curve for cents; gate full re-distill (M3) on that. Prefer `claude-haiku-4-5` or `sonnet` over `opus` for labeling. Watch `reports/cost_teacher_labeling.json` between runs.

### 2. The cost win collapses when the student is scaled
- **Risk.** F3's larger student has lower throughput, raising its $/1k and shrinking the `cost_multiple` the money table advertised for the 0.5B.
- **Why.** The money table's student column is throughput-bound; a 3B is slower than a 0.5B.
- **Mitigation.** Always compute the money table with the **deployed** student's *measured* throughput, never the 0.5B's. Report the cost win for the actual shipped model. The win still holds (an API at ~$8/1k vs even a slow local student at fractions of a dollar), just smaller — state the real number.

### 3. Teacher-quality regression changes the parity baseline
- **Risk.** A different (paid) teacher labels differently than `qwen3:14b`, so "96.5% of teacher" is measured against a *moved* reference.
- **Why.** Parity is relative to the teacher; swapping teachers swaps the 100% mark.
- **Mitigation.** In a full re-distill (M3), re-baseline: score the *paid* teacher on the human gold set as the new reference, and report student-vs-paid-teacher. Do not mix a local-teacher student with a paid-teacher baseline.

### 4. ToS / data egress with a real provider
- **Risk.** Training a task tool on a paid provider's outputs, or sending real client invoices to the API, breaches terms or privacy.
- **Why.** v1.0's local teacher had no egress; a paid API reintroduces both surfaces.
- **Mitigation.** Confirm the provider permits training an **internal/task-specific** tool on outputs; document the stance (`NFR-Comp1`). Use **synthetic seeds** (v1.0's default) so no real client data leaves the box; if real seeds are needed, get consent. State clearly that *inference* stays local even though *training-time labeling* does not.

## F2 — Human-verified gold

### 5. Low inter-annotator agreement
- **Risk.** Annotators disagree enough that the "human gold" is itself noisy, undermining the headline.
- **Why.** Invoice edge cases (which date, line-item boundaries, discount signs) are genuinely ambiguous without a tight guide.
- **Mitigation.** The `LABELING_GUIDE.md` already fixes the hard calls (invoice date not due date; discounts as negative line items; printed `grand_total` wins on arithmetic conflicts). Report **IAA (Cohen's κ)**; if κ is low, tighten the guide and re-verify *before* publishing any number. Adjudicate every disagreement explicitly.

### 6. Gold shrinks below a usable size
- **Risk.** After removing broken docs and keeping only agreed items, too few gold items remain for a stable field-F1 (v1.0 already scored only 37 of 60 silver).
- **Why.** Verification is stricter than cross-model agreement; some items get dropped.
- **Mitigation.** **Expand (M4)** before/alongside verifying so the human-verified count grows, not shrinks; target the SPEC's "few hundred." Report `n_gold` alongside every field-F1 so small-sample caveats are visible.

### 7. Expansion re-introduces leakage
- **Risk.** New gold invoices overlap (or near-duplicate) training items, inflating the score.
- **Why.** New items are sourced the same way as training seeds.
- **Mitigation.** Dedup every new gold item against train/dev/test (embedding/Jaccard, 0.90) — the pipeline already enforces `cross_split_leaks: 0`; extend it to the expanded set. Keep gold frozen and isolated.

## F3 — Scale the student

### 8. 3B doesn't fit / OOMs on 16 GB
- **Risk.** A 3B full FT (~36 GB) OOMs the RTX 5080; even QLoRA is tight with long sequences.
- **Why.** 16 GB is the binding constraint; full FT of 3B is out of reach.
- **Mitigation.** 1.5B stays full FT; **3B uses 4-bit QLoRA** (NF4, double-quant, paged AdamW 8-bit, gradient checkpointing) — the reference config already documented. Lower `per_device_train_batch_size` and raise `gradient_accumulation_steps` to hold the effective batch. The RUNBOOK's OOM guidance applies.

### 9. Scaling barely helps (or hurts) — and that's misread as failure
- **Risk.** field-F1 is already 96.5%, so a bigger student may add almost nothing; a naive reading calls the ablation a failure.
- **Why.** Small headroom means diminishing returns; the gains, if any, concentrate in **exact-match**, not field-F1.
- **Mitigation.** Frame the ablation around **exact-match** (54%, real headroom) and the **size-vs-data split**, not field-F1. A "diminishing returns above 0.5B" result is a *legitimate, publishable finding* — it strengthens the "tiny model is enough" story. Choose the deployed model on the quality/latency/footprint tradeoff.

### 10. The offline-base blocker recurs
- **Risk.** The same flaky download that forced the 0.5B fallback blocks the 1–3B bases again.
- **Why.** It was an environment/bandwidth problem, not a one-off.
- **Mitigation.** **Stage bases out-of-band** to a local cache and train from the local path (M5 task); never rely on an on-the-fly hub pull. Keep permissive alternates (Llama-3.2-1B/3B, Gemma-2-2B, Phi-3.5-mini) ready if a Qwen size is unavailable.

## F4 — One-command local run

### 11. GGUF converts but crashes the sampler
- **Risk.** The known failure: a GGUF loads then aborts with `Assertion 'found' … llama-sampling.cpp:660` for this fine-tuned Qwen2.5 base.
- **Why.** Ollama's built-in converter / a mismatched llama.cpp rev mishandles the tokenizer/sampler metadata.
- **Mitigation.** Convert with a **pinned llama.cpp `convert_hf_to_gguf.py`** (not Ollama's built-in), verify the sampler with `llama-cli` *before* declaring done, and keep the dependency-free `FROM .` safetensors import as the documented fallback. The Definition of Done is a **clean sampler load**, not just a produced file.

### 12. Prompt-scaffold mismatch silently degrades quality
- **Risk.** Users paste raw invoice text into `ollama run`, bypassing the `<document>…</document>\nOutput:` scaffold the 0.5B was trained on, and get garbage — blamed on the model.
- **Why.** A 0.5B student trained on one rigid format is brittle to prompt drift.
- **Mitigation.** The emitted `Modelfile` **bakes the scaffold into the ChatML template** (already implemented). Document the quick-start to pass invoice text as the user turn, and expose the schema-enforced `serve/infer.py` path for production. Track schema-valid rate on the served model.

### 13. Stale/auto-generated model card ships
- **Risk.** The TRL stub card (a "time-machine" example, `model="None"`) gets published, misrepresenting the model.
- **Why.** It was auto-generated by the trainer and never replaced.
- **Mitigation.** F4 explicitly replaces it with a real invoice card sourced from `training_recipe.json` + `eval_report.json` + `money_table.md`; treat the stub's removal as a checklist item (M7 DoD).

## Risk register (top items)

Likelihood × impact on the credibility of the v2.0 headline. L/M/H.

| # | Risk | Feature | Likelihood | Impact | Priority | Primary mitigation |
|---|---|---|---|---|---|---|
| 1 | Runaway API spend | F1 | Medium | Medium | High | Cache + price-only first + haiku/sonnet |
| 2 | Cost win collapses when scaled | F1/F3 | High | High | **Critical** | Money table uses deployed student's real throughput |
| 5 | Low inter-annotator agreement | F2 | Medium | High | High | Tight guide + report κ + adjudicate |
| 6 | Gold too small after verify | F2 | Medium | High | High | Expand (M4) before publishing |
| 8 | 3B OOM on 16 GB | F3 | Medium | Medium | Medium | QLoRA + batch/accum tuning |
| 9 | Scaling misread as failure | F3 | High | Medium | High | Frame on exact-match + size-vs-data |
| 11 | GGUF crashes the sampler | F4 | High | High | **Critical** | Pinned llama.cpp convert + sampler verify |
| 3 | Teacher-quality re-baseline | F1 | Medium | Medium | Medium | Re-score paid teacher on human gold |
| 4 | ToS / egress with paid API | F1 | Low | High | Medium | Confirm terms; synthetic seeds; document |
| 12 | Scaffold mismatch at run | F4 | Medium | Medium | Medium | Scaffold baked into Modelfile |

## Related docs
- Measurement detail behind these risks: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- Resource/licensing context: [04-data-and-resources.md](04-data-and-resources.md)
- Where risks land in the plan: [07-build-roadmap.md](07-build-roadmap.md)
- v1.0 pitfalls (still in force): [../08-risks-pitfalls.md](../08-risks-pitfalls.md)
