# 10 — Glossary (v2.0)

> New terms introduced by Phase 2, one clear sentence each. Terms already defined for v1.0 (teacher, student, distillation dataset, field-F1, exact-match, schema-valid rate, teacher-agreement, gold set, break-even volume, QLoRA, GGUF, constrained decoding, ToS, …) are in [../10-glossary.md](../10-glossary.md) and not repeated.

## Cost & the paid-teacher thesis

- **Paid frontier teacher** — a commercial API model (e.g. `claude-sonnet-4-5`) used as the teacher, whose per-token list price makes the dollar cost win measurable — unlike v1.0's free local teacher.
- **Cost arbitrage** — the project's core thesis: paying a frontier API once to generate training data, then serving a distilled local student far more cheaply per request; unprovable in v1.0 because the teacher was free.
- **Cost multiple** — teacher $/1k requests ÷ student $/1k requests; the "N× cheaper" headline. `0×` in v1.0 (free teacher), targeted `> 1` in v2.0.
- **Price-only mode** — a cheap F1 run that scores the paid teacher on a small sample only to measure its real per-request token profile and fill the money table, without a full re-distill.
- **Full re-distill** — the honest end-state F1 run: the paid teacher relabels the seeds, the student is retrained, and parity is re-measured student-vs-paid-teacher.
- **Token profile** — the average input/output tokens per teacher request (config `teacher_avg_input_tokens` / `teacher_avg_output_tokens`) that drives the money-table cost; placeholders in v1.0, measured in v2.0.
- **Content-addressed cache** — the disk cache keyed by a hash of `(provider, model, system, prompt, temperature, max_tokens, salt)`, so re-running an identical paid labeling pass re-bills $0.
- **Daily cost curve** — teacher vs student total daily USD across request volumes (`cost_model.daily_cost_curves`), whose crossover is the break-even point shown on the dashboard.

## Gold verification

- **Silver-grade gold** — the v1.0 gold set, verified by *cross-model agreement* (teacher vs an independent model) rather than by humans; stronger than dev-grade but still model-vs-model.
- **Human-verified gold** — a gold set whose labels a human checked against the source text and flagged `human_verified: true`; the true held-out ground truth the SPEC requires for the headline number.
- **Cross-model circularity** — the weakness of silver gold: the "correct" answer is defined by models, so scoring a distilled model against it partly measures model-vs-model agreement, not correctness; F2 removes it.
- **Adjudication protocol** — the documented rules for resolving field-level disagreements between annotators (arithmetic-sanity tie-breaks; printed `grand_total` wins; broken docs removed).
- **Inter-annotator agreement (IAA)** — how much independent annotators agree; reported as field-level agreement rate + Cohen's κ.
- **Cohen's κ (kappa)** — a chance-corrected agreement statistic for two annotators on categorical fields (`currency`, `payment_terms` present/absent); a low κ signals an ambiguous labeling guide.
- **Gold expansion** — growing the gold set beyond v1.0's 60 items toward the SPEC's "few hundred," deliberately adding hard/edge invoices while staying leak-free.

## Scaling & ablation

- **SPEC band (1–3B)** — the student-size range the SPEC targets; v1.0's 0.5B fell *below* it (a download-driven fallback), and F3 brings the student into it.
- **Capacity × data ablation** — a controlled grid varying model size (0.5B/1.5B/3B) and training-data fraction on identical data and gold, to separate how much quality comes from a bigger model vs more data.
- **Offline base copy** — a base model pre-downloaded to local storage and trained from a local path, so training does not depend on the flaky download that forced v1.0's 0.5B choice.
- **Capacity gap** — the portion of the residual student-vs-teacher gap (especially the low 54% exact-match) attributable to model size rather than data quality/quantity.
- **Diminishing returns (in this context)** — the honest possibility that, since field-F1 is already 96.5%, a larger student adds little on field-F1 (headroom concentrates in exact-match) — a legitimate finding that strengthens the "tiny model is enough" story.
- **Deployment tradeoff** — choosing the shipped student by balancing quality (field-F1/exact-match) against latency and footprint, not by size alone.

## Serving & packaging

- **`convert_hf_to_gguf.py`** — llama.cpp's Hugging-Face-to-GGUF converter; F4 uses it (not Ollama's built-in converter) to produce a GGUF that loads cleanly.
- **Sampler crash** — the known failure where a GGUF loads but aborts with `Assertion 'found' … llama-sampling.cpp:660` for the fine-tuned Qwen2.5 base; F4's Definition of Done is a clean sampler load, not merely a produced file.
- **Sampler verification** — running the GGUF through `llama-cli` (or Ollama) to confirm it loads and generates without the sampler crash before declaring F4 done.
- **`FROM .` import** — Ollama's fallback that imports the safetensors directory directly (no llama.cpp needed); the current v1.0 path, kept as a documented backup.
- **Prompt scaffold** — the exact `<document>…</document>\nOutput:` wrapper the student was trained on, baked into the Modelfile's ChatML template so `ollama run` matches training; omitting it silently degrades a small student.
- **Model card (real)** — the invoice-extraction Hugging Face card (recipe, base, data size, eval table, quick-start, license) that F4 substitutes for the auto-generated TRL stub currently in `models/student/README.md`.

## Related docs
- These terms in the pipeline: [02-architecture.md](02-architecture.md)
- Metric/cost definitions in depth: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- Source references: [09-references.md](09-references.md)
- v1.0 glossary: [../10-glossary.md](../10-glossary.md)
