# 03 — Requirements

> Functional and non-functional requirements, explicit scope boundaries, assumptions, dependencies, and the concrete deliverables checklist for the task-distilled small-model case study.

## Functional requirements

The system must:

- **FR-1 — Task definition.** Capture a single narrow task as a written spec: task description, a strict **output schema**, and a rubric the teacher follows.
- **FR-2 — Teacher generation.** Produce `input → gold output` pairs from a frontier (or strong open) teacher using a fixed strong instruction + few-shot examples + strict schema, optionally with rationale/CoT traces.
- **FR-3 — Seed-input sourcing.** Assemble realistic, **diverse** seed inputs (real client docs, public samples, or teacher-generated), prioritizing diversity over volume.
- **FR-4 — Quality filtering.** Schema-validate every teacher output (`pydantic`/`jsonschema`) and drop malformed items; dedup by embedding similarity; apply a consistency check (double-run agreement or a second-model checker).
- **FR-5 — Dataset splitting.** Produce de-duplicated train / dev / test splits, with no leakage across splits.
- **FR-6 — Human-verified gold test set.** Curate a few-hundred-item, human-checked ground-truth test set that is **never** used in training.
- **FR-7 — Student fine-tuning.** Fine-tune a 1–3B student on the single RTX 5080 (full FT for ≤3B; QLoRA for 7–8B) using Unsloth or TRL `SFTTrainer` + PEFT.
- **FR-8 — Task-metric evaluation.** Score the student with the task-native metric M (exact-match / F1 / schema-valid rate / accuracy) on the gold test set.
- **FR-9 — Teacher-agreement evaluation.** Measure teacher-agreement % on a fresh, unseen input pool.
- **FR-10 — Cost & latency measurement.** Measure teacher $/1k requests, student amortized $/1k (GPU + electricity), and p50/p95 latency for both; compute break-even volume.
- **FR-11 — Quality gate.** Compare the student against the pre-committed bar and either loop (regenerate/retrain) or proceed to packaging.
- **FR-12 — Schema-safe inference.** Enforce the output schema at inference via constrained decoding / retries so occasional format slips don't crash downstream.
- **FR-13 — Local serving.** Serve the student locally with a one-command invocation (`ollama run your-model`).
- **FR-14 — Failure analysis.** Categorize the residual quality gap and state which cases still require the teacher (optionally route hard cases up).
- **FR-15 — Packaging & reporting.** Publish the model + card, eval report, cost dashboard, local run, and blog post.

## Non-functional requirements

### Quality
- **NFR-Q1 — Quality bar.** Student reaches **≥ 95%** of the teacher's score on metric M on the human-verified gold test set (exact bar/metric committed in Phase 0). Illustrative achieved figure in SPEC: **96%**.
- **NFR-Q2 — Eval integrity.** The gold test set is human-made and never trained on; splits are de-duplicated; the test covers realistic input diversity, not just easy cases.

### Cost
- **NFR-C1 — Cost win.** Student per-request cost is materially below the teacher's; illustrative target ≈ **40×** cheaper per 1k requests.
- **NFR-C2 — Honest cost model.** Student cost includes **amortized GPU cost + electricity**, not "it's free locally." Report the break-even request volume above which local wins.

### Latency
- **NFR-L1 — Latency win.** Report p50/p95 for teacher and student; the local student's p95 is materially lower (illustrative: teacher p95 ~2.1s vs student p95 ~180ms).

### Hardware
- **NFR-H1 — Single consumer GPU.** All training and inference fit a single **RTX 5080 (16GB)**. Fit guidance: ≤3B full FT ✅; 7–8B QLoRA (4-bit) ✅; 13B QLoRA ⚠️ tight/possible with care.

### Privacy
- **NFR-P1 — Data locality.** At inference, data stays local (no egress to a third-party API).

### Robustness / operability
- **NFR-R1 — Format robustness.** Schema enforced at inference (constrained decoding / retries).
- **NFR-R2 — Reproducibility.** Training runs are tracked (W&B / TensorBoard); the recipe (base model, data size, hyperparameters, eval) is documented on the model card.

### Compliance
- **NFR-Comp1 — ToS.** Confirm the teacher provider's terms allow using outputs to train the student for this use case; document the compliance stance (frame as an internal/task-specific tool).

## In scope

- One **narrow** task (e.g., structured extraction, classification/triage, query rewriting/SQL for a fixed schema, or fixed-format domain summarization).
- Teacher-generated + filtered distillation dataset and a human-verified gold test set.
- A 1–3B student, full fine-tuned (QLoRA only if scaling to 7B) on a single RTX 5080.
- A measured eval: task metric M, teacher-agreement, cost model, latency.
- Local quantized serving + a one-command run.
- Eval report, cost dashboard, model card, blog post.

## Out of scope

- **Inventing a new distillation algorithm** — this is a rigorous case study, not novel research.
- **Beating the frontier teacher** on quality — the goal is *parity within tolerance* at lower cost, not superiority.
- **General-purpose capability** — the student is deliberately narrow; broad tasks are not evaluated or claimed.
- **Soft-label / logit KD** unless an open-weights teacher is used (API-only teachers don't expose logits).
- **Multi-GPU / datacenter training** — everything targets one consumer GPU.
- **Full production MLOps platform** — logging → a small dashboard is sufficient; no large serving infra.
- **13B+ students** as the headline (13B is only noted as a tight QLoRA edge case).

## Assumptions

- A frontier (or strong open) teacher is accessible and of sufficient quality on the task.
- Enough realistic, diverse seed inputs can be sourced or synthesized.
- For a narrow task, **2k–20k** clean examples suffice for a small student (quality > quantity).
- The single RTX 5080 (16GB) is available for training and inference.
- The task metric M is well-defined and automatically computable against the gold set.
- The teacher provider's ToS permits training on outputs for this internal/task-specific use.

## Dependencies

- **Teacher:** frontier API model or a strong open model (e.g., a 70B) run elsewhere.
- **Student base:** Qwen2.5-1.5B/3B, Llama-3.2-1B/3B, Phi-3.5-mini, or Gemma-2-2B (permissive license + strong base for the task).
- **Fine-tuning:** Unsloth or TRL `SFTTrainer` + PEFT.
- **Serving:** Ollama / llama.cpp / vLLM / TGI.
- **Data tooling:** `datasets`, `pydantic`, `jsonschema`.
- **Eval:** `lm-evaluation-harness`, task-specific scorers, promptfoo / Braintrust for A/B.
- **Tracking:** W&B / TensorBoard.
- **Cost/latency dashboard:** logging → a small Streamlit/HTML page.

Full install detail is in [06-environment-setup.md](06-environment-setup.md).

## Deliverables checklist

From SPEC §8, the shareable deliverables:

- [ ] **HF model + card** — recipe, base model, data size, eval table.
- [ ] **Eval report** (notebook or markdown) — task metric M + teacher-agreement + cost/latency.
- [ ] **Cost dashboard** — a small interactive page showing break-even volume ("above N requests/day, local wins").
- [ ] **One-command local run** — `ollama run your-model` + a short demo video.
- [ ] **Blog post** — "How I got frontier-level [task] at 1/40th the cost on a single GPU."

Supporting artifacts implied by the pipeline:

- [ ] Task spec + output schema + committed quality bar (Phase 0).
- [ ] Filtered distillation dataset (train/dev/test) + human-verified gold test set.
- [ ] Trained student checkpoint(s) with tracked runs.
- [ ] Documented ToS/compliance stance.
- [ ] Failure analysis categorizing the residual gap.

## Related docs

- Success criteria in context: [01-overview.md](01-overview.md)
- Metric definitions and the money table: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- Phased plan mapping requirements to work: [07-build-roadmap.md](07-build-roadmap.md)
