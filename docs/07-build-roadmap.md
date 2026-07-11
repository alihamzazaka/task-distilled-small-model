# 07 — Build Roadmap

> The phased build plan (Phase 0–4) with objectives, key tasks, and a Definition of Done for each phase, the week-by-week milestone table, and the training skeleton from the SPEC.

## At a glance

| Phase | When | Objective |
|---|---|---|
| **Phase 0** | Day 1 | Define the task + commit the quality bar and metric M |
| **Phase 1** | Week 1 | Generate + filter data; build splits + human gold test |
| **Phase 2** | Week 2 | Train the student on the single 5080 |
| **Phase 3** | Week 2–3 | Evaluate: task metric, teacher-agreement, cost/latency |
| **Phase 4** | Week 3 | Package: model, eval report, cost dashboard, demo, blog |

## Phase 0 — Define task + bar (day 1)

**Objective.** Lock the task and the standard of success before spending any teacher tokens or GPU hours.

**Key tasks.**
- Write the **task spec** and the **output schema** (the exact structure the teacher and student must produce).
- Set the **quality bar** — e.g., *"student must reach ≥ 95% of teacher on metric M on the held-out test."*
- **Decide the metric M now** (exact-match / F1 / accuracy / schema-valid rate — see [05-evaluation-metrics.md](05-evaluation-metrics.md)).
- Pick the **student base** (Qwen2.5-1.5B/3B, Llama-3.2-1B/3B, Phi-3.5-mini, Gemma-2-2B) and confirm its license.
- Confirm the **teacher** and its **ToS** for training on outputs.

**Definition of Done.**
- A written task spec + output schema exists.
- Metric M and the numeric quality bar are committed in writing.
- Student base and teacher are chosen; ToS stance documented.

## Phase 1 — Generate + filter (week 1)

**Objective.** Produce a clean, diverse distillation dataset and an independent human-verified gold test set.

**Key tasks.**
- **Teacher generation** — strong fixed instruction + few-shot + strict schema; add rationales if doing CoT/sequence-level KD.
- **Validation** — schema-validate every output (`pydantic`/`jsonschema`); drop malformed.
- **Dedup** — embedding-similarity near-duplicate removal, within and across splits.
- **Consistency check** — double-run agreement or a second-model checker.
- **Splits** — train / dev / test, de-duplicated across each other.
- **Human-verified gold test** — hand-check a few hundred items; never train on them.

**Definition of Done.**
- 2k–20k clean, filtered examples split into train/dev/test with no cross-split leakage.
- A human-verified gold test set (a few hundred items) exists and is isolated from training.
- Drop reasons logged (schema fails, dupes, inconsistencies) as a data-quality record.

## Phase 2 — Train student (week 2)

**Objective.** Fine-tune the 1–3B student on the single RTX 5080.

**Key tasks.**
- Load the base with **Unsloth** (or TRL `SFTTrainer` + PEFT).
- Full fine-tune for ≤3B (bf16 + gradient checkpointing); QLoRA only if scaling to 7B.
- Track the run in **W&B / TensorBoard**; validate on the dev split.

**Training skeleton (from SPEC §6, Phase 2):**

```python
# Unsloth SFT skeleton (single 5080)
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B", max_seq_length=2048, load_in_4bit=False)  # 3B full FT fits
model = FastLanguageModel.get_peft_model(model, r=0)  # or LoRA for 7B

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=train_ds,
    args=SFTConfig(per_device_train_batch_size=8, gradient_accumulation_steps=4,
                   learning_rate=2e-5, num_train_epochs=3, bf16=True,
                   gradient_checkpointing=True, logging_steps=20,
                   eval_strategy="steps", eval_steps=200, output_dir="student"))
trainer.train()
```

**QLoRA reference config (if going 7B):** 4-bit NF4, double quant, LoRA `r=16–64`, `alpha=16–32`, target all linear layers, gradient checkpointing, paged AdamW 8-bit, bf16 compute.

**Definition of Done.**
- A trained student checkpoint exists, produced on the single 5080.
- The training run is tracked and reproducible (recipe recorded: base, data size, hyperparameters).
- Dev-split metrics look sane (no obvious divergence/overfitting).

## Phase 3 — Evaluate (week 2–3)

**Objective.** Measure the student against the bar and the teacher across quality, cost, and latency.

**Key tasks.**
- **Task metric** on the gold test (exact-match/F1 for extraction, accuracy for classification, etc.).
- **Teacher-agreement** — does the student match the teacher on a fresh, unseen input pool?
- **Cost/latency** — measure teacher $/1k and student throughput; compute **break-even volume**; record p50/p95.
- **Quality gate** — if below bar, loop back to Phase 1 (more/better/more-diverse data).

**Definition of Done.**
- The money table is filled with measured numbers (see [05-evaluation-metrics.md](05-evaluation-metrics.md)).
- Student meets the pre-committed bar (≥95% of teacher on M), or a concrete iteration plan is in motion.
- Failure analysis categorizes the residual gap and names which cases still need the teacher.

## Phase 4 — Package (week 3)

**Objective.** Ship the case study as reusable, shareable artifacts.

**Key tasks.**
- Publish the **model on HF** with a card (recipe, base, data size, eval table).
- Write the **eval report** (notebook or markdown).
- Build the **cost dashboard** (break-even volume page).
- Provide the **Ollama one-liner** to run it (`ollama run your-model`) + a short demo video.
- Write the **blog post**: *"How I got frontier-level [task] at 1/40th the cost on a single GPU."*

**Definition of Done.**
- All five SPEC §8 deliverables exist and are linked from the README.
- The one-command local run works from a clean machine.
- The blog post tells the honest cost/quality story with the measured numbers.

## Milestones & timeline (from SPEC §9)

| Week | Milestone |
|---|---|
| 1 | Task spec + generated/filtered dataset + gold test |
| 2 | Student trained, first eval vs bar |
| 3 | Cost/latency measured, package + blog shipped |

## Related docs

- Requirements each phase satisfies: [03-requirements.md](03-requirements.md)
- Data-generation detail for Phase 1: [04-data-and-datasets.md](04-data-and-datasets.md)
- Metric detail for Phase 3: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- Install detail before Phase 2: [06-environment-setup.md](06-environment-setup.md)
- Pitfalls to watch each phase: [08-risks-pitfalls.md](08-risks-pitfalls.md)