# 04 — Data & Datasets

> The data plan: generate the distillation set with the teacher, filter it ruthlessly, split without leakage, and hand-verify a gold test set the student never trains on.

## Core principle

You **generate** the dataset with the teacher — this is cheaper and more controllable than finding one. The teacher acts as an oracle that labels (or fully synthesizes) task inputs; your job is to turn its raw outputs into a clean, diverse, leak-free training corpus plus an independent human ground truth.

> **Quality > quantity.** For a narrow task, **2k–20k** clean examples often suffice for a small student. Diversity of inputs matters more than raw volume.

## Generation recipe

From SPEC §4, the four-step recipe:

### 1. Seed inputs
Collect or synthesize realistic inputs for the task. Sources, in rough order of value:
- **Real client documents** (best signal for a real ROI story) — subject to privacy/consent.
- **Public samples** representative of the task domain.
- **Teacher-generated diverse inputs** — ask the teacher to invent varied, realistic inputs when real data is scarce.

The objective is **coverage of realistic input diversity**, not volume. Include edge cases and hard inputs deliberately, because an over-easy input distribution produces an over-narrow eval later (see [08-risks-pitfalls.md](08-risks-pitfalls.md)).

### 2. Teacher outputs
Prompt the teacher with:
- a **strong, fixed instruction**,
- **few-shot examples**, and
- a **strict output schema**.

Ask for **rationales** if you want CoT / sequence-level distillation (richer supervision the student can imitate). Keep the prompt fixed across the run so the dataset is internally consistent.

### 3. Filter ruthlessly
Every teacher output must pass three gates before entering the set:

- **Schema-validate** with `pydantic` / `jsonschema` — drop malformed outputs. Nothing that fails schema validation enters the dataset.
- **Dedup** with embedding similarity — remove near-duplicates so the set isn't inflated by repetition (which also skews eval and wastes training compute).
- **Consistency check** — sample-run the teacher **twice** and keep agreeing items, **or** use a **second model as a checker**. This catches unstable / low-confidence teacher outputs before they become training targets.

### 4. Human-verify a gold test set
Hand-check a few-hundred-item test set. **The small model must never train on this** — it is the ground truth against which the headline parity claim is proven. This set is the eval's backbone and the reason the case study is credible.

## Splits

| Split | Source | Used for | Trained on? |
|---|---|---|---|
| **train** | teacher-generated + filtered | fine-tuning the student | ✅ yes |
| **dev** | teacher-generated + filtered | validation during training / early stopping | ✅ (as val) |
| **test (machine)** | teacher-generated + filtered | quick automated checks | ❌ no |
| **gold test (human-verified)** | human-checked | the headline eval + teacher-agreement anchor | ❌ **never** |

**Leakage discipline:** de-duplicate **across** splits, not just within. An item (or a near-duplicate by embedding similarity) that appears in train must not appear in dev/test/gold. Train/test leakage is one of the named pitfalls and would invalidate the parity claim.

## Scale targets

- **Distillation set:** **2k–20k** clean, filtered examples for a narrow task. Start small; grow only if the quality gate demands it.
- **Human-verified gold test set:** a **few hundred** items, chosen to span the realistic input distribution (including hard/edge cases).
- **Fresh unseen input pool:** a separate stream of unseen inputs for measuring **teacher-agreement** at eval time (not part of train/dev/test/gold).

## Held-out / ecological validity

The gold test set is the **ecological** anchor: it must reflect the *real* input distribution the student will face in production, not a sanitized subset. Practically:
- Include realistic noise, formatting variety, and hard cases.
- Keep it **human-made and human-verified** so it is not contaminated by teacher errors.
- Keep it **frozen** and **isolated** from all training and generation loops.

Teacher-agreement is measured on a *separate fresh unseen pool* precisely so it reflects generalization to new inputs rather than memorization.

## Labeling

For this project, "labeling" is primarily the **teacher acting as labeler**, with two human touch-points:
- **Human verification** of the gold test set (authoritative labels).
- Optional **human spot-checks** of filtered training items to audit teacher quality and calibrate the consistency check.

Where the teacher produces rationales, those rationales are part of the training target (sequence-level KD) but are **not** scored as ground truth — only the final task output is scored against the gold labels.

## Data validation contract

Every record that enters the distillation set should satisfy:
1. **Valid** against the task's `pydantic`/`jsonschema` schema.
2. **Non-duplicate** (below the embedding-similarity threshold vs the rest of the corpus and vs other splits).
3. **Consistent** (passed the double-run agreement or second-model check).

Records failing any check are dropped and logged (drop reasons are useful diagnostics for teacher prompt quality).

## Licensing & ToS notes

From SPEC §4:

- **Check the teacher provider's terms** regarding using outputs to train (potentially competing) models.
- **Frame the artifact as an internal / task-specific tool** and stay within terms.
- **Document your compliance stance** explicitly (this is a deliverable-adjacent artifact, not an afterthought).
- For **student base models**, pick a **permissive license** appropriate to the task (see candidate bases in [06-environment-setup.md](06-environment-setup.md)).
- For **seed inputs sourced from real client data**, respect the client's privacy/consent and data-handling agreements — one of the project's selling points is that inference keeps data local.

## Related docs

- Where the data flows in the system: [02-architecture.md](02-architecture.md)
- How the gold set is scored and the win proven: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- Leakage / teacher-mistake pitfalls and mitigations: [08-risks-pitfalls.md](08-risks-pitfalls.md)