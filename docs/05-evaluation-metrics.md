# 05 — Evaluation & Metrics

> Precise metric definitions, the named baseline to beat, the cost/quality "money table," the eval harness, and exactly how the headline win is proven — with all example numbers kept as illustrative placeholders.

## What the eval must establish

The project lives or dies on a single measured claim: **the student reaches ≥ 95% of the teacher's quality on metric M, on a human-verified held-out set, at ~1/40th the cost, on one consumer GPU.** The eval therefore has four pillars: **task quality**, **teacher-agreement**, **cost**, and **latency** — each defined precisely below.

## Metric definitions

### Task-native metric M
The primary quality metric, chosen to fit the task and committed in Phase 0 **before** any data is generated. Options from SPEC:
- **Exact match** — fraction of outputs identical to the gold answer (natural for strict structured extraction).
- **F1** — token/field-level overlap between prediction and gold (for extraction where partial credit is meaningful).
- **Accuracy** — fraction correct (for classification / triage / intent detection).
- **Schema-valid rate** — fraction of outputs that parse against the task schema (a robustness metric; see below).

M is always computed on the **human-verified gold test set**.

### Schema-valid rate
The fraction of student outputs that validate against the task's `pydantic`/`jsonschema` schema. This is both a **quality** signal and a **robustness** signal: a low schema-valid rate means downstream systems will choke, which is why schema is also enforced at inference via constrained decoding / retries.

### Teacher-agreement %
The fraction of a **fresh, unseen input pool** on which the student's output matches the teacher's output. This measures **generalization to new inputs** (does the student behave like the teacher off the gold set?), complementing M (which measures correctness against human ground truth). It is deliberately computed on inputs outside train/dev/test/gold.

### Cost per 1k requests
- **Teacher (API):** the provider's price for 1,000 task requests (`$X`).
- **Student (local, 5080):** **amortized GPU cost + electricity** for 1,000 requests — *not* "it's free locally." Honest amortization is mandatory (see [08-risks-pitfalls.md](08-risks-pitfalls.md)).
- **Break-even volume:** the requests/day above which running the local student is cheaper than the teacher, given fixed GPU amortization vs per-call API cost.

### Latency (p50 / p95)
Wall-clock time per request at the 50th and 95th percentiles, measured for both teacher and student under comparable conditions. p95 is the headline latency number because it reflects tail behavior users actually feel.

## Baselines to beat (named)

The **frontier teacher** is the reference baseline (its quality on M is defined as 100% / the ceiling). The student is judged as a **percentage of the teacher**, not on an absolute leaderboard. Named precedents that establish task-specific distillation is achievable (context, not competitors):

- **Prometheus / Prometheus 2** — distilled evaluator (Mistral fine-tunes).
- **Distil-Whisper** — task-specific distillation done well.
- **Alpaca / Vicuna / Self-Instruct** — synthetic-data distillation precedent.

The win is **not** "beat GPT" — it is "reach ~parity with the teacher on M at a fraction of the cost."

## The money table (from SPEC §7)

| Metric | Teacher (API) | Student (local, 5080) | Win |
|---|---|---|---|
| $/1k requests | e.g. $X | ~electricity + amortized GPU | e.g. **40×** |
| p95 latency | e.g. 2.1s | e.g. 180ms | lower |
| Data egress | leaves org | stays local | privacy |
| Quality (metric M) | 100% (ref) | e.g. **96%** | acceptable |

> **All figures ($X, 40×, 2.1s, 180ms, 96%) are illustrative placeholders.** The deliverable is this table filled with *your* measured numbers. The contract is: quality within tolerance of the teacher, at a large cost win and a latency win, with data staying local.

## How the win is proven

The headline claim is proven when the eval report shows **all** of:

1. **Quality parity within tolerance** — M(student) ≥ 95% of M(teacher) on the **human-verified gold test set** (the exact bar committed in Phase 0). Illustrative: 96%.
2. **Teacher-agreement** — a high agreement % on a fresh unseen pool, demonstrating the student generalizes like the teacher, not just memorizes gold.
3. **Cost win** — the money table's $/1k row shows a large multiple (illustrative 40×), computed with **amortized** GPU + electricity, plus the **break-even request volume**.
4. **Latency win** — student p95 materially below teacher p95 (illustrative 180ms vs 2.1s).
5. **Privacy** — data egress stays local (qualitative but real).
6. **Failure analysis** — the residual gap is categorized, stating which cases still need the teacher (optionally routed up). This is what turns "96%" into a trustworthy, actionable number.

## Failure analysis (required)

Categorize the residual gap between student and teacher: cluster the student's misses by type (e.g., rare formats, long inputs, ambiguous cases, specific field errors) and state **clearly which cases still need the teacher**. You may even **route hard cases up** to the teacher — trading a little cost for coverage — which ties into model-routing work. Failure analysis is a first-class deliverable, not an appendix: it is the difference between a demo and an engineering case study.

## Eval harness & tooling (from SPEC §3)

| Purpose | Tool |
|---|---|
| Standardized eval scaffolding | **`lm-evaluation-harness`** |
| Task-specific scoring | custom task scorers (exact-match / F1 / accuracy / schema-valid) |
| A/B comparison (teacher vs student) | **promptfoo** or **Braintrust** |
| Schema validation of outputs | `pydantic` / `jsonschema` |
| Cost/latency reporting | simple logging → a small **Streamlit / HTML** dashboard |
| Run tracking | **W&B / TensorBoard** |

The eval runs are logged, and the cost/latency numbers feed the **cost dashboard** deliverable, which visualizes the break-even volume ("above N requests/day, local wins").

## Reporting deliverables (eval-specific)

- **Eval report** (notebook or markdown): task metric M + teacher-agreement + cost/latency, on the gold set and fresh pool.
- **Cost dashboard**: interactive page showing the break-even volume.
- **Model card**: base model, data size, recipe, and the eval table.

See the full deliverables list in [03-requirements.md](03-requirements.md).

## Related docs

- Success criteria / headline claim: [01-overview.md](01-overview.md)
- Gold set and fresh-pool construction: [04-data-and-datasets.md](04-data-and-datasets.md)
- Cost-hand-waving and over-narrow-eval pitfalls: [08-risks-pitfalls.md](08-risks-pitfalls.md)