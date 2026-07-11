# 08 — Risks & Pitfalls

> Every common pitfall from the SPEC, expanded into risk → why it happens → concrete mitigation, plus a short risk register (likelihood × impact) for the top items.

## Pitfalls (from SPEC §10), expanded

### 1. Distilling the teacher's mistakes
- **Risk.** The student faithfully imitates the teacher's *errors*, especially on edge cases, baking wrong behavior into the model.
- **Why it happens.** The teacher is treated as an oracle, but frontier models still hallucinate and slip — particularly on unusual or ambiguous inputs. If every teacher output is trusted, its mistakes become training targets.
- **Mitigation.** Filter/verify — don't blindly trust teacher output. Schema-validate, dedup, and run a **consistency check** (double-run agreement or a second-model checker). Keep a **human-verified gold set** as independent ground truth so teacher errors can't silently define "correct."

### 2. Train/test leakage
- **Risk.** Training data (or near-duplicates of it) appears in the test/gold set, inflating measured quality and invalidating the parity claim.
- **Why it happens.** Teacher-generated data is easy to accidentally reuse across splits; near-duplicates evade naive exact-match dedup.
- **Mitigation.** The **gold test must be human-made and never in training**. **Dedupe across splits** (not just within) using embedding similarity so near-duplicates are caught. Freeze and isolate the gold set from all generation/training loops.

### 3. Over-narrow eval
- **Risk.** The test set covers only easy cases, so a high score doesn't reflect real-world performance.
- **Why it happens.** Teacher-generated inputs skew toward clean, typical examples; hard/edge cases are underrepresented unless deliberately included.
- **Mitigation.** Make the test set **cover realistic input diversity, not just easy cases**. Deliberately seed edge cases and realistic noise; construct the gold set to be **ecologically valid** (reflects the production input distribution). See [04-data-and-datasets.md](04-data-and-datasets.md).

### 4. Cost hand-waving
- **Risk.** The cost win is overstated by treating local inference as "free," making the headline number indefensible.
- **Why it happens.** GPU hardware and electricity are easy to ignore because they're not a per-call invoice; "it runs locally" feels free.
- **Mitigation.** Include **amortized GPU cost + electricity**, not just "it's free locally." Be honest — it still wins at volume. Report the **break-even request volume** so the claim survives scrutiny (see [05-evaluation-metrics.md](05-evaluation-metrics.md)).

### 5. Format brittleness
- **Risk.** The small model occasionally emits malformed output (e.g., invalid JSON), crashing downstream systems.
- **Why it happens.** Small models are less reliable at strict formatting than frontier models; a small fraction of outputs slip the schema.
- **Mitigation.** **Enforce schema at inference** — constrained decoding and/or retries — so occasional format slips don't crash downstream. Track the **schema-valid rate** as a robustness metric.

### 6. ToS (terms of service)
- **Risk.** Training on the teacher's outputs violates the provider's terms, creating legal/compliance exposure.
- **Why it happens.** Some providers restrict using outputs to train competing models; this is easy to overlook when focused on the technical build.
- **Mitigation.** **Confirm the teacher provider allows training on outputs for your use case.** Frame the artifact as an **internal/task-specific tool**, stay within terms, and **document the compliance stance** (see [04-data-and-datasets.md](04-data-and-datasets.md)).

## Additional watch-items (grounded in SPEC)

### 7. Wrong distillation signal for the teacher's access
- **Risk.** Attempting logit / soft-label KD against an API-only teacher — which is impossible, since APIs don't expose token distributions.
- **Why it happens.** Logit KD is the "textbook" distillation; it's tempting to reach for it without checking teacher access.
- **Mitigation.** Match signal to access: **synthetic-data / sequence-level KD** for API-only teachers; **soft-label KL** only with an **open-weights teacher** (see [02-architecture.md](02-architecture.md)).

### 8. Hardware over-reach on 16GB
- **Risk.** Choosing a student too large to train comfortably on the RTX 5080 (16GB), causing OOM or forcing awkward configs.
- **Why it happens.** Bigger base = tempting quality, but 13B is tight even with QLoRA on 16GB.
- **Mitigation.** Follow the fit table — ≤3B full FT ✅, 7–8B QLoRA ✅, 13B ⚠️ (only with care). For a case study, **1–3B full FT is the sweet spot**; smaller also strengthens the "runs on a laptop" story.

### 9. Volume-over-quality data
- **Risk.** Chasing a huge dataset instead of a clean, diverse one, wasting compute and inflating the set with near-duplicates.
- **Why it happens.** "More data = better" instinct; teacher generation makes scaling cheap.
- **Mitigation.** For a narrow task, **2k–20k clean examples** usually suffice — **quality > quantity**, and **diversity matters more than volume**.

## Risk register (top items)

Likelihood and impact are rated Low / Medium / High. "Impact" is impact on the credibility of the headline claim if unmitigated.

| # | Risk | Likelihood | Impact | Priority | Primary mitigation |
|---|---|---|---|---|---|
| 2 | Train/test leakage | Medium | High | **Critical** | Human-made gold set, dedupe across splits |
| 4 | Cost hand-waving | High | High | **Critical** | Amortized GPU + electricity; break-even volume |
| 1 | Distilling teacher's mistakes | High | Medium | High | Filter + consistency check + human gold |
| 3 | Over-narrow eval | Medium | High | High | Ecologically valid, diverse gold set |
| 5 | Format brittleness | Medium | Medium | Medium | Constrained decoding / retries; schema-valid rate |
| 6 | ToS violation | Low | High | Medium | Confirm terms; document compliance stance |
| 7 | Wrong distillation signal | Low | Medium | Low | Match signal to teacher access |
| 8 | Hardware over-reach (16GB) | Low | Medium | Low | Follow fit table; prefer 1–3B full FT |

## Related docs

- Metric and cost-model detail: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- Data filtering and gold-set discipline: [04-data-and-datasets.md](04-data-and-datasets.md)
- Distillation-signal decision: [02-architecture.md](02-architecture.md)