# 01 — Overview

> Why distill a frontier model into a tiny local student for one narrow task, how this differs from the crowded SOTA, and the single headline metric+constraint the project is judged on.

## Problem → solution framing

| | |
|---|---|
| **Problem** | Companies want a specific LLM capability in production but can't afford frontier API calls at scale, can't send data to third parties (privacy), or need low, predictable latency. Per-request frontier cost kills the unit economics. |
| **Solution** | Use the frontier model as a *teacher* to generate high-quality labeled data for one task, fine-tune a small *student* (1–3B) to imitate it, and deploy the student locally. Prove parity on a held-out eval and quantify the cost/latency win. |
| **Why it's rare** | Most "small model" demos skip the eval and the cost math. A disciplined teacher → student → **measured** pipeline is exactly the LLMOps competence enterprises pay for. |

The core move is simple to state and hard to do well: turn an expensive, general, cloud-hosted teacher into a cheap, narrow, local student — and *prove* the swap is safe with numbers a skeptical engineer and a cost-conscious executive both accept.

## Honest positioning vs current SOTA

**This is the most important section. Do not overclaim novelty.**

**What already exists.** Knowledge distillation is well-established. The following are all prior art and must be acknowledged, not competed with on the axis of "did I invent distillation":

- **Prometheus / Prometheus 2** — distilled evaluator models (Mistral fine-tunes) that imitate a stronger judge. Direct precedent that task-specific distillation into a smaller open model works.
- **JudgeLM** — distilled LLM-as-judge precedent.
- **Distil-Whisper** — task-specific distillation done well (ASR), a clean example of a smaller model matching a larger one on a bounded task.
- **Alpaca / Vicuna / Self-Instruct** — synthetic-data distillation precedent: fine-tune a base model on a stronger model's generated instruction/response pairs.
- Countless task-specific small models.

Because of all this, **"I distilled a model" is not, by itself, novel.**

**Where the genuine value is (the opening).** The rare, credible artifact is a **rigorous cost/quality case study on a specific, real task** with three things most demos omit:

1. A **clean task definition** and a **held-out eval** the small model never trained on.
2. A **measured cost curve** — teacher $/1k calls vs the local model's amortized cost — *with latency*.
3. **Honest failure analysis** — where the residual ~5% gap is, and whether it matters for the use case.

**The credible headline.** Not *"small model beats GPT."* Instead:

> *"On [specific task], a distilled 3B model reaches Z% of the frontier teacher's quality at ~1/40th the cost and runs on a single consumer GPU — here's the exact recipe and the eval."*

Executives feel the cost story; engineers respect the eval. The defensibility comes from **discipline and measurement**, not from a novel algorithm.

## Choosing the task (narrow is the point)

Pick **ONE** narrow, high-value task where the ROI is obvious. Good candidates:

- **Structured extraction** from a document type (invoices, contracts, CVs) → JSON.
- **Classification / triage** (support-ticket routing, intent detection, PII flagging).
- **Query rewriting / SQL generation** for a fixed schema.
- **Domain summarization** (e.g., legal clause summaries) with a fixed output format.
- **Tie it to a real client:** a task someone actually pays per-API-call for today. That makes the cost win concrete and the case study sellable.

The narrower the task, the smaller the student can be, the stronger the "runs on a laptop" story, and the more defensible the parity claim.

## Distillation signal — what's actually feasible

The available distillation signal depends entirely on **teacher access**:

- **Hard-label / synthetic-data distillation** — train the student on the teacher's *output text* as the target. Works with an **API-only** teacher.
- **Sequence-level KD (+ rationale / CoT traces)** — richer supervision: include the teacher's reasoning traces alongside the answer. Still API-compatible.
- **Logit / soft-label KD (KL on token distributions)** — requires an **open-weights teacher** because you need the teacher's token probability distributions, which API-only providers don't expose.

> With API-only frontier teachers you can't get logits — use **synthetic-data / sequence-level distillation**. Soft-label KL distillation requires an open-weights teacher.

## Why it's rare / defensible (expanded)

- **The eval is the moat.** Anyone can fine-tune a small model on scraped teacher outputs. Almost nobody publishes a **human-verified, leak-free held-out test** and reports honest teacher-agreement on *fresh* inputs. That rigor is the differentiator.
- **The cost model is the sell.** A believable break-even analysis (amortized GPU + electricity vs teacher $/1k, plus latency and data-egress/privacy) is what turns a demo into a business case.
- **Failure analysis builds trust.** Categorizing the residual gap and stating clearly which cases still need the teacher (and optionally routing hard cases up) signals maturity, not weakness.

## Success criteria — the single headline metric + constraint

The project is judged on **one** headline claim, stated as a metric with an explicit constraint:

> **On the chosen narrow task, the distilled 1–3B student reaches ≥ 95% of the frontier teacher's quality on the pre-committed task metric M — measured on a human-verified held-out test set the student never trained on — at roughly 1/40th the per-request cost, running on a single RTX 5080 (16GB).**

Concretely, "done and successful" means **all** of the following hold:

1. **Quality bar met.** Student score on metric M ≥ 95% of the teacher's score on the gold test set (the exact bar and metric are committed in Phase 0, before any data is generated).
2. **Cost win quantified.** A cost table shows teacher $/1k requests vs student amortized $/1k (GPU + electricity), with the break-even request volume above which local wins. The illustrative target is on the order of a **40×** per-request reduction (see [05-evaluation-metrics.md](05-evaluation-metrics.md)).
3. **Latency win measured.** p50/p95 latency reported for both; the local student's p95 is materially lower (illustrative: teacher ~2.1s vs student ~180ms).
4. **Runs locally.** The student runs on the single consumer GPU and via a one-command local invocation (`ollama run your-model`).
5. **Failure analysis published.** The residual gap is categorized, with an explicit statement of which cases still require the teacher.

> All specific figures (95%, 40×, 2.1s, 180ms, 96%) are **illustrative placeholders** from the SPEC. The *contract* is: commit the bar up front, then measure honestly against it.

## Related docs

- Architecture and flow: [02-architecture.md](02-architecture.md)
- What "done" means as requirements: [03-requirements.md](03-requirements.md)
- How the win is proven: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- Full pitfalls: [08-risks-pitfalls.md](08-risks-pitfalls.md)
