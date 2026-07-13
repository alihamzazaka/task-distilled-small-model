# Phase 2 / v2.0 — Documentation Suite

> The plan to turn the v1.0 *proof-of-pipeline* into a *proof-of-thesis*: distill from a **paid frontier teacher** to finally show the signature $/1k cost win, upgrade the silver gold set to a **human-verified** one, **scale the student** into the SPEC's 1–3B band with a clean capacity ablation, and ship a **valid GGUF + one-command `ollama run`** with a real demo. This is the next-phase design; anchor every "where we are" statement on the v1.0 baseline that IS built.

> ⚠️ **Planned, not yet built.** Everything in `docs/phase-2/` is a *design and build plan for v2.0*. No v2.0 artifact (paid-teacher dataset, human gold set, scaled student, verified GGUF) exists yet. The numbers that ARE real are the **v1.0 baseline** measured in [`reports/`](../../reports/) and reproduced below. v2.0 targets are labeled **target/planned** and are illustrative until measured — exactly the honesty contract of the Phase 1 docs (see [../01-overview.md](../01-overview.md)).

## What Phase 2 is

v1.0 shipped a *complete, tested teacher → student → measured-eval pipeline* and proved a 0.5B student can imitate a 14B teacher on invoice→JSON extraction at **96.5% of teacher field-F1, 100% schema-valid**. But it was distilled from a **free local** teacher (`qwen3:14b` via Ollama), so the project's entire reason for being — *"frontier quality at 1/40th the cost"* — is **unproven** on the axis that matters: dollars. The gold set is **silver-grade** (cross-model agreement, not human ground truth), the student is **below** the SPEC's 1–3B band (a 0.5B fallback because the 3B download was unreliable), exact-match is a low **54%**, and no **valid llama.cpp GGUF** has been produced (`ollama run` currently falls back to a `FROM .` safetensors import).

Phase 2 closes exactly those four gaps, in dependency order. It does **not** re-open the architecture, the schema, or the metric — those are locked from Phase 0 and stay locked.

## The v1.0 → v2.0 delta (one table)

| Axis | v1.0 (built, measured) | v2.0 (planned target) | Feature |
|---|---|---|---|
| **Teacher** | `qwen3:14b`, **local, $0/1k** — cost thesis unprovable | Paid frontier API (`claude-sonnet-4-5`) priced into the cost model; **measured $/1k win + break-even curve** | **F1** |
| **Cost story** | `cost_multiple = 0` (both free-local); dollar win N/A | Headline `N×` cheaper vs API list price, published break-even volume | **F1** |
| **Gold set** | 60 items, **SILVER** (37 passed cross-model agreement), **0 human-verified** | Human-verified + expanded gold, adjudication protocol, inter-annotator agreement; circularity removed | **F2** |
| **Parity claim** | 96.5% of teacher on a *silver* set | Same bar on a **true held-out human** set, as the SPEC requires | **F2** |
| **Student size** | **0.5B** (`Qwen2.5-0.5B-Instruct`, full FT) — below spec band | **1–3B** (offline base copy) + capacity-vs-data ablation | **F3** |
| **Exact match** | **54.1%** | Quantify how much of the residual gap + low EM closes with size | **F3** |
| **Local run** | Prints GGUF commands; **no valid GGUF** (Ollama converter crashes llama.cpp sampler); `FROM .` fallback | **Valid GGUF** via `convert_hf_to_gguf.py`; `ollama run` loads the student; real quick-start | **F4** |
| **Model card** | Auto-generated TRL stub (a "time-machine" example, not invoices) | Real invoice-extraction card + demo video + blog | **F4** |

*The v1.0 column is measured (see [`reports/eval_report.json`](../../reports/eval_report.json), [`reports/money_table.md`](../../reports/money_table.md)). The v2.0 column is the contract to fill in.*

## The four v2.0 features

1. **Paid frontier teacher (F1)** — re-run distillation (or at least re-price) against a real paid API teacher so the money table finally shows a *dollar* win and a break-even volume. The plumbing already exists: `AnthropicTeacher`, the disk cache, `CostTracker`, the `price_table`, and the whole `cost_model.py` (break-even, `cost_multiple`, `daily_cost_curves`) are built and unit-tested — v1.0 simply never pointed them at a billed provider.
2. **Human-verified gold set (F2)** — turn silver into gold: hand-verify the 60-item set and expand it, with a written adjudication protocol and a reported inter-annotator agreement, removing the teacher-vs-model circularity so the ≥95% parity number is measured against true ground truth.
3. **Scale the student (F3)** — move from the 0.5B fallback up into the SPEC's **1–3B** band using an offline base copy, and run a controlled **capacity × data** ablation that quantifies how much of the residual field-F1 gap and the low 54% exact-match closes with model size vs more data.
4. **One-command local run (F4)** — produce a *valid* GGUF via llama.cpp's `convert_hf_to_gguf.py` so `ollama run` actually loads the distilled student, replace the stub model card with a real invoice quick-start, and publish the demo video + blog that Phase 1 listed as deliverables.

## Read the docs in order

| # | Doc | What it covers |
|---|---|---|
| — | [README.md](README.md) | This index: what Phase 2 is, the delta table, the planned-not-built banner |
| 01 | [01-overview.md](01-overview.md) | v2.0 vision, motivation from the v1.0 gaps, goals, non-goals, headline success criteria |
| 02 | [02-architecture.md](02-architecture.md) | Architectural additions v2.0 makes to the v1.0 pipeline; components, flows, Mermaid |
| 03 | [03-requirements.md](03-requirements.md) | Functional + non-functional requirements per feature; in/out of scope; deliverables checklist |
| 04 | [04-data-and-resources.md](04-data-and-resources.md) | New datasets, models, API keys, hardware, licensing each feature needs |
| 05 | [05-evaluation-metrics.md](05-evaluation-metrics.md) | How each feature's success is measured; new metrics; target numbers vs v1.0; the money table |
| 06 | [06-environment-setup.md](06-environment-setup.md) | New deps, services, env vars, and setup steps v2.0 adds on top of Phase 1 |
| 07 | [07-build-roadmap.md](07-build-roadmap.md) | Milestones M1–M8 with Definition of Done, effort/impact, sequencing, dependencies |
| 08 | [08-risks-pitfalls.md](08-risks-pitfalls.md) | Risks + mitigations per feature; a risk register |
| 09 | [09-references.md](09-references.md) | Papers, benchmarks, datasets, tools, standards per feature |
| 10 | [10-glossary.md](10-glossary.md) | New terms v2.0 introduces |

## Relationship to the Phase 1 docs

Phase 2 is **additive**. It reuses the Phase 1 architecture, schema, metric, and roadmap wholesale and only layers new work on top. Where a Phase 1 doc already says everything ("what is the invoice schema", "why generate data with the teacher"), the Phase 2 doc points back rather than repeating:

- Baseline architecture and pipeline: [../02-architecture.md](../02-architecture.md)
- The locked Phase 0 task/metric/bar: [../01-overview.md](../01-overview.md), [`configs/default.yaml`](../../configs/default.yaml)
- v1.0 data plan and gold-set discipline: [../04-data-and-datasets.md](../04-data-and-datasets.md)
- v1.0 metric definitions and the money table: [../05-evaluation-metrics.md](../05-evaluation-metrics.md)
- The end-to-end build sequence that produced v1.0: [../../RUNBOOK.md](../../RUNBOOK.md)

## Success in one sentence

> **v2.0 is "done" when the money table shows a real dollar win against a paid frontier teacher, the ≥95% parity bar is met on a human-verified gold set, a student in the 1–3B band with a documented capacity ablation beats the 0.5B baseline, and `ollama run distil-invoice` loads a valid GGUF and extracts a real invoice — with every number honestly measured and its residual gap analyzed.**
