# Task-Distilled Small Model — Frontier Quality, Tiny-Model Cost

> Take one narrow, high-value capability, distill a frontier model's behavior on it into a 1–3B model you can run locally, and prove it matches the teacher at a fraction of the cost — *"95% of the quality, 40× cheaper, runs on a laptop."*

## Honest positioning (read this first)

Knowledge distillation is well-established — Prometheus, JudgeLM, Distil-Whisper, and Alpaca/Vicuna-style synthetic-data fine-tunes all exist, so *"I distilled a model"* is not novel by itself. **The genuine, rare artifact is a rigorous cost/quality case study on one specific, real task**: a clean task definition, a human-verified held-out eval the student never trained on, a measured teacher-vs-student cost curve with latency, and honest failure analysis of the residual gap. This repo builds exactly that pipeline and ships the receipts.

## Headline goal

> **On one narrow task, a distilled 1–3B student reaches ≥ 95% of the frontier teacher's quality on the chosen task metric M, measured on a human-verified held-out test set, at roughly 1/40th the per-request cost, running on a single consumer GPU (RTX 5080, 16GB).**

The claim is **not** "small model beats GPT." It is a *measured* win on a *named* axis: quality parity within a stated tolerance, at a quantified cost and latency advantage, with the exact recipe and eval published.

"Done and successful" means **all** of the following hold (see [docs/01-overview.md](docs/01-overview.md)):

1. **Quality bar met** — student score on metric M is ≥ 95% of the teacher's on the human-verified gold test set (bar + metric committed in Phase 0, before any data is generated).
2. **Cost win quantified** — a cost table shows teacher $/1k vs student amortized $/1k, plus the break-even request volume (illustrative target ≈ 40× cheaper per request).
3. **Latency win measured** — p50/p95 reported for both; the local student's p95 is materially lower (illustrative: teacher ~2.1s vs student ~180ms).
4. **Runs locally** — one-command invocation (`ollama run your-model`) on the single consumer GPU.
5. **Failure analysis published** — the residual gap is categorized, stating which cases still need the teacher.

> All figures (95%, 40×, 2.1s, 180ms) are **illustrative placeholders** from the SPEC. The contract is: commit the bar up front, then measure honestly against it.

## Why it's rare / defensible

Most "small model" demos skip the eval and the cost math. A disciplined **teacher → student → measured** pipeline — with a held-out gold set, a real cost model (amortized GPU + electricity, not "it's free locally"), and categorized failure analysis — is exactly the LLMOps competence enterprises pay for.

## How it works (pipeline at a glance)

The system is a closed loop, gated on a pre-committed quality bar:

1. **Task + rubric** — define one narrow task, its strict output schema, and metric M.
2. **Teacher generation** — the frontier/open teacher produces `input → gold output` pairs (optionally with rationales).
3. **Quality filtering** — schema-validate (`pydantic`/`jsonschema`), dedup by embedding similarity, consistency-check.
4. **Distillation dataset** — de-duplicated train / dev / test splits (2k–20k clean examples; quality > quantity).
5. **Student fine-tune** — 1–3B full FT (or QLoRA for 7B) on the single RTX 5080 via Unsloth / TRL.
6. **Eval harness** — task metric M + teacher-agreement + cost + latency, on a **human-verified gold set never trained on**.
7. **Quality gate** — below bar → loop back to generation; at/above bar → package and ship.

See [docs/02-architecture.md](docs/02-architecture.md) for the Mermaid diagrams and the deployment/inference path.

## Documentation

Start with the SPEC, then read the docs in order.

- [SPEC.md](SPEC.md) — the authoritative build spec (source of truth for every claim below).
- [docs/01-overview.md](docs/01-overview.md) — Problem → solution, honest positioning vs SOTA, success criteria.
- [docs/02-architecture.md](docs/02-architecture.md) — System architecture, Mermaid diagrams, data/inference flow, key design decisions.
- [docs/03-requirements.md](docs/03-requirements.md) — Functional + non-functional requirements, scope, assumptions, deliverables checklist.
- [docs/04-data-and-datasets.md](docs/04-data-and-datasets.md) — Teacher-generated data plan, filtering, splits, gold test set, licensing.
- [docs/05-evaluation-metrics.md](docs/05-evaluation-metrics.md) — Metric definitions, baselines, the "money table," eval harness, how the win is proven.
- [docs/06-environment-setup.md](docs/06-environment-setup.md) — Full tech stack, install commands, hardware fit, verify-your-install smoke check.
- [docs/07-build-roadmap.md](docs/07-build-roadmap.md) — Phased build plan (Phase 0–4), milestones, training skeleton.
- [docs/08-risks-pitfalls.md](docs/08-risks-pitfalls.md) — Every pitfall expanded (risk / why / mitigation) plus a risk register.
- [docs/09-references.md](docs/09-references.md) — Papers, benchmarks, models, datasets, tools from the SPEC.
- [docs/10-glossary.md](docs/10-glossary.md) — Every domain term, one clear sentence each.

## Tech stack (compact)

**Teacher:** frontier API model or a strong open model (e.g., a 70B) · **Student base:** Qwen2.5-1.5B/3B, Llama-3.2-1B/3B, Phi-3.5-mini, Gemma-2-2B · **Fine-tuning:** Unsloth or TRL `SFTTrainer` + PEFT · **Serving:** Ollama / llama.cpp / vLLM / TGI · **Data:** `datasets`, `pydantic`, `jsonschema` · **Eval:** `lm-evaluation-harness`, promptfoo / Braintrust · **Tracking:** W&B / TensorBoard · **Hardware:** single RTX 5080 (16GB).

## Distillation signal (what's feasible)

With **API-only** frontier teachers you cannot access logits, so use **synthetic-data / sequence-level distillation** — train the student on the teacher's output text, optionally with rationale/CoT traces. **Soft-label (logit KL) distillation requires an open-weights teacher.** Pick the signal based on teacher access.

## Candidate tasks (pick ONE, narrow)

- Structured extraction from a document type (invoices, contracts, CVs) → JSON.
- Classification / triage (support-ticket routing, intent detection, PII flagging).
- Query rewriting / SQL generation for a fixed schema.
- Domain summarization (e.g., legal clause summaries) in a fixed format.
- Ideally, a task a real client already pays per-API-call for (obvious ROI).

## Deliverables (what ships)

- **HF model + card** — recipe, base, data size, eval table.
- **Eval report** (notebook or markdown) — task metric M + teacher-agreement + cost/latency.
- **Cost dashboard** — an interactive page showing the break-even volume ("above N requests/day, local wins").
- **One-command local run** — `ollama run your-model` + a short demo video.
- **Blog post** — "How I got frontier-level [task] at 1/40th the cost on a single GPU."

Full checklist and supporting artifacts: [docs/03-requirements.md](docs/03-requirements.md).

## Repository layout

```
02-task-distilled-small-model/
├── configs/
│   └── default.yaml            # single source of config (task, teacher, filtering, splits, training, cost)
├── src/distil_task/            # importable package (pure-Python core, no heavy deps at import time)
│   ├── config.py               # config load + project-root/path resolution
│   ├── schema.py               # pydantic Invoice contract + money/date/currency normalization
│   ├── prompts.py              # fixed teacher/student extraction prompts + few-shots + seed-gen prompt
│   ├── teacher.py              # provider-pluggable teacher client (Anthropic), caching, retry, cost accounting
│   ├── io_utils.py             # JSONL I/O + robust JSON extraction from raw model text
│   ├── filtering.py            # schema gate, char-ngram Jaccard / embedding dedup, consistency check
│   ├── metrics.py              # field F1 / exact-match / schema-valid-rate / teacher-agreement (money tolerance)
│   └── cost_model.py           # teacher $/1k vs amortized student $/1k + break-even volume
├── scripts/
│   ├── 00_generate_seed_inputs.py    # Phase 1: seed pool (raw + teacher-synthesized)   [laptop, API-only]
│   ├── 01_generate_teacher_labels.py # Phase 1: teacher labels + consistency 2nd pass    [laptop, API-only]
│   ├── 02_filter_and_split.py        # Phase 1: filter → splits + gold template          [laptop]
│   ├── train.py                      # Phase 2: SFT (Unsloth primary, TRL fallback)      [GPU]
│   ├── evaluate.py                   # Phase 3: money table + eval_report.json           [GPU]
│   ├── export_ollama.py              # Phase 4: merge + Modelfile + GGUF instructions     [GPU]
│   └── serve_path_shim.py            # puts serve/ on sys.path for evaluate.py
├── serve/
│   └── infer.py                # constrained-output inference (generate → parse → validate → retry)
├── dashboard/
│   └── app.py                  # Streamlit cost dashboard (break-even chart + money table)
├── tests/                      # pytest: schema, metrics, filtering, cost_model (pure-Python, no GPU)
├── docs/                       # 01–10 design docs (architecture, data, eval, roadmap, risks, …)
├── requirements.txt            # full GPU training/eval stack
├── requirements-laptop.txt     # API-only data-gen + tests (no GPU, tiny install)
├── RUNBOOK.md                  # exact two-machine build sequence (venv → torch cu128 → Phase 0–4)
└── SPEC.md                     # authoritative build spec
```

## Status / next step

**Pipeline implemented; ready to run on the GPU box.** The full teacher→student→eval→package
code path is written and the pure-logic layer is covered by a green test suite:

- **Done:** task + schema + metric committed (Phase 0); data-generation, filtering, splitting,
  training, evaluation, constrained-output serving, Ollama export, and the cost dashboard are all
  coded against `configs/default.yaml`; `pytest tests/` passes (schema / metrics / filtering /
  cost-model), and `python serve/infer.py --demo` exercises the retry loop with no GPU.
- **Not yet produced (require the RTX 5080 + a teacher API key):** the generated distillation
  dataset, the human-verified gold set, the trained student checkpoint, and the filled-in money
  table. These are *runs*, not code — follow [RUNBOOK.md](RUNBOOK.md).

**Next step:** on the laptop, `pip install -r requirements-laptop.txt` and run Phase 1 data
generation (`scripts/00…02`, API-only); hand-verify the gold set; then move to the GPU box for
training (`scripts/train.py`), evaluation (`scripts/evaluate.py`), and packaging
(`scripts/export_ollama.py`). See [RUNBOOK.md](RUNBOOK.md) for the exact commands.
