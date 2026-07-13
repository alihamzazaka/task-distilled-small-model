# 09 — References (v2.0)

> Papers, benchmarks, datasets, tools, and standards each v2.0 feature relies on, with canonical identifiers/links where available. The v1.0 references (Hinton; Kim & Rush; Prometheus/JudgeLM; Alpaca/Vicuna/Self-Instruct; Distil-Whisper; Unsloth/TRL/PEFT; Ollama/llama.cpp/vLLM/TGI; datasets/pydantic/jsonschema; lm-eval/promptfoo/Braintrust) still apply and are not repeated — see [../09-references.md](../09-references.md). Nothing here is invented beyond the SPEC, this repo, or well-known fact.

## F1 — Paid frontier teacher & cost modeling

**Provider / API**
- **Anthropic Claude API** — the paid teacher. Models used in the config `price_table`: `claude-sonnet-4-5` (default teacher), `claude-haiku-4-5` (cheapest), `claude-opus-4-6` (top-quality reference). Client: `src/distil_task/teacher.py::AnthropicTeacher` (uses the `anthropic` Python SDK, `messages.create`, reads `ANTHROPIC_API_KEY`).
- **Provider Terms of Service** — the provider's usage terms governing training a task-specific tool on model outputs (the `NFR-Comp1` compliance obligation). Confirm and document before a full re-distill.

**Cost model (in-repo)**
- `src/distil_task/cost_model.py` — `TeacherPricing`, `teacher_cost_per_1k`, `StudentCostParams`, `gpu_amortized_usd_per_hour`, `student_cost_per_1k`, `break_even_requests_per_day`, `cost_multiple`, `daily_cost_curves`, `params_from_config`. The authoritative implementation of the money table math.
- `tests/test_cost_model.py` — the unit tests pinning that math.

**Cost accounting concepts**
- **Straight-line amortization** — the GPU capex-over-lifetime model (`$1,100 / 3 yr`) used for the honest student cost.
- **Break-even analysis** — fixed (GPU amortization) vs variable (API per-call) cost crossover; the "above N requests/day, local wins" curve.

## F2 — Human-verified gold & inter-annotator agreement

**Protocol / guide (in-repo)**
- `data/gold/LABELING_GUIDE.md` — the field-by-field human labeling rules and arithmetic-sanity tie-breaks (the adjudication basis).
- `src/distil_task/schema.py` — the `Invoice` contract the labels must satisfy.
- `scripts/03_silver_verify.py` — the v1.0 cross-model (silver) verifier that F2 supersedes with human labels.

**Agreement metrics**
- **Cohen's κ (kappa)** — chance-corrected agreement between two annotators on categorical fields (`currency`, `payment_terms` present/absent). Cohen, J. (1960), "A Coefficient of Agreement for Nominal Scales," *Educational and Psychological Measurement*.
- **Fleiss' κ** — the multi-annotator generalization, if more than two annotators are used.
- **Field-level agreement rate** — the exact/near-match fraction per field, the extraction-native complement to κ.
- **Adjudication / gold-standard construction** — standard annotation practice: independent labeling → disagreement adjudication → authoritative gold. (General NLP annotation methodology; e.g. Artstein & Poesio, "Inter-Coder Agreement for Computational Linguistics," *Computational Linguistics*, 2008.)

## F3 — Scaling the student & capacity ablation

**Base models (offline copies)**
- **Qwen2.5-1.5B-Instruct** — `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0). Full FT fits 16 GB.
- **Qwen2.5-3B-Instruct** — `Qwen/Qwen2.5-3B-Instruct` (Qwen license — check redistribution). QLoRA (4-bit) fits 16 GB; full FT (~36 GB) does not.
- **Qwen2.5-0.5B-Instruct** — `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0), the v1.0 baseline cell.
- Permissive alternates from the SPEC: **Llama-3.2-1B/3B**, **Phi-3.5-mini**, **Gemma-2-2B**.

**Efficient fine-tuning for the ablation**
- **QLoRA** — Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) — 4-bit NF4 + double quant + paged optimizers; the mechanism that fits 3B on 16 GB.
- **LoRA** — Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).
- **Unsloth / TRL `SFTTrainer` / PEFT** — the training backends already in the repo (`scripts/train.py`).

**Scaling-law framing (for reading the ablation)**
- **Chinchilla** — Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) — the capacity-vs-data tradeoff framing behind the capacity×data grid.
- **Kaplan et al.**, "Scaling Laws for Neural Language Models" (2020) — the original model-size/data scaling reference.

## F4 — Valid GGUF, serving & packaging

**Conversion / quantization / serving**
- **llama.cpp** — `github.com/ggerganov/llama.cpp`. Tools: `convert_hf_to_gguf.py` (HF → GGUF) and `llama-quantize` (e.g. `Q4_K_M`, `Q5_K_M`, `Q8_0`). The commands `scripts/export_ollama.py` prints; pin a known-good revision to avoid the sampler regression.
- **GGUF** — the llama.cpp/Ollama quantized model file format.
- **Ollama** — `ollama create -f Modelfile` / `ollama run`; the one-command local-run deliverable.
- **Known issue** — the `Assertion 'found' … llama-sampling.cpp:660` sampler crash for the fine-tuned Qwen2.5 base, recorded in `scripts/export_ollama.py::build_modelfile`; the reason F4 verifies the sampler load, not just the conversion.

**Packaging artifacts (in-repo)**
- `scripts/export_ollama.py` — merge + Modelfile (ChatML template with the `<document>…</document>\nOutput:` scaffold + `STUDENT_SYSTEM_PROMPT`) + GGUF instructions.
- `serve/infer.py` — the constrained-output (parse → validate → retry) serving path to wrap over the GGUF.
- `models/student/README.md` — the auto-generated TRL stub card to be replaced (the "Model Card for student" with the time-machine example).
- **Hugging Face Model Cards** — the model-card standard the real card follows (recipe, base, data size, eval table, license).

## Standards & formats (cross-feature)

- **ISO-4217** — currency codes (`USD`, `EUR`, `PKR`, …) normalized by `schema.py::normalize_currency`.
- **ISO-8601** — date format `YYYY-MM-DD` produced by `schema.py::parse_date_iso`.
- **JSON Schema (2020-12)** — the exported `Invoice` schema (`schema.py::export_json_schema`) used in prompts and validation.

## In-repo artifacts referenced throughout Phase 2

- `configs/default.yaml` — teacher/price_table/training/cost config (the single source of truth).
- `reports/eval_report.json`, `reports/money_table.md`, `reports/cost_teacher_labeling.json`, `reports/filtering_report.json`, `reports/gold_predictions.jsonl` — the measured v1.0 numbers.
- `models/student/training_recipe.json` — the v1.0 training recipe.
- `RUNBOOK.md` — the end-to-end build sequence.

## Related docs
- How these tools slot into the v2.0 pipeline: [02-architecture.md](02-architecture.md)
- Setup/install for them: [06-environment-setup.md](06-environment-setup.md)
- Term definitions: [10-glossary.md](10-glossary.md)
- v1.0 references: [../09-references.md](../09-references.md)
