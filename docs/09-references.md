# 09 — References

> All references from the SPEC — concept papers, distillation precedents, tools, and candidate base models — grouped, with canonical identifiers preserved where the SPEC provides them. Nothing here is fabricated beyond the SPEC or well-known fact.

## Concepts & papers (from SPEC §11)

- **Hinton et al., "Distilling the Knowledge in a Neural Network"** — the foundational soft-label knowledge-distillation paper (KL on the teacher's softened output distribution). Relevant here as the *logit / soft-label KD* baseline, which requires an open-weights teacher.
- **Kim & Rush, "Sequence-Level Knowledge Distillation"** — distillation at the sequence level rather than per-token; the conceptual basis for training the student on the teacher's *output sequences* (with optional rationales).
- **Prometheus / Prometheus 2** — distilled evaluator precedent (Mistral fine-tunes); shows task-specific distillation into a smaller open model works for a bounded capability (evaluation/judging).
- **Alpaca / Vicuna / Self-Instruct** — synthetic-data distillation precedent: fine-tune a base model on a stronger model's generated instruction/response pairs. The direct precedent for the *hard-label / synthetic-data* signal used here.
- **Distil-Whisper** — task-specific distillation done well (speech recognition); a clean example of a smaller model matching a larger one on a bounded task.

> Note: **JudgeLM** is referenced in the overview as a distilled LLM-as-judge precedent (context for "distillation is well-established"); the SPEC's formal reference list names Prometheus/Prometheus 2 as the evaluator-distillation citation.

## Distillation-signal taxonomy (from SPEC §2)

- **Hard-label / synthetic-data distillation** — teacher output text as the target; feasible with **API-only** teachers.
- **Sequence-level KD (+ rationale / CoT traces)** — richer supervision via reasoning traces; API-compatible.
- **Logit / soft-label KD (KL on token distributions)** — requires an **open-weights teacher** (APIs don't expose logits).

## Tools (from SPEC §3 and §11)

**Fine-tuning**
- **Unsloth** — efficient single-GPU fine-tuning (fastest for ≤7B on one consumer GPU). Repo: `github.com/unslothai/unsloth`.
- **TRL `SFTTrainer`** — supervised fine-tuning trainer (Hugging Face TRL).
- **🤗 PEFT** — parameter-efficient fine-tuning, including **QLoRA** (4-bit) adapters.

**Serving / quantized inference**
- **Ollama** — local model runner (fits an existing local setup; the one-command `ollama run` deliverable).
- **llama.cpp** — GGUF quantized inference.
- **vLLM** — high-throughput serving.
- **TGI** — Text Generation Inference (containerized serving).

**Data tooling**
- **`datasets`** — dataset loading/processing (Hugging Face).
- **`pydantic`** — schema validation of teacher/student outputs.
- **`jsonschema`** — JSON-schema validation.
- Embedding-similarity dedup (e.g., a sentence-embedding model) for near-duplicate filtering.

**Evaluation**
- **`lm-evaluation-harness`** — standardized eval scaffolding.
- **promptfoo** — A/B comparison of prompts/models (teacher vs student).
- **Braintrust** — A/B / eval tooling alternative.
- Task-specific scorers (exact-match / F1 / accuracy / schema-valid rate).

**Tracking & dashboards**
- **W&B** (Weights & Biases) — experiment tracking.
- **TensorBoard** — experiment tracking alternative.
- **Streamlit / HTML** — the cost/latency dashboard (break-even volume).

## Candidate base models (from SPEC §3 and §11)

Pick a **permissive license + strong base** for the task:

- **Qwen2.5-1.5B / 3B**
- **Llama-3.2-1B / 3B**
- **Phi-3.5-mini**
- **Gemma-2-2B**

The SPEC's training skeleton uses `unsloth/Qwen2.5-3B` as the example base (3B full FT fits the 16GB RTX 5080).

## Teacher options (from SPEC §3)

- A **frontier API model** (your choice) — implies hard-label / sequence-level KD (no logit access).
- A **strong open model** (e.g., a 70B) run elsewhere — enables logit / soft-label KD if desired.

## Hardware reference (from SPEC §5)

- Target GPU: **NVIDIA RTX 5080 (16GB)**.
- Fit: ≤3B full FT ✅ · 7–8B QLoRA (4-bit) ✅ · 13B QLoRA ⚠️ (tight, with care).

## Related docs

- How these tools slot into the pipeline: [02-architecture.md](02-architecture.md)
- Install commands for the tools above: [06-environment-setup.md](06-environment-setup.md)
- Term definitions: [10-glossary.md](10-glossary.md)