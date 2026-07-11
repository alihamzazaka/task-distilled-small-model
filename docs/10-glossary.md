# 10 — Glossary

> Every domain term relevant to this task-distillation project, one clear sentence each.

## Distillation core

- **Knowledge distillation** — Training a smaller "student" model to reproduce the behavior of a larger, stronger "teacher" model on a task.
- **Teacher** — The frontier (or strong open) model whose behavior is being copied; here it also generates the training data.
- **Student** — The small (1–3B) model being trained to imitate the teacher and deployed locally.
- **Task distillation** — Distillation focused on one narrow, well-defined capability rather than general ability.
- **Hard-label / synthetic-data distillation** — Distillation where the student trains on the teacher's *output text* as the target; the only option when the teacher is API-only.
- **Sequence-level KD** — Distillation that supervises whole output sequences (optionally with rationales), following Kim & Rush.
- **Logit / soft-label KD** — Distillation that matches the teacher's full token probability distribution (KL divergence on softened logits); requires an open-weights teacher.
- **Soft labels** — The teacher's probability distribution over tokens/classes, used as a richer training target than a single hard answer.
- **Rationale / CoT (chain-of-thought) traces** — Step-by-step reasoning the teacher produces alongside its answer, which the student can be trained to imitate for richer supervision.
- **Teacher-agreement %** — The fraction of a fresh, unseen input pool on which the student's output matches the teacher's, used to measure generalization.

## Data & evaluation

- **Distillation dataset** — The teacher-generated, filtered corpus (train/dev/test) used to fine-tune the student.
- **Seed inputs** — Realistic task inputs (real, public, or teacher-invented) fed to the teacher to produce labeled pairs.
- **Held-out set** — Data withheld from training so it can measure true generalization.
- **Gold test set** — The human-verified, never-trained-on ground-truth test set that anchors the headline quality claim.
- **Ecological validity** — The degree to which the test set reflects the real production input distribution rather than only easy cases.
- **Train/test leakage** — Contamination where training items (or near-duplicates) appear in the test set, falsely inflating scores.
- **Dedup (deduplication)** — Removing near-duplicate examples (here via embedding similarity) within and across splits.
- **Schema validation** — Checking that an output conforms to a defined structure using `pydantic` / `jsonschema`.
- **Consistency check** — Keeping only teacher outputs that agree across repeated runs or pass a second-model checker.
- **Task-native metric (M)** — The primary quality metric chosen to fit the task (e.g., exact-match, F1, accuracy).
- **Exact match** — The fraction of predictions identical to the gold answer.
- **F1** — The harmonic mean of precision and recall, used for partial-credit overlap scoring.
- **Accuracy** — The fraction of predictions that are correct (typical for classification).
- **Schema-valid rate** — The fraction of student outputs that parse against the task schema; a robustness signal.
- **Quality bar** — The pre-committed threshold the student must clear (e.g., ≥95% of the teacher on M).
- **Quality gate** — The decision point that loops back for more data or proceeds to packaging based on the bar.
- **Failure analysis** — Categorizing the residual student-vs-teacher gap and identifying which cases still need the teacher.

## Cost, latency & serving

- **$/1k requests** — Cost per one thousand task requests, compared between the teacher (API price) and student (amortized).
- **Amortized GPU cost** — The GPU's hardware cost spread across the requests it serves, included so the cost win is honest.
- **Break-even volume** — The request rate above which running the local student is cheaper than calling the teacher.
- **p50 / p95 latency** — The 50th- and 95th-percentile per-request wall-clock times; p95 captures the tail users feel.
- **Data egress** — Data leaving the organization to a third-party API; avoided by local inference (the privacy win).
- **Constrained decoding** — Restricting generation so outputs conform to a schema/grammar, preventing malformed results.
- **Quantized inference** — Running a model with reduced-precision weights (e.g., 4-bit GGUF) for cheaper, faster local serving.
- **GGUF** — The quantized model file format used by llama.cpp / Ollama for local inference.

## Training & hardware

- **Full fine-tune (Full FT)** — Updating all of the student's weights during training; the preferred approach for ≤3B on 16GB.
- **LoRA (Low-Rank Adaptation)** — Fine-tuning by training small low-rank adapter matrices instead of all weights.
- **QLoRA** — LoRA applied on top of a 4-bit quantized base model, enabling larger (7–8B) students on limited VRAM.
- **NF4 (4-bit NormalFloat)** — The 4-bit quantization data type used in QLoRA.
- **Double quantization** — Quantizing the quantization constants themselves to save additional memory in QLoRA.
- **LoRA rank (r)** — The dimensionality of the LoRA adapter matrices; higher `r` means more capacity (e.g., 16–64).
- **LoRA alpha** — The scaling factor applied to LoRA updates (e.g., 16–32).
- **Gradient checkpointing** — Trading compute for memory by recomputing activations during backprop to fit larger models.
- **Paged AdamW 8-bit** — A memory-efficient 8-bit optimizer that pages state to host memory to avoid OOM.
- **bf16 (bfloat16)** — A 16-bit floating-point compute format used for efficient training.
- **SFT (Supervised Fine-Tuning)** — Training on labeled input→output pairs, here via TRL `SFTTrainer` or Unsloth.
- **VRAM** — GPU memory; the binding constraint for training on the single RTX 5080 (16GB).
- **RTX 5080 (16GB)** — The single consumer GPU targeted for all training and inference in this project.

## Tools & ecosystem

- **Unsloth** — A library for fast, memory-efficient single-GPU fine-tuning of ≤7B models.
- **TRL** — Hugging Face's Transformer Reinforcement Learning library, providing `SFTTrainer`.
- **PEFT** — Hugging Face's parameter-efficient fine-tuning library (LoRA/QLoRA).
- **Ollama** — A local runner for quantized models offering a one-command `ollama run` experience.
- **llama.cpp** — A C/C++ engine for running quantized (GGUF) models on CPU/GPU.
- **vLLM** — A high-throughput LLM serving engine.
- **TGI (Text Generation Inference)** — Hugging Face's containerized model-serving stack.
- **lm-evaluation-harness** — A standardized framework for running LLM evaluations.
- **promptfoo / Braintrust** — Tools for A/B comparing prompts/models (teacher vs student).
- **W&B / TensorBoard** — Experiment-tracking tools for training runs.
- **pydantic / jsonschema** — Libraries for validating structured outputs against a schema.
- **datasets** — Hugging Face's dataset loading/processing library.

## Compliance

- **ToS (Terms of Service)** — The teacher provider's usage terms, which must permit training on its outputs for this use case.
- **Compliance stance** — The documented position on how the project stays within the teacher provider's terms (framing it as an internal/task-specific tool).

## Related docs

- Terms in context of the pipeline: [02-architecture.md](02-architecture.md)
- Metric definitions in depth: [05-evaluation-metrics.md](05-evaluation-metrics.md)
- Source references for these terms: [09-references.md](09-references.md)