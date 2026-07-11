# 06 — Environment Setup

> The full tech stack, prerequisites, concrete install commands, the RTX 5080 (16GB) hardware fit, env configuration, and a "verify your install" smoke check.

## Full tech stack (from SPEC §3)

| Layer | Choice | Notes |
|---|---|---|
| **Teacher** | A frontier API model (your choice) or a strong open model (e.g., a 70B) run elsewhere | API-only ⇒ hard-label / sequence-level KD. |
| **Student base** | **Qwen2.5-1.5B/3B**, **Llama-3.2-1B/3B**, **Phi-3.5-mini**, **Gemma-2-2B** | Pick permissive license + strong base for the task. |
| **Fine-tuning** | **Unsloth** (fastest on a single consumer GPU) or **TRL `SFTTrainer`** + **PEFT** | Unsloth gives big VRAM/speed wins for ≤7B on one GPU. |
| **Quantized inference** | **llama.cpp / Ollama**, **vLLM**, **TGI** | Ollama fits an existing local setup. |
| **Data tooling** | **`datasets`**, **`pydantic`** (schema validation), **`jsonschema`** | Validate every teacher output before it enters the set. |
| **Eval** | **`lm-evaluation-harness`**, task-specific scorers, **Braintrust / promptfoo** for A/B | Report teacher-agreement + task metric. |
| **Cost/latency** | Simple logging → a small **Streamlit / HTML** dashboard | $/1k calls (teacher) vs kWh/GPU-amortized (student); p50/p95 latency. |
| **Tracking** | **W&B / TensorBoard** | |

## Prerequisites

- **OS:** Linux or Windows (with WSL2 recommended for CUDA-heavy training).
- **GPU:** NVIDIA RTX 5080 (16GB) or comparable; recent NVIDIA driver + CUDA toolkit compatible with your PyTorch build.
- **Python:** 3.10–3.11 (typical for the current PyTorch / Unsloth / TRL stack).
- **Package/env manager:** `conda`/`mamba` or `venv` + `pip`.
- **Node.js:** only if you use `promptfoo` (npm-distributed) for A/B.
- **Ollama:** installed locally for the one-command serving deliverable.
- **Teacher access:** an API key for the chosen frontier provider, or a box that can run the chosen open teacher.

## Install steps

> Commands below are standard, well-established invocations. Exact CUDA/PyTorch wheels depend on your driver — always match your CUDA version to the PyTorch install selector.

### 1. Create an environment

```bash
# conda / mamba
conda create -n distill python=3.11 -y
conda activate distill

# or: venv
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows PowerShell:  .venv\Scripts\Activate.ps1
```

### 2. Install PyTorch (CUDA build)

```bash
# Pick the wheel matching your CUDA version from the official PyTorch selector.
# Example (CUDA 12.1 build):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install the fine-tuning + data + eval stack

```bash
# Fine-tuning: Unsloth (fast single-GPU) OR TRL + PEFT
pip install unsloth
pip install trl peft transformers accelerate bitsandbytes

# Data tooling
pip install datasets pydantic jsonschema

# Embeddings for dedup (sentence-embedding based near-duplicate filtering)
pip install sentence-transformers

# Eval
pip install lm-eval           # lm-evaluation-harness

# Experiment tracking
pip install wandb tensorboard

# Cost/latency dashboard
pip install streamlit
```

### 4. A/B eval tool (optional, Node)

```bash
# promptfoo is distributed via npm
npm install -g promptfoo
# (Braintrust is an alternative for A/B; use whichever you prefer)
```

### 5. Serving runtime (Ollama)

Install Ollama from its official installer for your OS, then confirm it runs:

```bash
ollama --version
# After you have a GGUF of your student, register + run it:
#   ollama create your-model -f Modelfile
#   ollama run your-model
```

Alternatives for serving: **llama.cpp** (build from source for GGUF inference), **vLLM** (`pip install vllm`), or **TGI** (containerized).

## Hardware fit — RTX 5080 (16GB)

From SPEC §5:

| Student size | Approach | Fits 16GB? |
|---|---|---|
| ≤3B | **Full fine-tune** (bf16, grad checkpointing) | ✅ |
| 7–8B | **QLoRA** (4-bit) | ✅ |
| 13B | QLoRA, tight | ⚠️ possible with care |

For a distillation **case study**, a **1–3B full fine-tune** is the sweet spot: fast, cheap, and the "runs on a laptop" story is stronger the smaller you go. Use **Unsloth** to maximize throughput on the single GPU.

**QLoRA reference config (if going 7B):** 4-bit NF4, double quant, LoRA `r=16–64`, `alpha=16–32`, target all linear layers, gradient checkpointing, paged AdamW 8-bit, bf16 compute.

## Environment configuration

Set the credentials/keys your pipeline needs (never commit secrets):

```bash
# Teacher API access (provider-specific variable name)
export TEACHER_API_KEY="..."          # or the provider's own env var

# Experiment tracking (optional)
export WANDB_API_KEY="..."

# Hugging Face (for pulling bases / pushing the final model)
export HF_TOKEN="..."
```

On Windows PowerShell, use `$env:TEACHER_API_KEY = "..."` instead of `export`.

## Verify your install (smoke check)

Confirm the core pieces import and the GPU is visible before running any real job:

```bash
# 1. GPU + PyTorch
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

# 2. Core libraries import
python -c "import transformers, trl, peft, datasets, pydantic, jsonschema; print('core stack OK')"

# 3. Unsloth import
python -c "import unsloth; print('unsloth OK')"

# 4. Eval harness present
lm-eval --help >/dev/null && echo "lm-eval OK"

# 5. Serving runtime present
ollama --version
```

A tiny end-to-end sanity load (optional, downloads a small base):

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B", max_seq_length=2048, load_in_4bit=False)
print("loaded student base OK")
```

If all five checks pass and the base loads, the environment is ready for Phase 1 (data generation) in [07-build-roadmap.md](07-build-roadmap.md).

## Related docs

- Where each tool is used in the pipeline: [02-architecture.md](02-architecture.md)
- Training skeleton that uses this stack: [07-build-roadmap.md](07-build-roadmap.md)
- Full tool references: [09-references.md](09-references.md)