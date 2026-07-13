# 06 — Environment Setup (v2.0)

> The additional dependencies, services, env vars, config edits, and verification steps v2.0 layers on top of the Phase 1 environment. The full base stack (PyTorch cu128 for Blackwell `sm_120`, TRL/Unsloth, datasets, pydantic, streamlit, Ollama) is already documented in [../06-environment-setup.md](../06-environment-setup.md) and [../../RUNBOOK.md](../../RUNBOOK.md) — do that first. This doc is the delta.

## What v2.0 adds, at a glance

| Feature | New requirement | Where |
|---|---|---|
| **F1** paid teacher | `anthropic` SDK (already in `requirements-laptop.txt`) + `ANTHROPIC_API_KEY` + config edit | laptop |
| **F2** human gold | a small IAA script (stdlib/numpy); annotators | either |
| **F3** scale student | offline 1–3B base copies; QLoRA deps (`bitsandbytes`, already pinned) | GPU box |
| **F4** valid GGUF | a pinned llama.cpp checkout (`convert_hf_to_gguf.py`, `llama-quantize`) | GPU box |

No new heavyweight framework is introduced — the paid teacher and cost model are already coded; the only genuinely new external tool is a llama.cpp checkout.

## F1 — Paid frontier teacher

### 1. Dependency (already present)
The `anthropic` package is listed in [`requirements-laptop.txt`](../../requirements-laptop.txt) and imported lazily by `AnthropicTeacher`. Confirm:

```bash
python -c "import anthropic; print('anthropic', anthropic.__version__)"
```

### 2. Credentials
```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```
The client raises a clear `TeacherError` if the key is missing — never commit it.

### 3. Config edit ([`configs/default.yaml`](../../configs/default.yaml))
```yaml
teacher:
  provider: anthropic            # was: local_openai
  model: claude-sonnet-4-5       # was: qwen3:14b  (haiku-4-5 = cheaper, opus-4-6 = top-quality ref)
  temperature: 0.0
  # price_table rows for claude-* already exist — no edit needed
```
Leave the `qwen` rows in the price table; switching back to the local teacher is a one-line change.

### 4. Verify billing works (cheap smoke)
```bash
# Label a handful of seeds against the paid teacher and inspect the cost report:
python scripts/01_generate_teacher_labels.py --limit 5
python -c "import json; r=json.load(open('reports/cost_teacher_labeling.json')); print('usd_per_1k:', r['usd_per_1k_calls'], 'billed:', r['billed_calls'])"
```
Expect `usd_per_1k > 0` (unlike v1.0's $0). The disk cache means re-running the same 5 items re-bills $0.

## F2 — Human-verified gold

No new services. Add a lightweight IAA helper (new script, e.g. `scripts/04_gold_iaa.py`) using only stdlib + numpy (already installed) to compute field-level agreement and Cohen's κ across two labeled copies of `data/gold/gold_test.jsonl`. Annotators work directly in the JSONL per [`data/gold/LABELING_GUIDE.md`](../../data/gold/LABELING_GUIDE.md); no annotation platform is required at this scale.

```bash
# after two annotators produce gold_A.jsonl and gold_B.jsonl:
python scripts/04_gold_iaa.py --a data/gold/gold_A.jsonl --b data/gold/gold_B.jsonl
```

## F3 — Scaling the student

### 1. Stage bases offline (the v1.0 blocker)
Pre-download the 1–3B bases to a local cache so training never depends on the flaky link that forced the 0.5B fallback:

```bash
# On a machine with reliable bandwidth, then copy the cache to the GPU box:
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./bases/qwen2.5-1.5b
huggingface-cli download Qwen/Qwen2.5-3B-Instruct  --local-dir ./bases/qwen2.5-3b
```
Point `training.base_model` at the local path (e.g. `./bases/qwen2.5-1.5b`) instead of the hub id.

### 2. QLoRA deps for 3B (already pinned)
A 3B full FT needs ~36 GB; on the 16 GB card use 4-bit QLoRA. `bitsandbytes` is already in [`requirements.txt`](../../requirements.txt). Set in config:
```yaml
training:
  base_model: ./bases/qwen2.5-3b
  full_finetune: false     # QLoRA for 3B on 16 GB
  lora_r: 16
  lora_alpha: 16
```
1.5B stays `full_finetune: true`.

### 3. Run the ablation grid
Drive cells by config override (no new training code):
```bash
python scripts/train.py            # base_model per config
python scripts/evaluate.py         # score on the human gold set
# repeat per {0.5B, 1.5B, 3B} × {data fraction}, collecting training_recipe.json + eval_report.json
```

## F4 — Valid GGUF + one-command run

### 1. Pin a llama.cpp checkout
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && git checkout <known-good-rev>   # pin to dodge the sampler-crash regression
pip install -r requirements.txt                 # for convert_hf_to_gguf.py
make llama-quantize                             # or cmake build
```

### 2. Convert + quantize (commands `export_ollama.py` prints)
```bash
python scripts/export_ollama.py --llama-cpp /path/to/llama.cpp --quant Q4_K_M
python /path/to/llama.cpp/convert_hf_to_gguf.py models/student-merged \
    --outfile models/student-merged/distil-invoice-f16.gguf --outtype f16
/path/to/llama.cpp/llama-quantize \
    models/student-merged/distil-invoice-f16.gguf \
    models/student-merged/distil-invoice-q4_k_m.gguf Q4_K_M
```

### 3. Verify the sampler loads it (the actual F4 gate)
```bash
# Must NOT abort with: Assertion `found` ... llama-sampling.cpp:660
/path/to/llama.cpp/llama-cli -m models/student-merged/distil-invoice-q4_k_m.gguf -p "test" -n 8
```
If it crashes, re-convert with the pinned llama.cpp `convert_hf_to_gguf.py` (not Ollama's built-in converter) and check tokenizer metadata; only a clean load counts as done.

### 4. Register + run with Ollama
```bash
cd models/student-merged
ollama create distil-invoice -f Modelfile
ollama run distil-invoice "$(cat ../../data/gold/some_invoice.txt)"
```
The `Modelfile` already carries the ChatML template with the **exact training scaffold** (`<document>…</document>\nOutput:`) — feeding raw invoice text without it silently degrades the small student.

## Verify the v2.0 environment (smoke checklist)

```bash
# F1: paid teacher reachable + billing
python -c "import anthropic; print('anthropic ok')"
echo "${ANTHROPIC_API_KEY:+ANTHROPIC_API_KEY set}"

# F3: offline base present
ls ./bases/qwen2.5-1.5b/config.json && echo "1.5B base staged"

# F4: llama.cpp converter present
test -f /path/to/llama.cpp/convert_hf_to_gguf.py && echo "llama.cpp converter ok"

# unchanged core still green
pytest tests/ -q      # 118 pure-Python tests
```

## Related docs
- The base Phase 1 environment (do first): [../06-environment-setup.md](../06-environment-setup.md)
- The end-to-end build sequence: [../../RUNBOOK.md](../../RUNBOOK.md)
- Resources these steps install: [04-data-and-resources.md](04-data-and-resources.md)
- Where each tool is used: [02-architecture.md](02-architecture.md)
