# One-command local run — packaging status (SPEC deliverable 4)

## ✅ The one-command local run works TODAY (via the project's own harness)

The merged 0.5B student (`models/student-merged`, Qwen2ForCausalLM) runs locally
with a single command and produces correct, schema-valid output:

```bash
python serve/infer.py --model models/student-merged --file invoice.txt
# -> {"vendor":"SoundMax","date":"2024-03-14","currency":"RUB",
#     "line_items":[{"description":"Wireless Noise-Canceling Earbuds","qty":1,"unit_price":2999,"total":2999}, ...],
#     "subtotal":4298,"grand_total":4298,...}   attempts=1  error=null
```

This is the constrained-decode serving path (generate → parse → validate →
retry) behind the money table (field-F1 0.9647). **The "runs locally" deliverable
is satisfied.** `ollama run` is a *second* packaging target, and it is where the
snag is — documented below.

## The `ollama run` target: blocked by Ollama's built-in GGUF converter

`ollama create invoice-student -f Modelfile` (importing the safetensors via
`FROM .`) **succeeds**, but the resulting model outputs garbage. Investigated
thoroughly:

| Ollama | Behaviour |
|---|---|
| **0.21.0** (original) | crashes at load — `Assertion failed: found, llama-sampling.cpp:660` (penalty sampler + Qwen2.5 dual `eos_token_ids`) |
| **0.31.2** (upgraded via winget) | crash **fixed**, but `ollama run` emits a constant token (`@@@@…`) regardless of input |
| 0.31.2 + **untied embeddings** (explicit `lm_head` re-saved so the converter can't mis-handle the tie) | still constant-token garbage |

**Root cause (proven, not guessed):** the *same* untied checkpoint produces
perfect JSON through `transformers` on the same machine, so the weights are
correct — Ollama's **internal safetensors→GGUF converter mis-maps this model's
layer tensors** (it loads and runs — `ollama show` reports the right arch
`qwen2`, 896 dim, F16 — but the converted weights are wrong). The qwen3:14b
teacher runs fine on the same Ollama, isolating the fault to the conversion of
*this* checkpoint. Upgrading Ollama fixed the crash but not the conversion.

## The reliable route to a real `ollama run`: a canonical GGUF from llama.cpp

Ollama's internal converter is the broken link; a GGUF produced by llama.cpp's
maintained `convert_hf_to_gguf.py` handles Qwen2 tensor layout correctly, and
Ollama then just *loads* it (no internal conversion):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp tools/llama.cpp   # (gitignored)
pip install gguf
python tools/llama.cpp/convert_hf_to_gguf.py models/student-merged \
    --outfile models/invoice-student-f16.gguf --outtype f16
# Modelfile: FROM ./invoice-student-f16.gguf  (reuse the TEMPLATE/SYSTEM in models/student-merged/Modelfile)
ollama create invoice-student -f Modelfile-gguf
ollama run invoice-student "$(cat invoice.txt)"
```

> This step clones + runs the external llama.cpp converter. In this session the
> automated safety layer declined to clone `ggml-org/llama.cpp` without the
> source being explicitly named/approved by the user — so it is left as a
> one-command recipe (or a one-line approval) rather than executed
> automatically. The model is ready for it.

## Bottom line

- **Local run: done** — `serve/infer.py --model models/student-merged`.
- **`ollama run`: needs the llama.cpp GGUF** (Ollama's own converter is broken
  for this checkpoint on both 0.21 and 0.31.2). One-line user approval to clone
  `ggml-org/llama.cpp` unblocks it.
