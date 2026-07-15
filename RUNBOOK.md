# RUNBOOK — build the distilled invoice extractor end to end

Operational, copy-pasteable sequence for taking this repo from an empty
checkout to a packaged Ollama model. Two machines are involved:

- **Laptop (no GPU)** — Phase 1 **data generation** is API-only (teacher
  calls + pure-Python filtering) and the **test suite** run here.
- **GPU box (RTX 5080, 16 GB, Blackwell / `sm_120`)** — Phase 2 **training**,
  Phase 3 **eval**, Phase 4 **GGUF export**.

Every path below is relative to the project root (the directory containing
`pyproject.toml`); scripts resolve paths through `distil_task.config` so they
work regardless of your current directory.

---

## 0. Which machine runs what

| Phase | Command(s) | Machine | Deps |
|---|---|---|---|
| 0 — task/bar | (already committed in `configs/default.yaml` + `docs/`) | either | — |
| 1 — data gen + filter | `scripts/00_…`, `scripts/01_…`, `scripts/02_…` | **laptop** (API-only) | `requirements-laptop.txt` |
| 1 — gold verify | hand-edit `data/gold/gold_test.jsonl` | either | — |
| 2 — train student | `scripts/train.py` | **GPU box** | `requirements.txt` |
| 3 — evaluate | `scripts/evaluate.py` | **GPU box** | `requirements.txt` |
| 4 — export to Ollama | `scripts/export_ollama.py` + llama.cpp + Ollama | **GPU box** | `requirements.txt` |
| tests | `pytest tests/` | **laptop** | `requirements-laptop.txt` |
| dashboard | `streamlit run dashboard/app.py` | either | `streamlit` |

---

## 1. Laptop setup (data generation + tests, no GPU)

```bash
python -m venv .venv
# Linux/macOS:            source .venv/bin/activate
# Windows PowerShell:     .venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-laptop.txt      # pydantic, jsonschema, pyyaml, anthropic, pytest
```

Sanity-check the pure-Python core before spending any teacher tokens:

```bash
pytest tests/ -q                            # 100+ pure-logic tests, no heavy deps
python serve/infer.py --demo                # constrained-output retry loop self-check
```

`serve/infer.py --demo` scripts a student whose *first* reply is malformed and
*second* is valid, proving the parse → validate → retry loop recovers a schema
slip with no GPU or model present.

---

## 2. GPU box setup (RTX 5080 / Blackwell `sm_120`)

### 2.1 venv

```bash
python -m venv .venv
source .venv/bin/activate                   # (Windows: .venv\Scripts\Activate.ps1)
python -m pip install -U pip
```

### 2.2 Install PyTorch with a CUDA build that supports `sm_120` FIRST

> **RTX 5080 is Blackwell — compute capability `sm_120`.** Older CUDA wheels
> (cu121/cu123) do **not** contain `sm_120` kernels and will fail at runtime
> with *"no kernel image is available for execution on the device"* even
> though `torch.cuda.is_available()` returns `True`. Install **cu124 or newer;
> cu128 is recommended** because it ships full Blackwell kernels. Install
> torch **before** everything else so pip does not drag in a CPU-only wheel.

```bash
# Recommended for sm_120 (Blackwell):
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Minimum floor if cu128 is unavailable for your driver: cu124
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2.3 Verify CUDA sees the 5080 and the arch is supported

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available — check driver/wheel"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)          # (12, 0) on a 5080
print("device:", name, "capability sm_%d%d" % cap)
archs = torch.cuda.get_arch_list()
print("torch arch list:", archs)
assert "sm_120" in archs or cap[0] >= 12, (
    "this torch build lacks sm_120 kernels — reinstall a cu124+/cu128 wheel"
)
# prove a kernel actually launches on-device (not just that CUDA is visible):
x = torch.randn(1024, 1024, device="cuda")
print("matmul on GPU OK:", float((x @ x).sum()) == float((x @ x).sum()))
print("VRAM total (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY
```

### 2.4 Install the training / eval / serving stack

```bash
# unsloth (fast single-GPU SFT). Installs matching transformers/trl/peft/etc.
pip install unsloth

# the rest of the project stack (torch already installed above)
pip install -r requirements.txt
```

If `unsloth` is problematic on your driver, training still works via the TRL
fallback (`python scripts/train.py --backend trl`) — it uses
transformers + trl + peft, which `requirements.txt` already pins.

Verify the stack imports:

```bash
python -c "import transformers, trl, peft, datasets, pydantic, jsonschema; print('core stack OK')"
python -c "import unsloth; print('unsloth OK')"       # optional
```

---

## 3. Phase 0 — task + quality bar (already locked)

The task, output schema, metric M, and quality bar are committed in
`configs/default.yaml`:

- **Task:** invoice/receipt text → canonical JSON (`src/distil_task/schema.py`).
- **Metric M:** `field_f1` on the human-verified gold set.
- **Bar:** student ≥ `0.95 ×` teacher on M (`task.quality_bar_ratio`).
- **Student base:** `Qwen/Qwen2.5-0.5B-Instruct`, full fine-tune (fits 16 GB easily).
- **Teacher:** `qwen3:14b` via Ollama (provider `local_openai`, `think:false`).

The original plan targeted a 3B student and a paid `claude-sonnet-4-5` teacher;
the as-built config swapped in a free local open-weights teacher and a 0.5B base
(the 3B download was unreliable on this link) — which is exactly why the dollar
cost thesis remains gated on a paid teacher (see `docs/phase-2/`).

Nothing to run here — just confirm you agree with those before generating data.

---

## 4. Phase 1 — generate + filter data (LAPTOP, API-only)

Runs entirely on the laptop with `requirements-laptop.txt`; no GPU needed. All
teacher calls are cached in `data/cache/`, so re-runs are free and resumable.

```bash
# Anthropic key (never commit it):
export ANTHROPIC_API_KEY="sk-ant-..."       # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."

# 4.1 Seed inputs — ingest data/raw/*.txt (if any) + synthesize diverse docs
python scripts/00_generate_seed_inputs.py           # honors generation.n_seeds (3000)
#   quick smoke:  python scripts/00_generate_seed_inputs.py --n 50

# 4.2 Teacher labels every seed (+ a 2nd higher-temp pass for consistency)
python scripts/01_generate_teacher_labels.py        # writes data/labeled/*.jsonl
#   quick smoke:  python scripts/01_generate_teacher_labels.py --limit 50

# 4.3 Filter (schema + consistency + Jaccard/embedding dedup) and split
python scripts/02_filter_and_split.py               # writes data/splits/* + data/gold/*
```

After 4.3 you get `data/splits/{train,dev,test,agreement_pool}.jsonl`, a
filtering report in `reports/filtering_report.json`, and a **gold template**
at `data/gold/gold_test.jsonl` (every record `human_verified: false`).

### 4.4 Hand-verify the gold set (required for the headline number)

Open `data/gold/gold_test.jsonl` and, following
`data/gold/LABELING_GUIDE.md`, correct each `output` and flip
`"human_verified": true`. Records left `false` are **excluded** from the eval.
The student must **never** train on this file.

Move `data/` (or at least `data/splits/`, `data/gold/`, `data/labeled/`) to the
GPU box for training/eval.

---

## 5. Phase 2 — train the student (GPU box)

```bash
# Full run per configs/default.yaml (Qwen2.5-0.5B full FT, 3 epochs):
python scripts/train.py

# Force the transformers/TRL backend instead of unsloth:
python scripts/train.py --backend trl

# 5-step smoke on a tiny model — validates the pipeline anywhere (even CPU):
python scripts/train.py --smoke
```

Output: a checkpoint in `models/student/` plus `training_recipe.json`
(backend, base, hyperparameters, metrics). Tracking goes to W&B by default
(`training.report_to`); set it to `tensorboard` or `none` in the config to
change that.

---

## 6. Phase 3 — evaluate + money table (GPU box)

```bash
# Score the student on the verified gold set + teacher-agreement pool,
# and also score the cached teacher outputs as the reference baseline:
python scripts/evaluate.py --teacher-baseline

# Debug/dev options:
python scripts/evaluate.py --limit 25               # cap items per set
python scripts/evaluate.py --allow-unverified-gold  # dev only — NOT a headline number
```

Writes:
- `reports/eval_report.json` — metric M, teacher-agreement, latency p50/p95,
  throughput, and the full cost block (consumed by the dashboard).
- `reports/money_table.md` — the teacher-vs-student "money table".
- `reports/gold_predictions.jsonl` — per-item predictions for failure analysis.

**Quality gate:** if `meets_bar` is `false`, loop back to Phase 1 (more/better/
more-diverse data) before packaging.

---

## 7. Phase 4 — package for Ollama (GPU box)

### 7.1 Merge + emit the Modelfile and GGUF instructions

```bash
# Merge the student (LoRA -> merged, or full FT -> re-serialized fp16) into
# models/student-merged/, write the Ollama Modelfile, and PRINT the exact
# llama.cpp GGUF-conversion commands:
python scripts/export_ollama.py --llama-cpp /path/to/llama.cpp --quant Q4_K_M

# Laptop-safe: just (re)generate the Modelfile + print instructions, no merge:
python scripts/export_ollama.py --modelfile-only
```

### 7.2 Convert to GGUF (the commands the script prints)

```bash
# 1. merged HF checkpoint -> full-precision GGUF
python /path/to/llama.cpp/convert_hf_to_gguf.py models/student-merged \
    --outfile models/student-merged/distil-invoice-f16.gguf --outtype f16

# 2. quantize (Q4_K_M is a good size/quality point for a 3B)
/path/to/llama.cpp/llama-quantize \
    models/student-merged/distil-invoice-f16.gguf \
    models/student-merged/distil-invoice-q4_k_m.gguf Q4_K_M
```

### 7.3 Register + run with Ollama (the one-liner deliverable)

```bash
cd models/student-merged
ollama create distil-invoice -f Modelfile
ollama run distil-invoice "$(cat ../../data/gold/some_invoice.txt)"
```

The generated `Modelfile` carries the ChatML template + the distilled
`SYSTEM` prompt and pins `temperature 0` with `<|im_end|>` / `<|im_start|>`
stop tokens for clean one-object JSON output.

### 7.4 Programmatic serving with schema enforcement

For production calls, wrap the model in the constrained-output extractor so a
format slip returns a structured error instead of crashing downstream:

```bash
python serve/infer.py --model models/student-merged --file path/to/invoice.txt
```

---

## 8. Cost dashboard (either machine)

```bash
pip install streamlit
streamlit run dashboard/app.py
```

Reads `reports/eval_report.json` and lets you drag requests/day, teacher token
prices, and GPU price/watts/kWh to see the break-even crossover and the
quality-vs-cost table live. Without an eval report it still runs on config
defaults (quality cells show `n/a`).

---

## 9. Human-gold slice (turnkey: recruit → rate → adjudicate → score)

This is the executable version of the Phase 2 **F2 human-gold** lane. It turns
the silver `reports/gold_predictions.jsonl` into a hand-rated, κ-anchored gold
slice with **~10 minutes of human work per rater**. Everything here is
pure-Python + stdlib (no GPU, no API).

### 9.1 Build the rating sheet (active-learning sampler)

```bash
python scripts/gold_sample.py --n 24        # real slice: --n 150 (…to 200)
```

Picks the most *informative* pairs (low field-F1, schema-repair triggered,
exact-match miss), stratified by currency, and writes:
`data/gold/rating_sheet.csv` (blank `label`/`notes` columns) +
`data/gold/rating_instructions.md` (the rubric, decision rule, worked examples)
+ `reports/gold_sample_predictions.jsonl` (per-id signals for the scorer).

### 9.2 Recruit + rate (the only human step)

Give **each of two raters** a copy of `rating_sheet.csv` and
`rating_instructions.md`. They label **independently** — one word per row in
`label` (`correct` / `incorrect`), a short reason in `notes` for `incorrect`.
Save as `data/gold/rater_a.csv` and `data/gold/rater_b.csv`.

### 9.3 Score inter-rater agreement + build the adjudication queue

```bash
python scripts/gold_kappa.py \
    --rater-a data/gold/rater_a.csv --rater-b data/gold/rater_b.csv
```

Prints Cohen's κ (gate: **≥ 0.70**) + raw agreement, and writes the
disagreements to `data/gold/disagreements.csv`. **If κ < 0.70**, tighten
`rating_instructions.md` and re-rate before trusting any number.

### 9.4 Adjudicate → final gold → the defensible number

A third rater (or a consensus pass) fills the `adjudicated` column of
`disagreements.csv`; merge the agreements + adjudicated calls into
`data/gold/adjudicated.csv` (columns `id,label`). Then:

```bash
python scripts/gold_kappa.py \
    --rater-a data/gold/rater_a.csv --rater-b data/gold/rater_b.csv \
    --adjudicated data/gold/adjudicated.csv --json-out reports/gold_kappa.json
```

reports the **human-verified extraction accuracy** — the model-vs-human number
that replaces the silver headline.

### 9.5 Validate the whole loop NOW with a synthetic stand-in

No humans yet? Exercise the pipeline end-to-end with two LLM stand-in raters
(local Ollama `qwen3:14b`, two rubric phrasings/temperatures):

```bash
python scripts/gold_synthetic_raters.py --limit 20     # add --force-fallback if no Ollama
```

Writes `reports/gold_pipeline_demo.md` with the measured synthetic κ and the
model-vs-synthetic-gold accuracy. **These are SYNTHETIC raters (LLM stand-ins)
to validate the pipeline — NOT human gold;** replace with two human CSVs (§9.2)
for the real κ ≥ 0.70 result. Falls back to a deterministic rule-based stand-in
if Ollama is unreachable.

---

## 10. Troubleshooting

- **`no kernel image is available for execution on the device`** — your torch
  wheel lacks `sm_120`; reinstall cu128 (§2.2) and re-run the verify snippet (§2.3).
- **`ANTHROPIC_API_KEY is not set`** — export it before Phase 1 (§4).
- **`unsloth` import errors on the 5080** — train with `--backend trl`.
- **`no usable gold records`** at eval — hand-verify `data/gold/gold_test.jsonl`
  (§4.4) or pass `--allow-unverified-gold` for a non-headline dev run.
- **OOM during 3B full FT** — keep `gradient_checkpointing: true`, lower
  `per_device_train_batch_size` and raise `gradient_accumulation_steps` to keep
  the effective batch size, or switch to QLoRA (`full_finetune: false`).
