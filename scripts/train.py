#!/usr/bin/env python
"""Phase 2 — fine-tune the student (Qwen2.5-3B full FT) on the RTX 5080.

Backend order:
1. **Unsloth** FastLanguageModel (SPEC §6 skeleton) — fastest on one GPU.
2. **TRL SFTTrainer** plain-transformers fallback — used automatically if
   the unsloth import fails (and always used for --smoke on CPU).

Data: data/splits/{train,dev}.jsonl. Each record's (input, output) pair is
rendered through the tokenizer's chat template as
    [system: STUDENT_SYSTEM_PROMPT] [user: <document>...] [assistant: JSON]
and trained with completion-only-style SFT (full-sequence LM loss; the
fixed prompt tokens are identical across the corpus so they act as a
constant prefix).

Usage (GPU box):
    python scripts/train.py                       # full run per config
    python scripts/train.py --backend trl         # force the fallback
    python scripts/train.py --smoke               # tiny model, 5 steps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from distil_task.config import ensure_dir, load_config, resolve_path
from distil_task.io_utils import read_jsonl
from distil_task.prompts import STUDENT_SYSTEM_PROMPT, build_extraction_user_prompt


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

def build_messages(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": build_extraction_user_prompt(record["input"])},
        {
            "role": "assistant",
            "content": json.dumps(record["output"], ensure_ascii=False),
        },
    ]


def load_split_as_text_dataset(path: Path, tokenizer):
    """JSONL split -> HF Dataset with a single 'text' column rendered
    through the tokenizer's chat template."""
    from datasets import Dataset  # noqa: PLC0415

    records = read_jsonl(path)
    texts = [
        tokenizer.apply_chat_template(build_messages(r), tokenize=False, add_generation_prompt=False)
        for r in records
    ]
    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# Model loading (Unsloth primary, TRL/transformers fallback)
# ---------------------------------------------------------------------------

def load_model_unsloth(model_name: str, max_seq_length: int, full_finetune: bool,
                       lora_r: int, lora_alpha: int):
    from unsloth import FastLanguageModel  # noqa: PLC0415  (must import before transformers)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,               # 3B full FT fits 16 GB (SPEC §5)
        full_finetuning=full_finetune,
    )
    if not full_finetune:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    return model, tokenizer, "unsloth"


def load_model_trl(model_name: str, max_seq_length: int, full_finetune: bool,
                   lora_r: int, lora_alpha: int, bf16: bool):
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not full_finetune:
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415

        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    return model, tokenizer, "trl"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", choices=["auto", "unsloth", "trl"], default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model + 5 steps to validate the pipeline end-to-end")
    ap.add_argument("--output-dir", default=None, help="override paths.student_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr = cfg["training"]

    model_name = str(tr["smoke_model"] if args.smoke else tr["base_model"])
    max_seq = int(tr["max_seq_length"])
    full_ft = bool(tr["full_finetune"])
    bf16 = bool(tr["bf16"])
    out_dir = Path(args.output_dir) if args.output_dir else resolve_path(cfg, "student_dir")
    ensure_dir(out_dir)

    # ---- backend selection -------------------------------------------------
    backend = args.backend
    if args.smoke and backend == "auto":
        backend = "trl"  # smoke must run anywhere, incl. CPU
    model = tokenizer = None
    used = None
    if backend in ("auto", "unsloth"):
        try:
            model, tokenizer, used = load_model_unsloth(
                model_name, max_seq, full_ft, int(tr["lora_r"]), int(tr["lora_alpha"])
            )
        except ImportError as e:
            if backend == "unsloth":
                raise
            print(f"[backend] unsloth unavailable ({e}); falling back to TRL")
    if model is None:
        model, tokenizer, used = load_model_trl(
            model_name, max_seq, full_ft, int(tr["lora_r"]), int(tr["lora_alpha"]), bf16
        )
    print(f"[backend] {used}  model={model_name}  full_finetune={full_ft}")

    # ---- data ---------------------------------------------------------------
    splits_dir = resolve_path(cfg, "splits_dir")
    train_ds = load_split_as_text_dataset(splits_dir / "train.jsonl", tokenizer)
    dev_path = splits_dir / "dev.jsonl"
    eval_ds = load_split_as_text_dataset(dev_path, tokenizer) if dev_path.exists() else None
    if args.smoke:
        train_ds = train_ds.select(range(min(16, len(train_ds))))
        eval_ds = eval_ds.select(range(min(8, len(eval_ds)))) if eval_ds else None
    print(f"[data] train={len(train_ds)}  dev={len(eval_ds) if eval_ds else 0}")

    # ---- trainer (SPEC §6 config) -------------------------------------------
    import torch  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    use_bf16 = bf16 and torch.cuda.is_available()
    sft_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=1 if args.smoke else int(tr["per_device_train_batch_size"]),
        gradient_accumulation_steps=1 if args.smoke else int(tr["gradient_accumulation_steps"]),
        learning_rate=float(tr["learning_rate"]),
        num_train_epochs=float(tr["num_train_epochs"]),
        warmup_ratio=float(tr["warmup_ratio"]),
        weight_decay=float(tr["weight_decay"]),
        logging_steps=1 if args.smoke else int(tr["logging_steps"]),
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=bool(tr["gradient_checkpointing"]) and not args.smoke,
        max_length=max_seq,
        dataset_text_field="text",
        report_to=("none" if args.smoke else str(tr["report_to"])),
        save_steps=int(tr["save_steps"]),
        seed=int(cfg["splits"]["seed"]) % (2**31),
    )
    if eval_ds is not None:
        sft_kwargs.update(eval_strategy="steps", eval_steps=1 if args.smoke else int(tr["eval_steps"]))
    if args.smoke:
        sft_kwargs.update(max_steps=5, save_steps=5)
    if str(tr["report_to"]) == "wandb" and not args.smoke:
        import os  # noqa: PLC0415
        os.environ.setdefault("WANDB_PROJECT", str(tr["wandb_project"]))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**sft_kwargs),
    )
    result = trainer.train()
    print(f"[train] done: {result.metrics}")

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "training_recipe.json").write_text(
        json.dumps(
            {
                "backend": used,
                "base_model": model_name,
                "full_finetune": full_ft,
                "smoke": args.smoke,
                "n_train": len(train_ds),
                "n_dev": len(eval_ds) if eval_ds else 0,
                "sft_config": {k: str(v) for k, v in sft_kwargs.items()},
                "metrics": {k: float(v) for k, v in result.metrics.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] student saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
