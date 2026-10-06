#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
import trl
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--train-file", type=Path, default=Path("data/qasper_generated/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/qasper_generated/val.jsonl"))
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen-rag-lora"))
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--merge-and-save", action="store_true")
    parser.add_argument("--merged-output-dir", type=Path, default=Path("results/qwen-rag-merged"))
    return parser.parse_args()


def is_ampere_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


def get_bnb_config(quantization: str, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if quantization == "none":
        return None
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def print_banner(args: argparse.Namespace, compute_dtype: torch.dtype) -> None:
    print("=" * 80)
    print(" LoRA Fine-Tuning: Faithful, Structured RAG Generation")
    print("=" * 80)
    print(f"Libraries:        transformers={transformers.__version__}, trl={trl.__version__}, peft={peft.__version__}")
    print(f"Base Model:       {args.model_name}")
    print(f"Quantization:     {args.quantization} (Compute dtype: {compute_dtype})")
    print(f"LoRA Config:      r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Batch Config:     batch_size={args.batch_size}, grad_accum={args.grad_accum} (effective={args.batch_size * args.grad_accum})")
    print(f"Learning Rate:    {args.lr}")
    print(f"Context Length:   {args.max_seq_len} tokens")
    print(f"Output Dir:       {args.output_dir}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cap = torch.cuda.get_device_capability()
        print(f"Compute Device:   {gpu_name} ({vram_gb:.1f} GB VRAM, SM capability {cap[0]}.{cap[1]})")
    else:
        print("Compute Device:   CPU")
    print("=" * 80)


def to_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
    msgs = example["messages"]
    return {
        "prompt": msgs[:2],
        "completion": msgs[2:],
    }


def main() -> None:
    args = parse_args()
    use_bf16 = is_ampere_or_newer()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print_banner(args, compute_dtype)

    if not args.train_file.exists():
        raise SystemExit(f"Train file not found: {args.train_file}")
    if not args.val_file.exists():
        raise SystemExit(f"Validation file not found: {args.val_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n[2/5] Loading datasets and mapping to prompt-completion format...")
    raw_datasets = load_dataset("json", data_files={
        "train": str(args.train_file),
        "validation": str(args.val_file)
    })
    
    train_dataset = raw_datasets["train"].map(to_prompt_completion, remove_columns=["messages"])
    val_dataset = raw_datasets["validation"].map(to_prompt_completion, remove_columns=["messages"])

    def n_tokens(ex: dict[str, Any]) -> int:
        text = tokenizer.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False)
        return len(tokenizer.encode(text, add_special_tokens=False))

    for name in ("train", "val"):
        ds = train_dataset if name == "train" else val_dataset
        lens = [n_tokens(ex) for ex in ds]
        keep = [i for i, n in enumerate(lens) if n <= args.max_seq_len]
        print(f"    {name}: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.0f} tokens; "
              f"dropping {len(lens) - len(keep)} examples over {args.max_seq_len}")
        ds = ds.select(keep)
        if name == "train":
            train_dataset = ds
        else:
            val_dataset = ds

    if args.sanity_check:
        print("--> SANITY CHECK MODE: subsampling 32 train / 16 val examples.")
        train_dataset = train_dataset.select(range(min(32, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(16, len(val_dataset))))

    print(f"Training examples:   {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    print(f"\n[3/5] Loading base model ({args.model_name}) with {args.quantization} quantization...")
    bnb_config = get_bnb_config(args.quantization, compute_dtype)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = "auto" if torch.cuda.is_available() else None

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print("\n[4/5] Initializing LoRA adapter...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("\n[5/5] Configuring SFTTrainer...")

    if args.sanity_check:
        eval_steps = 5
        save_steps = 5
        max_steps = 10
        num_train_epochs = 1
    else:
        eval_steps = 50
        save_steps = 50
        max_steps = -1
        num_train_epochs = args.epochs

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_length=args.max_seq_len,
        logging_steps=1 if args.sanity_check else 10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        completion_only_loss=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        seed=args.seed,
        ddp_find_unused_parameters=False if local_rank != -1 else None,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\n--> Verifying completion-only loss masking through data collator...")
    batch = trainer.data_collator([trainer.train_dataset[0]])
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    kept = ids[labels != -100]
    unmasked_text = tokenizer.decode(kept, skip_special_tokens=True).strip()
    print(f"    Tokens in sequence: {len(ids)}, tokens with loss: {len(kept)}")
    print(f"    Text the loss is computed on: {unmasked_text[:150]!r}")
    if not (unmasked_text.startswith("{") and "citations" in unmasked_text and len(kept) < len(ids) // 2):
        raise SystemExit("Masking check FAILED: loss is not restricted to the JSON completion.")
    print("    [PASS] Only assistant JSON trained on.")

    print("\n" + "=" * 80)
    print(f" Starting training run ({'Sanity check: 10 steps' if args.sanity_check else f'{args.epochs} epochs'})...")
    print("=" * 80)
    train_result = trainer.train()

    print(f"\nTraining completed! Saving best adapter checkpoint to {args.output_dir}...")
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(args.output_dir))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\n" + "=" * 80)
    print(f" Training complete. Best adapter saved to: {args.output_dir}")
    print("=" * 80)

    if args.merge_and_save:
        print("\n--> Merging LoRA adapter into base model weights...")
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=compute_dtype,
            device_map="cpu",
        )
        merged_model = peft.PeftModel.from_pretrained(base_model, str(args.output_dir))
        merged_model = merged_model.merge_and_unload()
        args.merged_output_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(args.merged_output_dir))
        tokenizer.save_pretrained(str(args.merged_output_dir))
        print(f"Merged model saved to: {args.merged_output_dir}")


if __name__ == "__main__":
    main()
