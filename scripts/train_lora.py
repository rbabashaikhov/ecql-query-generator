
"""Fine-tune Qwen3-1.7B for Russian-to-ECQL with LoRA on CPU."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen3-1.7B"
SYSTEM_PROMPT = (
    "Переводи запросы пользователей на язык ECQL. "
    "Возвращай только ECQL без объяснений."
)
SEED = 42


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(records) != 160:
        raise ValueError(
            f"Expected 160 training records, got {len(records)}"
        )

    required = {"instruction", "input", "output"}
    for row in records:
        if not required.issubset(row):
            raise ValueError(f"Invalid record: {row}")

    return records


def build_dataset(
    records: list[dict],
    tokenizer,
) -> Dataset:
    examples = []

    for row in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input"]},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        examples.append({
            "prompt": prompt,
            "completion": row["output"] + "<|im_end|>",
        })

    return Dataset.from_list(examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--output", default="models/ecql-lora")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...", flush=True)
    records = load_records(Path(args.data))

    # Deterministic validation split from Train only.
    shuffled = records.copy()
    random.Random(SEED).shuffle(shuffled)

    train_records = shuffled[:144]
    val_records = shuffled[144:]

    print(
        f"Train: {len(train_records)} | "
        f"Validation: {len(val_records)}",
        flush=True,
    )

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset(train_records, tokenizer)
    eval_dataset = build_dataset(val_records, tokenizer)

    print("Loading model in BF16...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    config = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
        disable_tqdm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting LoRA training...", flush=True)
    train_result = trainer.train()

    print("Evaluating validation set...", flush=True)
    eval_metrics = trainer.evaluate()

    print("Saving LoRA adapter...", flush=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    history = trainer.state.log_history
    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "model": MODEL_ID,
        "method": "LoRA SFTTrainer",
        "seed": SEED,
        "train_examples": len(train_records),
        "validation_examples": len(val_records),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.grad_accum,
        "train_loss": train_result.training_loss,
        "evaluation": eval_metrics,
    }

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(summary, ensure_ascii=False, indent=2),
        flush=True,
    )
    print(f"Adapter saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
