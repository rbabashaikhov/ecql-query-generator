
"""Measure real Qwen3-1.7B LoRA training performance on CPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-1.7B"
SYSTEM_PROMPT = (
    "Переводи запросы пользователей на язык ECQL. "
    "Возвращай только ECQL без объяснений."
)


def load_examples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        examples = [json.loads(line) for line in file if line.strip()]
    if not examples:
        raise ValueError(f"Dataset is empty: {path}")
    return examples


def encode_example(
    tokenizer,
    example: dict,
    max_length: int,
) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
    ]

    # Same non-thinking prompt format as baseline evaluation.
    prompt_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
)

    prompt_ids = prompt_encoding["input_ids"]

    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.squeeze(0).tolist()

    # ECQL completion plus Qwen chat end-of-message token.
    completion_ids = tokenizer.encode(
        example["output"],
        add_special_tokens=False,
    )
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    completion_ids.append(end_id)

    if len(prompt_ids) + len(completion_ids) > max_length:
        raise ValueError(
            f"Example exceeds max_length={max_length}: "
            f"prompt={len(prompt_ids)}, "
            f"completion={len(completion_ids)}"
        )

    input_ids = prompt_ids + completion_ids

    # Calculate loss only on the assistant's ECQL response.
    labels = [-100] * len(prompt_ids) + completion_ids

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones(
            (1, len(input_ids)), dtype=torch.long
        ),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--output",
        default="results/cpu_benchmark.json",
    )
    args = parser.parse_args()

    if args.steps < 1 or args.threads < 1:
        parser.error("--steps and --threads must be positive")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)

    process = psutil.Process(os.getpid())
    examples = load_examples(Path(args.data))

    print(f"Model: {MODEL_ID}", flush=True)
    print(f"CPU threads: {args.threads}", flush=True)
    print(f"BF16 supported: {torch.backends.cpu.is_bf16_supported() if hasattr(torch.backends.cpu, 'is_bf16_supported') else 'not reported'}", flush=True)
    print(f"Training steps: {args.steps}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading base model in BF16...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to("cpu")

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.train()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=2e-4,
    )

    measurements = []
    started = time.perf_counter()

    for step in range(args.steps):
        example = examples[step % len(examples)]
        batch = encode_example(
            tokenizer, example, args.max_length
        )

        optimizer.zero_grad(set_to_none=True)

        step_started = time.perf_counter()

        output = model(**batch)
        loss = output.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step + 1}: {loss.item()}"
            )

        loss.backward()
        optimizer.step()

        seconds = time.perf_counter() - step_started
        ram_gib = process.memory_info().rss / (1024**3)

        measurement = {
            "step": step + 1,
            "loss": round(loss.item(), 6),
            "seconds": round(seconds, 3),
            "process_ram_gib": round(ram_gib, 3),
            "tokens": int(batch["input_ids"].shape[1]),
        }
        measurements.append(measurement)

        print(
            f"Step {step + 1}/{args.steps} | "
            f"loss={measurement['loss']:.4f} | "
            f"time={seconds:.2f}s | "
            f"RAM={ram_gib:.2f} GiB | "
            f"tokens={measurement['tokens']}",
            flush=True,
        )

    average_seconds = sum(
        m["seconds"] for m in measurements
    ) / len(measurements)

    result = {
        "model": MODEL_ID,
        "training": "LoRA CPU BF16",
        "threads": args.threads,
        "steps": args.steps,
        "lora_rank": 8,
        "lora_alpha": 16,
        "max_length": args.max_length,
        "batch_size": 1,
        "average_step_seconds": round(average_seconds, 3),
        "total_elapsed_seconds": round(
            time.perf_counter() - started, 3
        ),
        "measurements": measurements,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "\nBenchmark saved to:", output_path,
        "\nAverage step:", round(average_seconds, 2), "seconds",
        flush=True,
    )


if __name__ == "__main__":
    main()
