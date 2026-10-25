
"""Evaluate fine-tuned Qwen3 LoRA using the baseline protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_baseline import (
    MODEL_ID,
    evaluate_prediction,
    generate_prediction,
    read_jsonl,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"LoRA adapter not found: {args.adapter}")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    rows = read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        parser.error("Evaluation dataset is empty")

    print(f"Base model: {MODEL_ID}", flush=True)
    print(f"LoRA adapter: {args.adapter}", flush=True)
    print(f"Dataset: {args.data}", flush=True)
    print(f"Examples: {len(rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(args.adapter),
        is_trainable=False,
    )
    model.eval()

    results = []

    for index, row in enumerate(rows, start=1):
        prediction, elapsed = generate_prediction(
            model, tokenizer, row["input"]
        )

        metrics = evaluate_prediction(
            prediction, row["output"]
        )

        result = {
            "id": index,
            "input": row["input"],
            "reference": row["output"],
            "prediction": prediction,
            "generation_seconds": round(elapsed, 3),
            **metrics,
        }
        results.append(result)
        write_jsonl(args.output, results)

        print(
            f"[{index}/{len(rows)}] "
            f"valid={metrics['syntax_valid']} "
            f"exact={metrics['exact_match']} "
            f"sql={metrics['sql_hallucination']} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    count = len(results)
    summary = {
        "model": MODEL_ID,
        "adapter": str(args.adapter),
        "dataset": str(args.data),
        "evaluated": count,
        "syntax_validity": sum(
            r["syntax_valid"] for r in results
        ) / count,
        "exact_match_accuracy": sum(
            r["exact_match"] for r in results
        ) / count,
        "sql_hallucination_rate": sum(
            r["sql_hallucination"] for r in results
        ) / count,
        "average_generation_seconds": sum(
            r["generation_seconds"] for r in results
        ) / count,
    }

    summary_path = args.output.with_suffix(".metrics.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== LORA METRICS ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
