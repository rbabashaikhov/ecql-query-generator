
"""Baseline evaluation for Qwen3-1.7B on ECQL tasks."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ecql_query_generator.validator import validate_ecql


MODEL_ID = "Qwen/Qwen3-1.7B"

SYSTEM_PROMPT = (
    "Переводи запросы пользователей на язык ECQL. "
    "Возвращай только ECQL без объяснений."
)

SQL_PATTERN = re.compile(
    r"\b(SELECT|FROM|JOIN|ORDER\s+BY|GROUP\s+BY|LIMIT)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_ecql(text: str) -> str:
    """Normalize whitespace only; preserve ECQL semantics."""
    return " ".join(text.strip().split())


def evaluate_prediction(prediction: str, reference: str) -> dict:
    normalized_prediction = normalize_ecql(prediction)
    normalized_reference = normalize_ecql(reference)

    return {
        "syntax_valid": validate_ecql(normalized_prediction),
        "exact_match": (
            normalized_prediction == normalized_reference
        ),
        "sql_hallucination": bool(
            SQL_PATTERN.search(normalized_prediction)
        ),
    }


def generate_prediction(model, tokenizer, instruction: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )

    started = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - started

    new_tokens = output[0, inputs["input_ids"].shape[-1]:]

    prediction = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return prediction, elapsed


def run_evaluation(
    data_path: Path,
    output_path: Path,
    limit: int | None,
    threads: int,
) -> None:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    rows = read_jsonl(data_path)

    if limit is not None:
        rows = rows[:limit]

    if not rows:
        raise ValueError("Evaluation dataset is empty")

    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Dataset: {data_path}", flush=True)
    print(f"Examples: {len(rows)}", flush=True)
    print(f"CPU threads: {threads}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    results = []

    for index, row in enumerate(rows, start=1):
        prediction, elapsed = generate_prediction(
            model,
            tokenizer,
            row["input"],
        )

        metrics = evaluate_prediction(
            prediction,
            row["output"],
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

        print(
            f"[{index}/{len(rows)}] "
            f"valid={metrics['syntax_valid']} "
            f"exact={metrics['exact_match']} "
            f"sql={metrics['sql_hallucination']} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

        # Preserve progress even if the process is interrupted.
        write_jsonl(output_path, results)

    count = len(results)

    summary = {
        "model": MODEL_ID,
        "dataset": str(data_path),
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

    summary_path = output_path.with_suffix(".metrics.json")

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== BASELINE METRICS ===", flush=True)
    print(
        json.dumps(summary, ensure_ascii=False, indent=2),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/test.jsonl"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline_test.jsonl"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    run_evaluation(
        data_path=args.data,
        output_path=args.output,
        limit=args.limit,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
