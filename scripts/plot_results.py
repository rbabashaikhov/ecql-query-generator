
"""Generate reproducible ECQL experiment charts."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_json(filename):
    return json.loads(
        (RESULTS / filename).read_text(encoding="utf-8")
    )


def plot_training():
    history = load_json("training_history.json")

    train = [r for r in history if "loss" in r and "step" in r]
    eval_rows = [
        r for r in history
        if "eval_loss" in r and "step" in r
    ]

    # Deduplicate validation evaluations by optimizer step.
    validation = {
        r["step"]: r["eval_loss"] for r in eval_rows
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        [r["step"] for r in train],
        [r["loss"] for r in train],
        marker="o",
        markersize=3,
        label="Training loss",
    )

    ax.plot(
        list(validation),
        list(validation.values()),
        marker="s",
        linestyle="--",
        label="Validation loss",
    )

    ax.set_title("Qwen3-1.7B — LoRA Fine-tuning")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "training_loss.png", dpi=180)
    plt.close(fig)


def plot_comparison():
    datasets = ["Test (40)", "Challenge (20)"]
    base = [
        load_json("baseline_test.metrics.json"),
        load_json("baseline_challenge.metrics.json"),
    ]
    lora = [
        load_json("lora_test.metrics.json"),
        load_json("lora_challenge.metrics.json"),
    ]

    metrics = [
        ("syntax_validity", "Syntax validity"),
        ("exact_match_accuracy", "Exact match"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(datasets))
    width = 0.35

    for ax, (metric, title) in zip(axes, metrics):
        base_values = [100 * r[metric] for r in base]
        lora_values = [100 * r[metric] for r in lora]

        left = ax.bar(
            x - width / 2, base_values, width, label="Base"
        )
        right = ax.bar(
            x + width / 2, lora_values, width, label="LoRA"
        )

        ax.bar_label(left, fmt="%.1f%%", padding=3)
        ax.bar_label(right, fmt="%.1f%%", padding=3)

        ax.set_xticks(x, datasets)
        ax.set_ylim(0, 115)
        ax.set_title(title)
        ax.set_ylabel("Percent")
        ax.legend()

    fig.suptitle("Qwen3-1.7B: Base vs LoRA")
    fig.tight_layout()
    fig.savefig(FIGURES / "base_vs_lora.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_training()
    plot_comparison()
    print(f"Charts saved in {FIGURES}")
