#!/usr/bin/env python3
"""Render a benchmark bar chart from runs/<model>/summary.json files."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


COLOR_PALETTES: Dict[str, List[str]] = {
    "human": ["#222222", "#444444"],
    "random": ["#888888", "#aaaaaa"],
    "google": ["#2e8b57", "#3cb371", "#228b22"],
    "openai": ["#1f77b4", "#4a90e2", "#5dade2"],
    "anthropic": ["#ff8c00", "#ffa94d", "#ffb347"],
    "claude": ["#ff8c00", "#ffa94d", "#ffb347"],
    "qwen": ["#ff66cc", "#ff8fd6", "#ff6f91"],
    "opensource": ["#ff66cc", "#ff8fd6", "#ff6f91"],
    "x-ai": ["#5c5c5c", "#7f7f7f", "#a0a0a0"],
    "grok": ["#5c5c5c", "#7f7f7f", "#a0a0a0"],
}

FALLBACK_COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]


def load_run_summaries(runs_dir: Path) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    if not runs_dir.exists():
        return items
    for summary_file in sorted(runs_dir.glob("*/summary.json")):
        try:
            doc = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = doc.get("summary", {}) if isinstance(doc, dict) else {}
        percent = summary.get("percent_correct")
        if percent is None:
            acc = summary.get("accuracy_exact")
            percent = acc * 100 if isinstance(acc, (int, float)) else None
        if percent is None:
            continue
        params = doc.get("params", {}) if isinstance(doc, dict) else {}
        reasoning_effort = params.get("reasoning_effort")
        base_model = doc.get("model")
        label = doc.get("model_label")
        if not label:
            if base_model and reasoning_effort:
                label = f"{base_model} (reasoning +{reasoning_effort})"
            elif base_model:
                label = base_model
            elif reasoning_effort:
                label = f"{summary_file.parent.name} (reasoning +{reasoning_effort})"
            else:
                label = summary_file.parent.name
        items.append((label, float(percent)))
    return items


def ensure_baselines(data: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    names = {label.lower() for label, _ in data}
    if "human" not in names:
        data.append(("Human", 100.0))
    if "random guess" not in names and "random" not in names:
        data.append(("Random Guess", 16.7))
    return data


def normalize_provider(label: str) -> str:
    lowered = label.lower()
    if "human" in lowered:
        return "human"
    if "random" in lowered:
        return "random"
    if "grok" in lowered:
        return "grok"
    if lowered.startswith("openrouter/"):
        lowered = lowered[len("openrouter/") :]
    parts = [part for part in lowered.replace("\\", "/").split("/") if part]
    if not parts:
        return lowered
    if parts[0] == "openrouter" and len(parts) > 1:
        return parts[1]
    if parts[0] in {"providers", "models"} and len(parts) > 1:
        return parts[1]
    return parts[0]


def pick_color(label: str, provider_counts: Dict[str, int]) -> str:
    provider = normalize_provider(label)
    palette_key = provider
    # Map known aliases
    if provider.startswith("google"):
        palette_key = "google"
    elif provider.startswith("openai") or provider in {"o3", "o4"}:
        palette_key = "openai"
    elif provider.startswith("anthropic") or "claude" in provider:
        palette_key = "anthropic"
    elif provider.startswith("qwen") or "vl" in provider:
        palette_key = "qwen"
    elif provider.startswith("x-ai"):
        palette_key = "x-ai"
    elif provider in {"grok"}:
        palette_key = "grok"
    elif provider in {"human", "random"}:
        palette_key = provider

    palette = COLOR_PALETTES.get(palette_key)
    if palette is None:
        palette = FALLBACK_COLORS
        palette_key = "fallback"

    idx = provider_counts[palette_key] % len(palette)
    provider_counts[palette_key] += 1
    return palette[idx]


def render_chart(data: List[Tuple[str, float]], output: Path) -> None:
    data = ensure_baselines(data)
    if len(data) <= 2 and all(label in {"Human", "Random Guess"} for label, _ in data):
        print("No runs found; rendering baseline comparison only.")
    sorted_items = sorted(data, key=lambda x: x[1])
    labels = [label for label, _ in sorted_items]
    values = [value for _, value in sorted_items]
    provider_counts = defaultdict(int)
    colors = [pick_color(label, provider_counts) for label in labels]

    plt.figure(figsize=(10, max(5, len(labels) * 0.5)))
    bars = plt.barh(labels, values, color=colors)
    plt.title("IBench")
    plt.xlabel("Percent Correct (%)")

    for bar, val in zip(bars, values):
        plt.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark chart from runs summaries")
    parser.add_argument("--runs", default="runs", help="Directory containing <model>/summary.json folders")
    parser.add_argument("--output", default="benchmark.jpg", help="Path for the generated chart")
    args = parser.parse_args()

    runs_dir = Path(args.runs).resolve()
    output_path = Path(args.output).resolve()
    data = load_run_summaries(runs_dir)
    render_chart(data, output_path)


if __name__ == "__main__":
    main()
