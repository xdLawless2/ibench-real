#!/usr/bin/env python3
"""
Reads all runs/*/summary.json and truth.txt, produces web/src/data/benchmark-data.json
and copies test images into web/public/imgs/
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
TRUTH_FILE = ROOT / "truth.txt"
IMGS_SRC = ROOT / "imgs"
WEB_DATA_OUT = ROOT / "web" / "src" / "data" / "benchmark-data.json"
WEB_IMGS_OUT = ROOT / "web" / "public" / "imgs"

PROVIDER_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "qwen": "qwen",
    "mistral": "mistral",
    "x-ai": "xai",
    "moonshotai": "moonshot",
}

MODEL_RELEASE_DATES = {
    "openai/o3":                            "2025-04-16",
    "openai/o4-mini":                       "2025-04-16",
    "openai/gpt-5":                         "2025-08-07",
    "openai/gpt-5-chat":                    "2025-08-07",
    "anthropic/claude-sonnet-4.5":          "2025-09-29",
    "openai/gpt-5.1":                       "2025-11-12",
    "openai/gpt-5.1-codex-max":            "2025-11-19",
    "google/gemini-3-pro-preview":          "2025-11-18",
    "anthropic/claude-opus-4.5":            "2025-11-24",
    "openai/gpt-5.2":                       "2025-12-11",
    "google/gemini-3-flash-preview":        "2025-12-17",
    "openai/gpt-5.2-codex":                "2025-12-18",
    "moonshotai/kimi-k2.5":                "2026-01-27",
    "anthropic/claude-opus-4.6":            "2026-02-05",
    "openai/gpt-5.3-codex":                "2026-02-06",
    "anthropic/claude-sonnet-4.6":          "2026-02-17",
    "google/gemini-3.1-pro-preview":        "2026-02-19",
    "qwen/qwen3.5-122b-a10b":              "2026-02-24",
    "qwen/qwen3.5-flash-02-23":            "2026-02-25",
    "google/gemini-3.1-flash-lite-preview": "2026-03-03",
    "openai/gpt-5.3-chat":                 "2026-03-03",
    "openai/gpt-5.4":                       "2026-03-05",
    "openai/gpt-5.4-pro":                  "2026-03-05",
}


def extract_provider(model_path: str) -> str:
    prefix = model_path.split("/")[0] if "/" in model_path else model_path
    return PROVIDER_MAP.get(prefix, prefix)


def load_truth() -> list[int]:
    with open(TRUTH_FILE) as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_run(summary_path: Path) -> dict | None:
    try:
        with open(summary_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    summary = data.get("summary", {})
    if not summary or summary.get("n", 0) == 0:
        return None

    progress = data.get("progress", {})
    if progress and not progress.get("is_final", True):
        return None

    model = data.get("model", "")
    provider = extract_provider(model)
    reasoning_effort = data.get("params", {}).get("reasoning_effort")
    has_reasoning = summary.get("reasoning_observed", False)

    items = []
    for item in data.get("items", []):
        items.append({
            "index": item.get("index"),
            "truth": item.get("truth"),
            "pred": item.get("pred"),
            "correct": item.get("correct", False),
            "latency": round(item.get("latency_s", 0), 2),
            "promptTokens": item.get("prompt_tokens", 0),
            "completionTokens": item.get("completion_tokens", 0),
            "reasoningTokens": item.get("reasoning_tokens", 0),
        })

    release_date = MODEL_RELEASE_DATES.get(model)

    return {
        "slug": data.get("run_slug", summary_path.parent.name),
        "model": data.get("benchmark_label", model.split("/")[-1]),
        "modelFull": model,
        "provider": provider,
        "reasoning": has_reasoning,
        "reasoningEffort": reasoning_effort,
        "releaseDate": release_date,
        "timestamp": data.get("timestamp", ""),
        "accuracy": round(summary.get("percent_correct", 0), 1),
        "mae": round(summary.get("mae", 0), 2),
        "latencyAvg": round(summary.get("latency_avg_s", 0), 2),
        "latencyP50": round(summary.get("latency_p50_s", 0), 2),
        "latencyP95": round(summary.get("latency_p95_s", 0), 2),
        "totalTime": round(summary.get("total_time_s", 0), 1),
        "cost": round(summary.get("estimated_cost", 0), 2),
        "numCorrect": summary.get("num_correct", 0),
        "failures": summary.get("failures", 0),
        "tokens": {
            "prompt": data.get("token_usage", {}).get("prompt_tokens", 0),
            "completion": data.get("token_usage", {}).get("completion_tokens", 0),
            "reasoning": summary.get("reasoning_tokens_total", 0),
            "total": data.get("token_usage", {}).get("total_tokens", 0),
        },
        "price": {
            "inputPerMillion": data.get("price", {}).get("input_per_million", 0),
            "outputPerMillion": data.get("price", {}).get("output_per_million", 0),
        },
        "items": items,
    }


def copy_images():
    if not IMGS_SRC.exists():
        print(f"Warning: {IMGS_SRC} not found, skipping image copy")
        return
    WEB_IMGS_OUT.mkdir(parents=True, exist_ok=True)
    existing = set(p.name for p in WEB_IMGS_OUT.iterdir())
    copied = 0
    for img in sorted(IMGS_SRC.glob("*.png")):
        if img.name not in existing:
            shutil.copy2(img, WEB_IMGS_OUT / img.name)
            copied += 1
    print(f"Copied {copied} new images to {WEB_IMGS_OUT}")


def main():
    truth = load_truth()
    print(f"Loaded {len(truth)} ground truth values")

    runs = []
    seen_slugs = set()
    for summary_path in sorted(RUNS_DIR.rglob("summary.json")):
        run = load_run(summary_path)
        if run and run["slug"] not in seen_slugs:
            seen_slugs.add(run["slug"])
            runs.append(run)

    runs.sort(key=lambda r: r["accuracy"], reverse=True)
    print(f"Loaded {len(runs)} completed runs")

    best_accuracy = max((r["accuracy"] for r in runs), default=0)
    providers = sorted(set(r["provider"] for r in runs))

    output = {
        "meta": {
            "totalModels": len(runs),
            "totalImages": len(truth),
            "bestAccuracy": best_accuracy,
            "providers": providers,
        },
        "truth": truth,
        "runs": runs,
    }

    WEB_DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(WEB_DATA_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {WEB_DATA_OUT} ({WEB_DATA_OUT.stat().st_size:,} bytes)")

    copy_images()
    print("Done!")


if __name__ == "__main__":
    main()
