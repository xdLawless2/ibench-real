#!/usr/bin/env python3
"""
Recalculate and patch cost estimates for a given model.

Uses config/model_prices.json to compute estimated_cost from token_usage
and updates matching runs/*/summary.json files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _coerce_float(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _coerce_price_entry(entry: Any) -> Optional[Tuple[float, float]]:
    if isinstance(entry, dict):
        prompt = _coerce_float(entry.get("input") or entry.get("prompt") or entry.get("prompt_per_million"))
        completion = _coerce_float(entry.get("output") or entry.get("completion") or entry.get("completion_per_million"))
        if prompt is None and completion is None:
            return None
        return (prompt or 0.0, completion or 0.0)
    coerced = _coerce_float(entry)
    if coerced is not None:
        return (coerced, coerced)
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        first = _coerce_float(entry[0])
        second = _coerce_float(entry[1])
        if first is not None or second is not None:
            return (first or 0.0, second or 0.0)
    return None


def normalize_model_id(model: str) -> str:
    m = (model or "").strip().lower()
    if m.startswith("openrouter/"):
        m = m[len("openrouter/") :]
    return m


def resolve_price(model_input: str, prices: Dict[str, Any]) -> Tuple[str, Tuple[float, float]]:
    if model_input in prices:
        entry = _coerce_price_entry(prices[model_input])
        if entry is None:
            raise ValueError(f"Price entry for '{model_input}' is invalid")
        return model_input, entry

    norm_input = normalize_model_id(model_input)
    candidates: List[str] = []
    for key in prices:
        if normalize_model_id(key) == norm_input:
            candidates.append(key)

    if not candidates and "/" not in model_input:
        base = norm_input.split("/")[-1]
        for key in prices:
            if normalize_model_id(key).split("/")[-1] == base:
                candidates.append(key)

    if len(candidates) == 1:
        key = candidates[0]
        entry = _coerce_price_entry(prices[key])
        if entry is None:
            raise ValueError(f"Price entry for '{key}' is invalid")
        return key, entry
    if len(candidates) > 1:
        raise ValueError(
            "Model name is ambiguous; matches multiple price entries: " + ", ".join(sorted(candidates))
        )
    raise ValueError(f"No price entry found for model '{model_input}'")


def extract_usage_counts(usage: Any) -> Tuple[int, int, int]:
    if not isinstance(usage, dict):
        return 0, 0, 0

    def _get_first(d: Dict[str, Any], keys: List[str]) -> int:
        for key in keys:
            val = d.get(key)
            if isinstance(val, int) and val >= 0:
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
        return 0

    prompt = _get_first(usage, ["prompt_tokens", "input_tokens", "input_token_count"])
    completion = _get_first(usage, ["completion_tokens", "output_tokens", "output_token_count"])
    total = _get_first(usage, ["total_tokens", "token_count"])
    if total == 0 and (prompt or completion):
        total = prompt + completion
    return prompt, completion, total


def matches_model(model_input: str, model_from_run: str) -> bool:
    norm_input = normalize_model_id(model_input)
    norm_run = normalize_model_id(model_from_run)
    if norm_input == norm_run:
        return True
    if "/" not in model_input:
        return norm_input.split("/")[-1] == norm_run.split("/")[-1]
    return False


def patch_summary(
    summary_path: Path,
    model_input: str,
    price_key: str,
    price_pair: Tuple[float, float],
    dry_run: bool,
) -> bool:
    try:
        doc = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[skip] Failed to read {summary_path}: {exc}")
        return False

    model = doc.get("model") or doc.get("model_label") or ""
    if not isinstance(model, str) or not matches_model(model_input, model):
        return False

    usage = doc.get("token_usage")
    prompt_tokens, completion_tokens, _ = extract_usage_counts(usage)
    if prompt_tokens == 0 and completion_tokens == 0:
        print(f"[skip] No token usage in {summary_path}")
        return False

    input_price, output_price = price_pair
    estimated_cost = (
        (prompt_tokens / 1_000_000.0) * input_price
        + (completion_tokens / 1_000_000.0) * output_price
    )

    summary = doc.get("summary")
    if not isinstance(summary, dict):
        summary = {}
        doc["summary"] = summary
    summary["estimated_cost"] = estimated_cost

    price_obj = doc.get("price")
    if not isinstance(price_obj, dict):
        price_obj = {}
        doc["price"] = price_obj
    price_obj["input_per_million"] = input_price
    price_obj["output_per_million"] = output_price
    price_obj["estimated_cost"] = estimated_cost
    price_obj["price_source"] = price_key

    if dry_run:
        print(f"[dry-run] {summary_path}: estimated_cost -> {estimated_cost:.6f}")
        return True

    summary_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"[updated] {summary_path}: estimated_cost -> {estimated_cost:.6f}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recalculate cost estimates for a specific model across runs."
    )
    parser.add_argument("model", help="Model name to fix (e.g., openai/gpt-5.2)")
    parser.add_argument("--runs", default=str(ROOT / "runs"), help="Runs directory (default: ./runs)")
    parser.add_argument(
        "--prices",
        default=str(ROOT / "config" / "model_prices.json"),
        help="Path to model_prices.json (default: ./config/model_prices.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files")
    args = parser.parse_args()

    runs_dir = Path(args.runs).resolve()
    prices_path = Path(args.prices).resolve()

    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        return 2
    if not prices_path.exists():
        print(f"Price file not found: {prices_path}")
        return 2

    try:
        prices = json.loads(prices_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to read prices: {exc}")
        return 2

    try:
        price_key, price_pair = resolve_price(args.model, prices)
    except ValueError as exc:
        print(str(exc))
        return 2

    updated = 0
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        if patch_summary(summary_path, args.model, price_key, price_pair, args.dry_run):
            updated += 1

    if updated == 0:
        print("No matching runs found.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
