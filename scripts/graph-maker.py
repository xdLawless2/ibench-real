#!/usr/bin/env python3
"""Render benchmark graphs from runs/<model>/summary.json files.

Generates charts into a `graphs/` folder by default:
1) Accuracy leaderboard (horizontal bars).
2) Average cost per image vs accuracy (scatter).
3) Total run cost vs accuracy (scatter).

Newly added run folders are highlighted the first time they appear. Previously seen
slugs are stored in a small state file next to the output directory (override with --state).
"""

import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from statistics import median


# Single color per provider for consistent visual grouping
PROVIDER_COLORS: Dict[str, str] = {
    "human": "#222222",
    "random": "#888888",
    "openai": "#1f77b4",      # Blue
    "anthropic": "#ffb347",   # Light orange
    "google": "#2e8b57",      # Green
    "x-ai": "#4a4a4a",        # Dark grey
    "grok": "#4a4a4a",        # Dark grey
    "opensource": "#9b59b6",  # Purple (qwen, mistral, etc.)
}

FALLBACK_COLOR = "#9b59b6"  # Purple for unknown/open-source models
MODEL_PRICE_FILE = Path("config/model_prices.json")


@dataclass
class RunMetric:
    slug: str
    model_id: str
    label: str
    accuracy: float
    avg_cost: Optional[float]
    total_cost: Optional[float]
    total_time_s: Optional[float]
    provider_hint: str
    reasoning_active: bool = False
    explicit_reasoning_effort: Optional[str] = None
    is_baseline: bool = False


def short_model_name(model_id: str) -> str:
    """Strip provider prefixes and return the bare model name."""
    raw = (model_id or "").strip()
    if not raw:
        return raw
    lowered = raw.lower()
    if lowered.startswith("openrouter/"):
        raw = raw[len("openrouter/") :]
    parts = [p for p in raw.replace("\\", "/").split("/") if p]
    return parts[-1] if parts else raw


def _coerce_float(val: object) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _load_price_overrides() -> Dict[str, Tuple[float, float]]:
    path = MODEL_PRICE_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Tuple[float, float]] = {}
    for key, val in raw.items():
        k = str(key).strip().lower()
        if not k:
            continue
        if isinstance(val, dict):
            inp = _coerce_float(val.get("input") or val.get("prompt") or val.get("prompt_per_million"))
            outp = _coerce_float(val.get("output") or val.get("completion") or val.get("output_per_million"))
            if inp is None and outp is None:
                continue
            out[k] = (inp or 0.0, outp or 0.0)
    return out


def _lookup_price_pair(model_id: str, slug: str, overrides: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    mid = (model_id or "").strip()
    short = short_model_name(mid).strip()
    candidates: List[str] = [
        mid.lower(),
        short.lower(),
        slug.lower(),
    ]
    # Common aliasing: "gpt-5-chat" should map to gpt-5 pricing.
    if mid.lower().endswith("/gpt-5-chat"):
        candidates.append("openai/gpt-5")
        candidates.append("gpt-5")
    # Try without provider prefix too.
    if "/" in mid:
        candidates.append(mid.split("/", 1)[1].lower())
    # De-duplicate preserving order.
    seen: Set[str] = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]
    for c in ordered:
        pair = overrides.get(c)
        if pair is not None:
            return pair
    return None


def load_run_metrics(runs_dir: Path) -> List[RunMetric]:
    """Load metrics from all runs/*/summary.json files."""
    items: List[RunMetric] = []
    if not runs_dir.exists():
        return items
    overrides = _load_price_overrides()
    for summary_file in sorted(runs_dir.glob("*/summary.json")):
        try:
            doc = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = doc.get("summary", {}) if isinstance(doc, dict) else {}
        if not isinstance(summary, dict):
            continue
        percent = summary.get("percent_correct")
        if not isinstance(percent, (int, float)):
            continue
        params = doc.get("params", {}) if isinstance(doc, dict) else {}
        reasoning_observed = bool(summary.get("reasoning_observed")) if isinstance(summary, dict) else False
        explicit_reasoning_effort = params.get("reasoning_effort") if isinstance(params, dict) else None
        base_model = doc.get("model") or doc.get("model_label") or summary_file.parent.name

        label_override = doc.get("benchmark_label") if isinstance(doc, dict) else None
        if not isinstance(label_override, str) or not label_override.strip():
            continue
        display = label_override.strip()

        provider_hint = str(doc.get("model") or doc.get("model_label") or base_model)

        n_val = summary.get("n") if isinstance(summary, dict) else None
        if not isinstance(n_val, int):
            continue

        total_time_s = _coerce_float(summary.get("total_time_s")) if isinstance(summary, dict) else None
        token_usage = doc.get("token_usage", {}) if isinstance(doc, dict) else {}
        prompt_tokens = _coerce_float(token_usage.get("prompt_tokens")) if isinstance(token_usage, dict) else None
        completion_tokens = _coerce_float(token_usage.get("completion_tokens")) if isinstance(token_usage, dict) else None
        if prompt_tokens is None or completion_tokens is None:
            continue

        price_pair = _lookup_price_pair(str(base_model), summary_file.parent.name, overrides)
        if price_pair is None:
            continue
        in_p, out_p = price_pair
        cost_total = (prompt_tokens / 1_000_000.0) * in_p + (completion_tokens / 1_000_000.0) * out_p
        avg_cost = cost_total / n_val if n_val > 0 else None

        items.append(
            RunMetric(
                slug=summary_file.parent.name,
                model_id=str(base_model),
                label=display,
                accuracy=float(percent),
                avg_cost=avg_cost,
                total_cost=cost_total,
                total_time_s=total_time_s,
                provider_hint=provider_hint,
                reasoning_active=bool(
                    reasoning_observed
                    or (
                        isinstance(explicit_reasoning_effort, str)
                        and explicit_reasoning_effort.strip().lower() != "none"
                    )
                ),
                explicit_reasoning_effort=(
                    explicit_reasoning_effort.strip().lower()
                    if isinstance(explicit_reasoning_effort, str) and explicit_reasoning_effort.strip()
                    else None
                ),
            )
        )
    return items


def prune_superseded_unset_reasoning_runs(data: List[RunMetric]) -> List[RunMetric]:
    """Drop unset-reasoning runs when the exact same model has an explicit reasoning run."""
    explicit_models = {
        (m.model_id or "").strip().lower()
        for m in data
        if not m.is_baseline and m.explicit_reasoning_effort not in (None, "none")
    }
    if not explicit_models:
        return data

    filtered: List[RunMetric] = []
    for m in data:
        model_key = (m.model_id or "").strip().lower()
        has_unset_reasoning = m.explicit_reasoning_effort is None
        if not m.is_baseline and has_unset_reasoning and model_key in explicit_models:
            continue
        filtered.append(m)
    return filtered


def ensure_baselines(data: List[RunMetric]) -> List[RunMetric]:
    names = {m.label.lower() for m in data}
    if "human" not in names:
        data.append(
            RunMetric(
                slug="__baseline_human__",
                model_id="__baseline_human__",
                label="Human",
                accuracy=100.0,
                avg_cost=None,
                total_cost=None,
                total_time_s=None,
                provider_hint="Human",
                explicit_reasoning_effort=None,
                is_baseline=True,
            )
        )
    if "random guess" not in names and "random" not in names:
        data.append(
            RunMetric(
                slug="__baseline_random__",
                model_id="__baseline_random__",
                label="Random Guess",
                accuracy=10.0,
                avg_cost=None,
                total_cost=None,
                total_time_s=None,
                provider_hint="Random Guess",
                explicit_reasoning_effort=None,
                is_baseline=True,
            )
        )
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


def pick_color(label: str) -> str:
    """Return a single consistent color based on the model's provider."""
    provider = normalize_provider(label)

    # Map to provider color
    if provider in {"human", "random"}:
        return PROVIDER_COLORS[provider]
    if provider.startswith("openai") or provider in {"o3", "o4", "o4-mini"}:
        return PROVIDER_COLORS["openai"]
    if provider.startswith("anthropic") or "claude" in provider:
        return PROVIDER_COLORS["anthropic"]
    if provider.startswith("google") or "gemini" in provider:
        return PROVIDER_COLORS["google"]
    if provider.startswith("x-ai") or "grok" in provider:
        return PROVIDER_COLORS["x-ai"]
    # Open-source models: qwen, mistral, etc.
    if provider.startswith(("qwen", "mistral", "meta", "llama", "deepseek")):
        return PROVIDER_COLORS["opensource"]

    # Default to purple for unknown/open-source models
    return FALLBACK_COLOR


def is_non_reasoning_run(metric: RunMetric) -> bool:
    return (not metric.is_baseline) and (not metric.reasoning_active)


def format_metric_label(metric: RunMetric) -> str:
    label = metric.label
    if is_non_reasoning_run(metric) and not label.endswith("*"):
        label = f"{label}*"
    return label


def render_accuracy_bar(data: List[RunMetric], output: Path, new_slugs: Set[str]) -> None:
    data = ensure_baselines(data)
    if len(data) <= 2 and all(m.label in {"Human", "Random Guess"} for m in data):
        print("No runs found; rendering baseline comparison only.")
    sorted_items = sorted(data, key=lambda x: x.accuracy)
    slugs = [m.slug for m in sorted_items]
    labels = [format_metric_label(m) for m in sorted_items]
    values = [m.accuracy for m in sorted_items]
    provider_hints = [m.provider_hint for m in sorted_items]
    colors = [pick_color(hint) for hint in provider_hints]

    # More horizontal layout: wider figure, reduced bar height
    fig, ax = plt.subplots(figsize=(14, max(4, len(labels) * 0.35)))
    bars = ax.barh(labels, values, color=colors, height=0.7)
    ax.set_title("EyeBench-V2", fontsize=14, fontweight="bold")
    ax.set_xlabel("Percent Correct (%)")
    ax.set_xlim(0, 110)  # Extra space for value labels

    # Annotate values and highlight new runs.
    highlight_color = "#f59e0b"  # amber/gold
    for bar, val, slug in zip(bars, values, slugs):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=8)
        if slug in new_slugs:
            bar.set_edgecolor(highlight_color)
            bar.set_linewidth(2.5)
            bar.set_hatch("//")
            ax.text(
                min(val + 5, 108),
                bar.get_y() + bar.get_height() / 2,
                "NEW",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=highlight_color,
            )

    # Build legend for providers actually used in the chart
    from matplotlib.patches import Patch
    legend_items = {
        "OpenAI": PROVIDER_COLORS["openai"],
        "Anthropic": PROVIDER_COLORS["anthropic"],
        "Google": PROVIDER_COLORS["google"],
        "X-AI / Grok": PROVIDER_COLORS["x-ai"],
        "Open Source": PROVIDER_COLORS["opensource"],
        "Human": PROVIDER_COLORS["human"],
        "Random": PROVIDER_COLORS["random"],
    }
    # Only include providers that appear in the chart
    used_colors = set(colors)
    legend_handles = [
        Patch(facecolor=color, label=name)
        for name, color in legend_items.items()
        if color in used_colors
    ]
    if new_slugs:
        legend_handles.append(
            Patch(facecolor="white", edgecolor=highlight_color, hatch="//", label="New run")
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")


def render_scatter(
    data: List[RunMetric],
    x_attr: str,
    x_label: str,
    title: str,
    output: Path,
    new_slugs: Set[str],
    log_x: bool = False,
) -> None:
    points = [m for m in data if not m.is_baseline and getattr(m, x_attr) is not None]
    if log_x:
        points = [m for m in points if (getattr(m, x_attr) or 0) > 0]
    if not points:
        print(f"No data for {title}; skipping.")
        return

    highlight_color = "#f59e0b"
    normal = [m for m in points if m.slug not in new_slugs]
    new_pts = [m for m in points if m.slug in new_slugs]

    fig, ax = plt.subplots(figsize=(10, 6))

    xs_all = [getattr(m, x_attr) for m in points if getattr(m, x_attr) is not None]
    ys_all = [m.accuracy for m in points]
    x_mid = median(xs_all)
    y_mid = median(ys_all)
    y_min, y_max = 0.0, 105.0
    y_mid = 50.0

    if log_x:
        ax.set_xscale("log")

    x_min = min(xs_all)
    x_max = max(xs_all)
    if x_min == x_max:
        if log_x:
            x_min = max(x_min * 0.9, 1e-9)
            x_max = x_max * 1.5
        else:
            x_min -= 1.0
            x_max += 1.0
    else:
        if log_x:
            x_min = max(x_min * 0.9, 1e-9)
            x_max = x_max * 1.5
        else:
            pad = (x_max - x_min) * 0.05
            x_min -= pad
            x_max += pad

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Subtle quadrant background
    quad_alpha = 0.08
    quad_specs = [
        (x_min, y_mid, x_mid - x_min, y_max - y_mid, "#d1fae5"),  # low cost, high perf
        (x_min, y_min, x_mid - x_min, y_mid - y_min, "#fef3c7"),  # low cost, low perf
        (x_mid, y_min, x_max - x_mid, y_mid - y_min, "#fee2e2"),  # high cost, low perf
        (x_mid, y_mid, x_max - x_mid, y_max - y_mid, "#dbeafe"),  # high cost, high perf
    ]
    for x0, y0, w, h, color in quad_specs:
        if w > 0 and h > 0:
            ax.add_patch(
                Rectangle((x0, y0), w, h, facecolor=color, edgecolor="none", alpha=quad_alpha, zorder=0)
            )

    ax.axvline(x_mid, color="#666666", linewidth=0.8, alpha=0.4, zorder=1)
    ax.axhline(y_mid, color="#666666", linewidth=0.8, alpha=0.4, zorder=1)

    # Draw points first
    if normal:
        xs = [getattr(m, x_attr) for m in normal]
        ys = [m.accuracy for m in normal]
        cs = [pick_color(m.provider_hint) for m in normal]
        ax.scatter(xs, ys, c=cs, s=60, alpha=0.9, edgecolors="#111111", linewidths=0.5)

    if new_pts:
        xs = [getattr(m, x_attr) for m in new_pts]
        ys = [m.accuracy for m in new_pts]
        cs = [pick_color(m.provider_hint) for m in new_pts]
        ax.scatter(xs, ys, c=cs, s=90, alpha=1.0, edgecolors=highlight_color, linewidths=2.0)

    def place_labels(points_to_label: List[RunMetric]) -> None:
        for m in points_to_label:
            x_val = getattr(m, x_attr)
            y_val = m.accuracy
            if x_val is None:
                continue
            is_new = m.slug in new_slugs
            label_base = format_metric_label(m)
            label_text = f"{label_base} NEW" if is_new else label_base
            color = highlight_color if is_new else "#111111"
            fontweight = "bold" if is_new else "normal"
            ax.text(x_val, y_val, label_text, fontsize=7, color=color, fontweight=fontweight)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Percent Correct (%)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_yticks(list(range(0, 101, 10)))

    fig.tight_layout()
    place_labels(points)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)


def render_reasoning_time_vs_performance(data: List[RunMetric], output: Path) -> None:
    """Render time vs accuracy for runs that actually used reasoning."""
    points = [
        m for m in data
        if (not m.is_baseline)
        and m.reasoning_active
        and (m.total_time_s is not None)
        and ((m.total_time_s or 0.0) > 0)
    ]
    if not points:
        print("No reasoning-enabled runs found; skipping time vs performance chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [(m.total_time_s or 0.0) / 60.0 for m in points]  # minutes
    ys = [m.accuracy for m in points]
    cs = [pick_color(m.provider_hint) for m in points]

    ax.set_xscale("log")
    x_min = min(xs)
    x_max = max(xs)
    if x_min == x_max:
        x_min = max(x_min * 0.9, 1e-6)
        x_max = x_max * 1.5
    else:
        x_min = max(x_min * 0.9, 1e-6)
        x_max = x_max * 1.5
    ax.set_xlim(x_min, x_max)

    ax.scatter(xs, ys, c=cs, s=90, edgecolors="#111111", linewidths=0.6)
    for m, x, y in zip(points, xs, ys):
        clean_label = format_metric_label(m)
        ax.text(x, y, clean_label, fontsize=8)

    ax.set_title("EyeBench-V2: Run time vs Performance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total run time (minutes)")
    ax.set_ylabel("Percent Correct (%)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_yticks(list(range(0, 101, 10)))
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)


def load_seen_slugs(state_path: Path) -> Set[str]:
    if not state_path.exists():
        return set()
    try:
        doc = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if isinstance(doc, list):
        return {str(x) for x in doc}
    if isinstance(doc, dict):
        data = doc.get("seen_slugs")
        if isinstance(data, list):
            return {str(x) for x in data}
    return set()


def save_seen_slugs(state_path: Path, slugs: Set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seen_slugs": sorted(slugs)}
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark graphs from runs summaries")
    parser.add_argument("--runs", default="runs", help="Directory containing <model>/summary.json folders")
    parser.add_argument(
        "--output-dir",
        default="graphs",
        help="Directory for generated graphs (default: graphs/).",
    )
    parser.add_argument(
        "--output",
        default="benchmark.jpg",
        help="Filename for the accuracy bar chart (placed inside output-dir unless absolute).",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Optional path to state file tracking previously seen runs (default: inside output-dir).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs).resolve()
    output_dir = Path(args.output_dir).resolve()
    bar_output = Path(args.output)
    if not bar_output.is_absolute():
        bar_output = output_dir / bar_output
    bar_output = bar_output.resolve()

    base_dir = bar_output.parent
    cost_output = base_dir / "cost_vs_accuracy.jpg"
    overall_cost_output = base_dir / "overall_cost_vs_accuracy.jpg"
    time_output = base_dir / "reasoning_time_vs_performance.jpg"

    data = load_run_metrics(runs_dir)
    current_slugs = {m.slug for m in data}
    state_path = Path(args.state).resolve() if args.state else (base_dir / ".graph-maker-state.json")
    prev_slugs = load_seen_slugs(state_path)
    new_slugs = current_slugs - prev_slugs
    render_accuracy_bar(data, bar_output, new_slugs=new_slugs)
    render_scatter(
        data,
        x_attr="avg_cost",
        x_label="Average cost per image (USD)",
        title="EyeBench-V2: Cost vs Accuracy",
        output=cost_output,
        new_slugs=new_slugs,
        log_x=True,
    )
    render_scatter(
        data,
        x_attr="total_cost",
        x_label="Total run cost (USD)",
        title="EyeBench-V2: Overall Cost vs Accuracy",
        output=overall_cost_output,
        new_slugs=new_slugs,
        log_x=True,
    )
    render_reasoning_time_vs_performance(data, time_output)
    save_seen_slugs(state_path, current_slugs)


if __name__ == "__main__":
    main()
