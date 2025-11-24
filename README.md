# ibench-real

OpenRouter-powered micro-benchmark for counting line-intersection points in 100 synthetic images.

## Features
- Async evaluation of a numbered image set (`./imgs/1.png` … `./imgs/100.png`; auto-detects `N`).
- Ground-truth comparison via `truth.txt` with accuracy, MAE, latency, and throughput metrics.
- Automatic per-model run archives under `./runs/<model_slug>/summary.json` containing per-item outcomes, token usage, and estimated costs.
- Optional structured logging (console + file) capturing raw model outputs and reasoning metadata.
- Queue multiple models sequentially with `--models`, separated by a configurable delay, and optionally give per-model reasoning overrides.

## Requirements
- Python 3.9+
- `aiohttp` (install via `pip install aiohttp`)
- OpenRouter API key exposed via `OPENROUTER_API_KEY` (optionally set `OPENROUTER_SITE_URL`/`OPENROUTER_APP_TITLE` for headers).

## Quick Start
```bash
pip install aiohttp
export OPENROUTER_API_KEY=sk-or-...
python main.py --model openrouter/qwen/qwen3-vl-235b-a22b-instruct
```

To exercise models that expose OpenRouter's reasoning feature, add `--reasoning-effort high` (or `minimal|low|medium`). The runner will surface a clear `reasoning_not_supported` error if the chosen model does not accept reasoning requests.

To queue multiple models in one go (with a 60s gap between runs by default), and mix reasoning/no-reasoning per model:
```bash
python main.py --models \
  openai/o3 \                           # inherits --reasoning-effort if provided
  openai/o4-mini:reasoning=high \       # force reasoning
  anthropic/claude-sonnet-4.5:reasoning=none  # disable reasoning for this model
```
If you also pass `--reasoning-effort medium`, that effort is the default for entries without an explicit override.

The script will:
1. Load ground-truth answers from `truth.txt` (one integer per line).
2. Convert the corresponding image files to base64 data URIs.
3. Query the specified vision-capable model with a fixed question.
4. Parse the first integer in the model response and compare against ground truth.
5. Print per-image results, a summary block, and persist a JSON summary under `runs/<model_slug>/summary.json` (or `runs/<model_slug>__reasoning+<effort>/summary.json` when reasoning is enabled) so reasoning and non-reasoning runs are kept separately.

## CLI Options
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | OpenRouter model alias (must support vision) | `openrouter/qwen/qwen3-vl-235b-a22b-instruct` |
| `--models` | Queue multiple models to run sequentially (overrides `--model`) | `None` |
| `--imgs` | Directory containing `1.png..N.png` | `imgs` |
| `--truth` | Path to truth labels file | `truth.txt` |
| `--n` | Number of images to evaluate (defaults to auto-detecting how many `*.png` images exist in `--imgs`) | auto |
| `--concurrency` | Max in-flight requests | `4` |
| `--request-timeout` | Per-item timeout seconds | `1200` |
| `--max-retries` | Retry attempts per image | `5` |
| `--rate-limit-backoff` | Base seconds to wait on rate limits | `5.0` |
| `--base-url` | OpenRouter-compatible base URL (default resolves to `https://openrouter.ai/api/v1`) | `https://openrouter.ai/api/v1` |
| `--model-delay` | Seconds to wait between sequential models when `--models` is used | `60` |
| `--reasoning-effort` | Default reasoning effort for all models (one of `minimal`, `low`, `medium`, `high`); can be overridden per model via `model:reasoning=<effort>` or disabled with `reasoning=none` | `None` |
| `--max-tokens` | Response token budget | `8192` |
| `--log-level` | Logging verbosity (`DEBUG`, `INFO`, etc.) | `INFO` |
| `--log-file` | Optional log output path | `None` |
| `--progress` | Print `[progress]` lines as items finish | `False` |

## Logging
Structured logging is provided via Python's `logging` module.
- Console logs respect `--log-level`.
- Supply `--log-file myrun.log` to capture the same output to disk.
- `DEBUG` level includes raw OpenRouter responses, text extraction previews, and reasoning metadata (e.g., `reasoning_len`, `thinking_blocks`).
- Pass `--progress` to print a running `[progress] completed/total` line whenever an item finishes.

## Run Records
- Every run writes (and overwrites) `runs/<model_slug>/summary.json`, where `model_slug` is a filesystem-safe version of the `--model` name (slashes replaced with underscores). If `--reasoning-effort` is set, the run folder becomes `runs/<model_slug>__reasoning+<effort>` so reasoning runs do not overwrite non-reasoning runs (or other reasoning levels).
- The summary captures per-item predictions, correctness, errors, token usage, aggregate latency/accuracy stats, and an estimated dollar cost driven by `model_prices.json` (or default environment variables). A `model_label` field records the display name used in charts (e.g., `openai/gpt-4o (reasoning +high)`).
- Delete the corresponding folder under `runs/` if you want to reset a model's history.

### Pricing data
- Edit `model_prices.json` to assign USD prices per 1M input/prompt tokens **and** per 1M output/completion tokens. Keys can be either the literal `--model` string or its filesystem-safe slug. Example:
  ```json
  {
    "openai/gpt-5-pro": { "input": 10.0, "output": 30.0 },
    "openrouter_qwen_qwen3-vl-235b-a22b-instruct": { "input": 1.2, "output": 4.8 }
  }
  ```
- If the file lacks an entry for a model, the runner reuses prices stored in that model's previous `summary.json` (if present) or falls back to environment defaults:
  - `OPENROUTER_DEFAULT_INPUT_PRICE_PER_MILLION`
  - `OPENROUTER_DEFAULT_OUTPUT_PRICE_PER_MILLION`
  - (legacy) `OPENROUTER_DEFAULT_PRICE_PER_MILLION` applies to both directions if the others are unset.

## Example
```bash
python main.py \
  --model openrouter/qwen/qwen3-vl-235b-a22b-instruct \
  --log-level DEBUG \
  --log-file logs/qwen3-run.log
```

Output excerpt:
```
01.png -> pred=3 truth=2 correct=0 latency_s=12.340 error=no_digit
    debug: finish=stop choices=1 content_type=list content_len=2 reasoning_len=512 thinking_blocks=0 usage={...}
    text: 'There are approximately three...'

--- summary ---
model=openrouter/qwen/qwen3-vl-235b-a22b-instruct
n=100 concurrency=8 total_time_s=650.000 throughput_ips=0.15
accuracy_exact=0.5000  mae=1.0000
```

## Tips
- Ensure the chosen model is multimodal; text-only models return refusals or empty outputs.
- Increase `--max-tokens` for reasoning models if truncation occurs (watch for `finish=length`).
- When using shared gateways (OpenRouter, etc.), keep `--concurrency` low (1–3) and tweak `--rate-limit-backoff` if you see 429s.
- For quick smoke tests use smaller subsets: `python main.py --n 5 --concurrency 2`.
- Review `runs/<model_slug>/summary.json` for downstream analysis or visualization.

## Troubleshooting
- **All predictions empty**: Verify API key access and that the model supports images. Consider increasing `--max-tokens` or checking network proxies.
- **High latency**: Lower `--n`/`--concurrency`, or run with a faster model. Logging at `INFO` helps correlate timing.
- **Unexpected parsing**: Examine logs (`--log-level DEBUG`) to confirm extracted text matches expectations.

## License
Not specified; consult repository owner.
