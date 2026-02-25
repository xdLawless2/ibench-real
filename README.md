# ibench-real

Multi-provider micro-benchmark for evaluating vision language models' ability to count line-intersection points in synthetic images. Supports OpenRouter and Google Gemini APIs.

## Features
- **Multi-provider support**: Run benchmarks via OpenRouter or directly against Google's Gemini API.
- **Synthetic image generation** with precise intersection counts using geometric algorithms.
- **Async evaluation** of a numbered image set (`./public/imgs/1.png` … `./public/imgs/N.png`; auto-detects count).
- **Ground-truth comparison** via `truth.txt` with accuracy, MAE, latency, and throughput metrics.
- **Automatic per-model run archives** under `./runs/<model_slug>/summary.json` containing per-item outcomes, token usage, and estimated costs.
- **Visualization tools** for generating comparison charts (accuracy leaderboard, cost vs accuracy, tokens vs accuracy).
- **Optional structured logging** (console + file) capturing raw model outputs and reasoning metadata.
- **Multi-model queuing** with `--models`, separated by a configurable delay, with per-model reasoning overrides.

## Requirements
- Python 3.9+
- Install dependencies: `pip install -r requirements.txt`

Or install manually:
- `aiohttp` - HTTP client for OpenRouter API
- `matplotlib` - for graph generation
- `Pillow` - for image generation
- `google-genai` - for Google Gemini API (optional, only needed for `google:` provider)

## API Keys

Set environment variables for the providers you want to use:

```bash
# For OpenRouter (default provider)
export OPENROUTER_API_KEY=sk-or-...

# For Google Gemini (required for google: provider)
export GEMINI_API_KEY=...
```

## Quick Start

### Using OpenRouter (default)
```bash
export OPENROUTER_API_KEY=sk-or-...
python main.py --model qwen/qwen3-vl-235b-a22b-instruct
```

### Using Google Gemini
```bash
export GEMINI_API_KEY=...
python main.py --model google:gemini-2.5-flash
```

### Mixed providers
```bash
export OPENROUTER_API_KEY=sk-or-...
export GEMINI_API_KEY=...
python main.py --models \
  google:gemini-2.5-flash \
  openai/gpt-4o \
  google:gemini-3-pro:reasoning=high
```

To exercise models that expose OpenRouter's reasoning feature, add `--reasoning-effort high` (or `minimal|low|medium|xhigh`). The runner will surface a clear `reasoning_not_supported` error if the chosen model does not accept reasoning requests.

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
| `--model` | Model with optional provider prefix. Format: `[provider:]model[:reasoning=LEVEL]` | `qwen/qwen3-vl-235b-a22b-instruct` |
| `--models` | Queue multiple models to run sequentially. Supports mixed providers. | `None` |
| `--imgs` | Directory containing `1.png..N.png` | `public/imgs` |
| `--truth` | Path to truth labels file | `truth.txt` |
| `--n` | Number of images to evaluate (defaults to auto-detecting how many `*.png` images exist in `--imgs`) | auto |
| `--concurrency` | Max in-flight requests | `4` |
| `--request-timeout` | Per-item timeout seconds | `1200` |
| `--max-retries` | Retry attempts per image | `5` |
| `--rate-limit-backoff` | Base seconds to wait on rate limits | `5.0` |
| `--base-url` | OpenRouter-compatible base URL (default resolves to `https://openrouter.ai/api/v1`) | `https://openrouter.ai/api/v1` |
| `--model-delay` | Seconds to wait between sequential models when `--models` is used | `60` |
| `--reasoning-effort` | Default reasoning effort for all models (one of `minimal`, `low`, `medium`, `high`, `xhigh`); can be overridden per model via `model:reasoning=<effort>` or disabled with `reasoning=none` | `None` |
| `--log-level` | Logging verbosity (`DEBUG`, `INFO`, etc.) | `INFO` |
| `--log-file` | Optional log output path | `None` |
| `--progress` | Print `[progress]` lines as items finish | `False` |

## Providers

### OpenRouter (default)
- Provider prefix: `openrouter:` (or omit for default)
- API key: `OPENROUTER_API_KEY`
- Supports all OpenRouter vision models
- Reasoning: Uses `--reasoning-effort` with levels `minimal`, `low`, `medium`, `high`

### Google Gemini
- Provider prefix: `google:`
- API key: `GEMINI_API_KEY`
- Supports Gemini 2.5 and 3.x vision models
- Reasoning:
  - **Gemini 2.5**: Uses dynamic thinking by default (no configuration needed)
  - **Gemini 3**: Supports `--reasoning-effort low` or `--reasoning-effort high` only

Examples:
```bash
# OpenRouter models (default provider)
python main.py --model openai/gpt-4o
python main.py --model anthropic/claude-sonnet-4.5

# Google Gemini models
python main.py --model google:gemini-2.5-flash
python main.py --model google:gemini-2.5-pro
python main.py --model google:gemini-3-pro:reasoning=high
```

## Logging
Structured logging is provided via Python's `logging` module.
- Console logs respect `--log-level`.
- Supply `--log-file myrun.log` to capture the same output to disk.
- `DEBUG` level includes raw OpenRouter responses, text extraction previews, and reasoning metadata (e.g., `reasoning_len`, `thinking_blocks`).
- Pass `--progress` to print a running `[progress] completed/total` line whenever an item finishes.

## Run Records
- Every run writes (and overwrites) `runs/<model_slug>/summary.json`, where `model_slug` is a filesystem-safe version of the `--model` name (slashes replaced with underscores). If `--reasoning-effort` is set, the run folder becomes `runs/<model_slug>__reasoning+<effort>` so reasoning runs do not overwrite non-reasoning runs (or other reasoning levels).
- The summary captures per-item predictions, correctness, errors, token usage, aggregate latency/accuracy stats, and an estimated dollar cost driven by `model_prices.json` (or default environment variables). It includes:
  - `model_label`: full run label (provider/model, plus reasoning when requested)
  - `benchmark_label`: chart label used by `scripts/graph-maker.py` (e.g., `gpt-4o (high reasoning)`)
- To rename a model in `benchmark.jpg`/`cost_vs_accuracy.jpg` without rerunning benchmarks, edit `runs/<run_slug>/summary.json` and change `benchmark_label`, then regenerate graphs.
- The summary also records `reasoning_observed`/`reasoning_tokens_total` and an `effective_reasoning_effort`. If you did not pass `--reasoning-effort` but the API returns reasoning tokens/content, the runner treats the effective effort as `medium` (implicit default).
- Delete the corresponding folder under `runs/` if you want to reset a model's history.

### Pricing data
- Edit `config/model_prices.json` to assign USD prices per 1M input/prompt tokens **and** per 1M output/completion tokens. Keys can be either the literal `--model` string or its filesystem-safe slug. Example:
  ```json
  {
    "openai/gpt-5-pro": { "input": 10.0, "output": 30.0 },
    "openrouter_qwen_qwen3-vl-235b-a22b-instruct": { "input": 1.2, "output": 4.8 }
  }
  ```
- If the file lacks an entry for a model, the runner reuses prices stored in that model's previous `summary.json` (if present). If no prior summary exists, it will try to fetch prices from the OpenRouter `/models` endpoint. If that also fails, it falls back to environment defaults:
  - `OPENROUTER_DEFAULT_INPUT_PRICE_PER_MILLION`
  - `OPENROUTER_DEFAULT_OUTPUT_PRICE_PER_MILLION`
  - (legacy) `OPENROUTER_DEFAULT_PRICE_PER_MILLION` applies to both directions if the others are unset.

## Example
```bash
python main.py \
  --model qwen/qwen3-vl-235b-a22b-instruct \
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

## Image Generation

Generate synthetic benchmark images with known intersection counts using `scripts/generate-imgs.py`:

```bash
python scripts/generate-imgs.py 100 -o public/imgs -s 42
```

| Flag | Description | Default |
|------|-------------|---------|
| `n` (positional) | Number of images to generate | required |
| `-o`, `--output` | Output directory | `benchmark_output` |
| `-s`, `--seed` | Random seed for reproducibility | `None` |

The generator:
1. Creates images with 2-12 random line segments each.
2. Targets 1-6 intersections per image using geometric algorithms.
3. Outputs `1.png` through `N.png` in the output directory.
4. Generates `truth.txt` with the ground-truth intersection count per image.

## Visualization

Generate comparison charts from benchmark results using `scripts/graph-maker.py`:

```bash
python scripts/graph-maker.py --runs runs --output-dir graphs
```

| Flag | Description | Default |
|------|-------------|---------|
| `--runs` | Directory containing `<model>/summary.json` folders | `runs` |
| `--output-dir` | Directory for generated graphs | `graphs` |
| `--output` | Filename for the accuracy bar chart | `benchmark.jpg` |
| `--state` | Path to state file tracking previously seen runs | inside output-dir |

Generated charts:
- **benchmark.jpg**: Horizontal bar chart accuracy leaderboard (new runs highlighted with gold borders).
- **output_tokens_vs_accuracy.jpg**: Scatter plot of output tokens vs accuracy.
- **cost_vs_accuracy.jpg**: Scatter plot of estimated cost vs accuracy (log scale).

## Tips
- Ensure the chosen model is multimodal; text-only models return refusals or empty outputs.
- When using shared gateways (OpenRouter, etc.), keep `--concurrency` low (1–3) and tweak `--rate-limit-backoff` if you see 429s.
- For quick smoke tests use smaller subsets: `python main.py --n 5 --concurrency 2`.
- Review `runs/<model_slug>/summary.json` for downstream analysis or visualization.

## Troubleshooting
- **All predictions empty**: Verify API key access and that the model supports images. Consider checking network proxies or reducing concurrency.
- **High latency**: Lower `--n`/`--concurrency`, or run with a faster model. Logging at `INFO` helps correlate timing.
- **Unexpected parsing**: Examine logs (`--log-level DEBUG`) to confirm extracted text matches expectations.

## License
Not specified; consult repository owner.
