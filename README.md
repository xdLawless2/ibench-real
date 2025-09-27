# ibench-real

LiteLLM-powered micro-benchmark for counting line-intersection points in 20 synthetic images.

## Features
- Async evaluation of a numbered image set (`./imgs/1.png` … `./imgs/20.png`).
- Ground-truth comparison via `truth.txt` with accuracy, MAE, latency, and throughput metrics.
- CSV export summarizing predictions, errors, and latencies per item.
- Optional structured logging (console + file) capturing raw model outputs and reasoning metadata.

## Requirements
- Python 3.9+
- `litellm` (install via `pip install litellm`)
- Provider API key(s) exposed via environment (e.g., `OPENAI_API_KEY`).

## Quick Start
```bash
pip install litellm
python main.py --model openai/gpt-5 --n 20
```

The script will:
1. Load ground-truth answers from `truth.txt` (one integer per line).
2. Convert the corresponding image files to base64 data URIs.
3. Query the specified vision-capable model with a fixed question.
4. Parse the first integer in the model response and compare against ground truth.
5. Print per-image results, a summary block, and write `results.csv` (unless overridden).

## CLI Options
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | LiteLLM model alias (must support vision) | `openai/gpt-5` |
| `--imgs` | Directory containing `1.png..N.png` | `imgs` |
| `--truth` | Path to truth labels file | `truth.txt` |
| `--n` | Number of images to evaluate | `20` |
| `--concurrency` | Max in-flight requests | `4` |
| `--request-timeout` | Per-item timeout seconds | `1200` |
| `--max-retries` | Retry attempts per image | `5` |
| `--rate-limit-backoff` | Base seconds to wait on rate limits | `5.0` |
| `--base-url` | Custom LiteLLM/OpenAI-compatible endpoint | `None` |
| `--max-tokens` | Response token budget | `8192` |
| `--log-level` | Logging verbosity (`DEBUG`, `INFO`, etc.) | `INFO` |
| `--log-file` | Optional log output path | `None` |
| `--csv` | Per-item results CSV path | `results.csv` |

## Logging
Structured logging is provided via Python's `logging` module.
- Console logs respect `--log-level`.
- Supply `--log-file myrun.log` to capture the same output to disk.
- `DEBUG` level includes raw LiteLLM responses, text extraction previews, and reasoning metadata (e.g., `reasoning_len`, `thinking_blocks`).

## Example
```bash
python main.py \
  --model openai/gpt-5 \
  --log-level DEBUG \
  --log-file logs/gpt-5-run.log
```

Output excerpt:
```
01.png -> pred=3 truth=2 correct=0 latency_s=12.340 error=no_digit
    debug: finish=stop choices=1 content_type=list content_len=2 reasoning_len=512 thinking_blocks=0 usage={...}
    text: 'There are approximately three...'

--- summary ---
model=openai/gpt-5
n=20 concurrency=8 total_time_s=130.433 throughput_ips=0.15
accuracy_exact=0.5000  mae=1.0000
```

## Tips
- Ensure the chosen model is multimodal; text-only models return refusals or empty outputs.
- Increase `--max-tokens` for reasoning models if truncation occurs (watch for `finish=length`).
- When using shared gateways (OpenRouter, etc.), keep `--concurrency` low (1–3) and tweak `--rate-limit-backoff` if you see 429s.
- For quick smoke tests use smaller subsets: `python main.py --n 5 --concurrency 2`.
- Inspect the generated CSV for downstream analysis or visualisation.

## Troubleshooting
- **All predictions empty**: Verify API key access and that the model supports images. Consider increasing `--max-tokens` or checking network proxies.
- **High latency**: Lower `--n`/`--concurrency`, or run with a faster model. Logging at `INFO` helps correlate timing.
- **Unexpected parsing**: Examine logs (`--log-level DEBUG`) to confirm extracted text matches expectations.

## License
Not specified; consult repository owner.
