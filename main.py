#!/usr/bin/env python3
"""
OpenRouter Vision Benchmark (lean + parallel)

- Reads images: ./imgs/1.png ... ./imgs/20.png  (configurable)
- Reads truths: ./truth.txt  (one integer per line; line i => image i.png)
- Asks each image the same question:
    "How many distinct intersections of different line segments are in this image?"
- Sends vision prompts to the given OpenRouter model alias (default: "openrouter/qwen/qwen3-vl-235b-a22b-instruct")
- Runs asynchronously with bounded concurrency
- Reports exact-match accuracy, MAE, latency stats, throughput
- Optionally writes a CSV of per-item results

Requirements:
    pip install aiohttp

Env:
    Set OPENROUTER_API_KEY (and optionally OPENROUTER_SITE_URL / OPENROUTER_APP_TITLE for headers)

Note:
    Use a vision-capable model (e.g., openai/gpt-4o, openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-20240620,
    google/gemini-1.5-flash). Text-only models will return non-vision refusals, yielding empty predictions.
"""

import argparse
import asyncio
import base64
import logging
import mimetypes
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


QUESTION = "How many distinct intersections of different line segments are in this image? Reply with ONLY a number and nothing else. Do not preamble or give any other information."

SYSTEM_MSG = (
    "You are a precise vision assistant. For the given image, return ONLY a single "
    "non-negative INTEGER: the count of distinct intersections formed by different line segments. "
    "No words, no punctuation, no preamble, just the number."
)

INT_RE = re.compile(r"\d+")

OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"


logger = logging.getLogger("ibench")


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error."""


class RateLimitError(OpenRouterError):
    """Raised when OpenRouter signals a rate-limit (HTTP 429)."""


def safe_preview(obj: Any, limit: int = 2000) -> str:
    try:
        text = json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)), ensure_ascii=False)
    except Exception:
        try:
            text = repr(obj)
        except Exception:
            text = "<unserializable>"
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


def preview_text(text: str, limit: int = 500) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


@dataclass
class ItemResult:
    index: int
    truth: Optional[int]
    pred: Optional[int]
    correct: bool
    latency_s: float
    error: Optional[str] = None
    # Debug-only fields (not written to CSV)
    debug: Optional[str] = None
    raw_text_preview: Optional[str] = None


def load_truths(truth_path: Path, n: int) -> Dict[int, int]:
    lines = truth_path.read_text(encoding="utf-8").strip().splitlines()
    vals: List[int] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        m = INT_RE.search(ln)
        if m:
            vals.append(int(m.group()))
    if len(vals) < n:
        raise ValueError(f"truth.txt has only {len(vals)} entries, but --n requested {n}")
    # Map i -> truth for i in 1..n
    return {i + 1: vals[i] for i in range(n)}


def detect_num_images(imgs_dir: Path) -> int:
    count = 0
    while True:
        candidate = imgs_dir / f"{count + 1}.png"
        if candidate.exists():
            count += 1
        else:
            break
    return count


def to_data_uri(img_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(img_path))
    if mime is None:
        # Default to PNG if unknown
        mime = "image/png"
    b = img_path.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_messages(data_uri: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]


def build_api_url(base_url: Optional[str]) -> str:
    base = (base_url or OPENROUTER_DEFAULT_BASE).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def build_openrouter_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_SITE_URL")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_APP_TITLE")
    if title:
        headers["X-Title"] = title
    return headers


async def openrouter_completion(
    session: aiohttp.ClientSession,
    api_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_s) if timeout_s and timeout_s > 0 else aiohttp.ClientTimeout(total=None)
    try:
        async with session.post(api_url, json=payload, headers=headers, timeout=timeout) as resp:
            text_body = await resp.text()
            if resp.status == 429:
                raise RateLimitError(f"429 rate limit: {text_body[:200]}")
            if resp.status >= 400:
                raise OpenRouterError(f"HTTP {resp.status}: {text_body[:200]}")
            if not text_body:
                raise OpenRouterError("Empty response body")
            try:
                return json.loads(text_body)
            except json.JSONDecodeError as exc:
                raise OpenRouterError(f"Invalid JSON response: {text_body[:200]}") from exc
    except asyncio.TimeoutError as exc:
        raise OpenRouterError(f"Request timed out after {timeout_s}s") from exc
    except aiohttp.ClientError as exc:
        raise OpenRouterError(f"HTTP client error: {exc}") from exc


def parse_first_int(text: str) -> Optional[int]:
    if not text:
        return None
    m = INT_RE.search(text)
    return int(m.group()) if m else None


def extract_text(resp: Any) -> str:
    """Best-effort extraction of assistant text across providers/object shapes.

    Tries OpenAI-style objects and dicts; flattens list content blocks if present.
    Returns an empty string if nothing textual is found.
    """
    try:
        choices = getattr(resp, "choices", None)
        if choices is None and isinstance(resp, dict):
            choices = resp.get("choices")
        if not choices:
            return ""

        choice0 = choices[0]
        msg = getattr(choice0, "message", None)
        if msg is None and isinstance(choice0, dict):
            msg = choice0.get("message")
        if msg is None:
            return ""

        # message.content can be a string or list of content parts
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        texts: List[str] = []

        # 1) Standard assistant content
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    # Prefer explicit textual fields; include common alternates
                    for key in (
                        "text",
                        "content",
                        "output_text",
                        "reasoning",
                        "reasoning_content",
                        "thinking",
                        "markdown",
                        "value",
                    ):
                        val = part.get(key)
                        if isinstance(val, str) and val:
                            texts.append(val)

        # 2) Gateway-standardized reasoning content (OpenRouter style)
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content is None and isinstance(msg, dict):
            reasoning_content = msg.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            texts.append(reasoning_content)

        # 3) Anthropic-style thinking blocks (standardized by common gateways)
        thinking_blocks = getattr(msg, "thinking_blocks", None)
        if thinking_blocks is None and isinstance(msg, dict):
            thinking_blocks = msg.get("thinking_blocks")
        if isinstance(thinking_blocks, list):
            for tb in thinking_blocks:
                if isinstance(tb, dict) and isinstance(tb.get("thinking"), str):
                    texts.append(tb["thinking"])

        return "\n".join(t for t in texts if isinstance(t, str) and t.strip())
    except Exception:
        return ""


async def eval_one(
    idx: int,
    model: str,
    data_uri: str,
    truth: Optional[int],
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    api_url: str,
    headers: Dict[str, str],
    request_timeout_s: float,
    max_retries: int,
    max_tokens: int,
    rate_limit_backoff: float,
) -> ItemResult:
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            choice0 = None
            logger.debug("Item %02d attempt %d starting", idx, attempt + 1)
            async with semaphore:
                t0 = time.perf_counter()

                payload = {
                    "model": model,
                    "messages": build_messages(data_uri),
                    "max_tokens": max_tokens,
                }
                resp = await openrouter_completion(
                    session=session,
                    api_url=api_url,
                    headers=headers,
                    payload=payload,
                    timeout_s=request_timeout_s,
                )
                t1 = time.perf_counter()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Item %02d raw response: %s", idx, safe_preview(resp))
                content = extract_text(resp)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Item %02d extracted text: %s", idx, preview_text(content))
                pred = parse_first_int(content)
                # Classify simple non-exception failure modes for easier debugging
                err: Optional[str] = None
                if not content:
                    err = "empty_content"
                elif pred is None:
                    err = "no_digit"

                # Build lean debug summary (printed on failure only)
                debug_str = None
                text_preview = None
                try:
                    # Basic fields from response
                    choices = getattr(resp, "choices", None)
                    if choices is None and isinstance(resp, dict):
                        choices = resp.get("choices")
                    choice0 = choices[0] if choices else None
                    msg = None
                    if choice0 is not None:
                        msg = getattr(choice0, "message", None)
                        if msg is None and isinstance(choice0, dict):
                            msg = choice0.get("message")
                    # finish reason
                    finish_reason = None
                    if choice0 is not None:
                        finish_reason = getattr(choice0, "finish_reason", None)
                        if finish_reason is None and isinstance(choice0, dict):
                            finish_reason = choice0.get("finish_reason")
                    # usage
                    usage = getattr(resp, "usage", None)
                    if usage is None and isinstance(resp, dict):
                        usage = resp.get("usage")
                    if hasattr(usage, "__dict__"):
                        usage = usage.__dict__
                    # content shape
                    msg_content = None
                    if msg is not None:
                        msg_content = getattr(msg, "content", None)
                        if msg_content is None and isinstance(msg, dict):
                            msg_content = msg.get("content")
                    content_type = type(msg_content).__name__ if msg_content is not None else "None"
                    content_len = (len(msg_content) if isinstance(msg_content, str) else (len(msg_content) if isinstance(msg_content, list) else 0))
                    # reasoning
                    reasoning_content = None
                    if msg is not None:
                        reasoning_content = getattr(msg, "reasoning_content", None)
                        if reasoning_content is None and isinstance(msg, dict):
                            reasoning_content = msg.get("reasoning_content")
                    reasoning_len = len(reasoning_content) if isinstance(reasoning_content, str) else 0
                    # thinking blocks
                    thinking_blocks = None
                    if msg is not None:
                        thinking_blocks = getattr(msg, "thinking_blocks", None)
                        if thinking_blocks is None and isinstance(msg, dict):
                            thinking_blocks = msg.get("thinking_blocks")
                    thinking_count = len(thinking_blocks) if isinstance(thinking_blocks, list) else 0
                    # Build preview of extracted content
                    text_preview = (content[:160] + ("..." if len(content) > 160 else "")) if content else None
                    usage_str = None
                    try:
                        usage_str = json.dumps(usage) if usage is not None else None
                    except Exception:
                        usage_str = str(usage)

                    debug_str = (
                        f"finish={finish_reason} choices={len(choices) if choices else 0} "
                        f"content_type={content_type} content_len={content_len} "
                        f"reasoning_len={reasoning_len} thinking_blocks={thinking_count} "
                        f"usage={usage_str}"
                    )
                except Exception as _dbg_e:
                    debug_str = f"debug_error={type(_dbg_e).__name__}: {_dbg_e}"

                correct = (pred == truth) if (pred is not None and truth is not None) else False
                item = ItemResult(
                    index=idx,
                    truth=truth,
                    pred=pred,
                    correct=bool(correct),
                    latency_s=(t1 - t0),
                    error=err,
                    debug=debug_str,
                    raw_text_preview=text_preview,
                )

                if logger.isEnabledFor(logging.INFO):
                    summary_parts = [
                        f"pred={item.pred}",
                        f"truth={item.truth}",
                        f"error={item.error or 'none'}",
                        f"latency_s={item.latency_s:.3f}",
                    ]
                    if item.debug:
                        summary_parts.append(item.debug)
                    logger.info("Item %02d result: %s", idx, " | ".join(summary_parts))
                if logger.isEnabledFor(logging.DEBUG):
                    if content:
                        logger.debug("Item %02d content preview: %s", idx, preview_text(content, limit=800))
                    if choice0 is not None:
                        logger.debug("Item %02d choice0 preview: %s", idx, safe_preview(choice0))

                return item
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            delay: float
            if isinstance(e, RateLimitError):
                delay = rate_limit_backoff * (attempt + 1)
                logger.warning(
                    "Item %02d attempt %d hit rate limit; sleeping %.2fs (%s)",
                    idx,
                    attempt + 1,
                    delay,
                    last_err,
                )
            else:
                delay = 0.4 * (2**attempt)
                logger.warning("Item %02d attempt %d failed with %s", idx, attempt + 1, last_err)
            # Add a small jitter to avoid thundering herd
            delay += random.uniform(0.0, 0.5)
            await asyncio.sleep(delay)

    logger.error("Item %02d failed after %d attempts: %s", idx, max_retries, last_err)
    return ItemResult(index=idx, truth=truth, pred=None, correct=False, latency_s=0.0, error=last_err)


def summarize(results: List[ItemResult]) -> Tuple[float, float, float, float, float, int]:
    lats = [r.latency_s for r in results if r.latency_s > 0]
    acc = sum(r.correct for r in results) / len(results) if results else 0.0
    abs_errs = [abs((r.pred or 0) - (r.truth or 0)) for r in results if r.pred is not None and r.truth is not None]
    mae = (sum(abs_errs) / len(abs_errs)) if abs_errs else float("nan")
    p50 = statistics.median(lats) if lats else float("nan")
    p95 = (statistics.quantiles(lats, n=20)[18] if len(lats) >= 20 else (max(lats) if lats else float("nan")))
    failures = sum(1 for r in results if r.pred is None)
    return acc, mae, (sum(lats) / len(lats) if lats else float("nan")), p50, p95, failures


def write_csv(path: Path, results: List[ItemResult]) -> None:
    total = len(results)
    correct_count = sum(1 for r in results if r.correct)
    percent_correct = (
        f"{(correct_count / total) * 100.0:.2f}%"
        if total
        else "N/A"
    )
    lines = ["index,truth,pred,correct,latency_s,error,percent_correct"]
    for r in sorted(results, key=lambda x: x.index):
        row = [
            str(r.index),
            "" if r.truth is None else str(r.truth),
            "" if r.pred is None else str(r.pred),
            str(int(r.correct)),
            f"{r.latency_s:.6f}",
            "" if r.error is None else repr(r.error).replace(",", ";"),
            "",
        ]
        lines.append(",".join(row))
    lines.append(",".join(["summary", "", "", "", "", "", percent_correct]))
    path.write_text("\n".join(lines), encoding="utf-8")


def setup_logging(log_level: str, log_file: Optional[str]) -> None:
    level_value = getattr(logging, log_level.upper(), None)
    if not isinstance(level_value, int):
        raise ValueError(f"Invalid log level: {log_level}")

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=level_value,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


async def main_async(args):
    imgs_dir = Path(args.imgs).resolve()
    if not imgs_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {imgs_dir}")

    truth_path = Path(args.truth).resolve()
    if args.n is None:
        detected = detect_num_images(imgs_dir)
        if detected <= 0:
            raise FileNotFoundError(f"No sequential PNG images found in {imgs_dir}")
        n = detected
    else:
        n = args.n
        if n <= 0:
            raise ValueError("--n must be a positive integer")
    truths = load_truths(truth_path, n)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set for OpenRouter access")
    api_url = build_api_url(args.base_url)
    headers = build_openrouter_headers(api_key)

    # Pre-encode images to data URIs (fast local CPU; avoids repeated disk I/O in tasks)
    img_indices = list(range(1, n + 1))
    data_uris: Dict[int, str] = {}
    for i in img_indices:
        p = imgs_dir / f"{i}.png"
        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")
        data_uris[i] = to_data_uri(p)

    sem = asyncio.Semaphore(args.concurrency)

    t_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [
            eval_one(
                idx=i,
                model=args.model,
                data_uri=data_uris[i],
                truth=truths.get(i),
                semaphore=sem,
                session=session,
                api_url=api_url,
                headers=headers,
                request_timeout_s=args.request_timeout,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                rate_limit_backoff=args.rate_limit_backoff,
            )
            for i in img_indices
        ]
        results = await asyncio.gather(*tasks)
    t_end = time.perf_counter()

    # Per-item lines
    for r in sorted(results, key=lambda x: x.index):
        print(
            f"{r.index:02d}.png -> pred={r.pred} truth={r.truth} "
            f"correct={int(r.correct)} latency_s={r.latency_s:.3f}"
            + (f" error={r.error}" if r.error else "")
        )
        # Lean debug by default: only print when we fail to parse a usable number
        if (r.pred is None) or (r.error is not None) or (not r.correct):
            if r.debug:
                print(f"    debug: {r.debug}")
            if r.raw_text_preview:
                print(f"    text: {r.raw_text_preview!r}")

    # Summary
    acc, mae, avg_lat, p50, p95, failures = summarize(results)
    num_correct = sum(1 for r in results if r.correct)
    total = len(results)
    total_time = t_end - t_start
    throughput = len(results) / total_time if total_time > 0 else float("nan")
    print("\n--- summary ---")
    print(f"model={args.model}")
    if args.base_url:
        print(f"base_url={args.base_url}")
    print(f"n={len(results)} concurrency={args.concurrency} total_time_s={total_time:.3f} throughput_ips={throughput:.2f}")
    print(f"accuracy_exact={acc:.4f}  mae={mae:.4f}")
    print(f"latency_avg_s={avg_lat:.3f}  p50_s={p50:.3f}  p95_s={p95:.3f}")
    print(f"failures={failures}")
    # End with a simple percent-correct summary
    if total > 0:
        print(f"percent_correct={acc * 100:.1f}% ({num_correct}/{total})")

    if args.csv:
        out_path = Path(args.csv).resolve()
        write_csv(out_path, results)
        print(f"csv={out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenRouter vision intersection-count benchmark")
    p.add_argument(
        "--model",
        type=str,
        default="openrouter/qwen/qwen3-vl-235b-a22b-instruct",
        help="OpenRouter model name/alias (default: openrouter/qwen/qwen3-vl-235b-a22b-instruct)",
    )
    p.add_argument("--imgs", type=str, default="imgs", help="Directory containing images named 1.png..N.png (default: ./imgs)")
    p.add_argument("--truth", type=str, default="truth.txt", help="Path to truth.txt (one integer per line)")
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of images to evaluate (default: auto-detect count in --imgs)",
    )
    p.add_argument("--concurrency", type=int, default=4, help="Max in-flight requests (default: 4)")
    p.add_argument("--request-timeout", type=float, default=1200.0, help="Per-request timeout seconds (default: 1200)")
    p.add_argument("--max-retries", type=int, default=5, help="Retries per item (default: 5)")
    p.add_argument(
        "--rate-limit-backoff",
        type=float,
        default=5.0,
        help="Seconds to wait (multiplied per retry) when a rate limit error occurs (default: 5.0)",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=OPENROUTER_DEFAULT_BASE,
        help="OpenRouter-compatible base URL (default: https://openrouter.ai/api/v1)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens for response (default: 8192; reasoning models often need >32)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR; default: INFO)",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write detailed logs",
    )
    p.add_argument("--csv", type=str, default="results.csv", help="Write per-item results CSV (default: results.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
