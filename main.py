#!/usr/bin/env python3
"""
OpenRouter Vision Benchmark (lean + parallel)

- Reads images: ./imgs/1.png ... ./imgs/100.png  (configurable; auto-detects count)
- Reads truths: ./truth.txt  (one integer per line; line i => image i.png; needs at least N lines)
- Asks each image the same question:
    "How many distinct intersections between different shapes are in this image?"
- Sends vision prompts to the given OpenRouter model alias (default: "qwen/qwen3-vl-235b-a22b-instruct")
- Runs asynchronously with bounded concurrency
- Reports exact-match accuracy, MAE, latency stats, throughput
- Optionally writes a CSV of per-item results

Requirements:
    pip install aiohttp

Env:
    Set OPENROUTER_API_KEY (and optionally OPENROUTER_SITE_URL / OPENROUTER_APP_TITLE for headers)

Note:
    Use a vision-capable model (e.g., openai/gpt-4o, openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-20240620).
    Text-only models will return non-vision refusals, yielding empty predictions.
"""

import argparse
import asyncio
import base64
import contextlib
import logging
import mimetypes
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import aiohttp


QUESTION = (
    "How many distinct intersections between different shapes are in this image? "
    "Count each distinct crossing point once. "
    "Return ONLY the final integer answer with no reasoning or explanation."
)

SYSTEM_MSG = (
    "You are a precise vision assistant. For the given image, return ONLY a single "
    "non-negative INTEGER: the count of distinct intersections formed by different shapes. "
    "Do not include reasoning traces. No words, no punctuation, no preamble, just the number."
)

INT_RE = re.compile(r"\d+")

OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"


logger = logging.getLogger("ibench")

RUNS_DIR = Path("runs")
MODEL_PRICE_FILE = Path("config/model_prices.json")


def _read_env_price(var_name: str) -> Optional[float]:
    raw = os.environ.get(var_name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_default_price_pair() -> Tuple[float, float]:
    base = _read_env_price("OPENROUTER_DEFAULT_PRICE_PER_MILLION")
    prompt = _read_env_price("OPENROUTER_DEFAULT_INPUT_PRICE_PER_MILLION")
    completion = _read_env_price("OPENROUTER_DEFAULT_OUTPUT_PRICE_PER_MILLION")
    prompt_val = prompt if prompt is not None else (base if base is not None else 0.0)
    completion_val = completion if completion is not None else (base if base is not None else 0.0)
    return prompt_val, completion_val


DEFAULT_PROMPT_PRICE, DEFAULT_COMPLETION_PRICE = _read_default_price_pair()


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error."""


class RateLimitError(OpenRouterError):
    """Raised when OpenRouter signals a rate-limit (HTTP 429)."""


class UnsupportedReasoningModelError(OpenRouterError):
    """Raised when reasoning was requested on a model that does not support it."""


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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    llm_output: Optional[str] = None


@dataclass
class ModelRunState:
    provider: str
    model: str
    reasoning_effort: Optional[str]
    status: str = "queued"  # queued|waiting|running|done|failed
    total_items: int = 0
    completed_items: int = 0
    correct_items: int = 0
    failure_items: int = 0
    running_latency_sum: float = 0.0
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    summary_path: Optional[str] = None
    last_event: str = ""
    wait_until_ts: Optional[float] = None
    in_flight: Dict[int, Tuple[int, float]] = field(default_factory=dict)  # item_idx -> (attempt, start_ts)
    completed_indices: set[int] = field(default_factory=set)
    last_error_by_item: Dict[int, str] = field(default_factory=dict)
    recent_events: List[str] = field(default_factory=list)


class RunDashboard:
    """Compact terminal dashboard for queued sequential model runs."""

    def __init__(
        self,
        specs: List[Tuple[str, str, Optional[str]]],
        items_per_model: int,
        concurrency: int,
        use_color: bool = True,
        verbose: bool = False,
        force_live: bool = False,
    ):
        self.states = [ModelRunState(provider=p, model=m, reasoning_effort=r) for p, m, r in specs]
        self.items_per_model = items_per_model
        self.concurrency = max(1, concurrency)
        self.start_ts = time.time()
        term = (os.environ.get("TERM") or "").lower()
        # On Windows shells (including Cursor's PowerShell terminal), TERM is often unset
        # even when ANSI + TTY are available. Requiring TERM there disables live TUI updates.
        if os.name == "nt":
            interactive_tty = bool(sys.stdout.isatty() and sys.stderr.isatty())
        else:
            interactive_tty = bool(sys.stdout.isatty() and sys.stderr.isatty() and term not in ("", "dumb"))
        self._enabled = bool(force_live or interactive_tty)
        self._use_color = bool(use_color and self._enabled)
        self._last_render_lines = 0
        self._verbose = bool(verbose)
        self._cursor_hidden = False
        self._screen_active = False
        self._dirty = True

    def _mark_dirty(self) -> None:
        self._dirty = True

    def start_model(self, model_idx: int, total_items: int) -> None:
        st = self.states[model_idx]
        st.status = "running"
        st.total_items = total_items
        st.start_ts = time.time()
        st.end_ts = None
        st.wait_until_ts = None
        self._mark_dirty()

    def set_waiting(self, model_idx: int, wait_seconds: float) -> None:
        st = self.states[model_idx]
        st.status = "waiting"
        st.wait_until_ts = time.time() + max(0.0, wait_seconds)
        st.last_event = f"waiting {wait_seconds:.0f}s"
        self._mark_dirty()

    def on_item(self, model_idx: int, res: ItemResult) -> None:
        st = self.states[model_idx]
        st.completed_items += 1
        st.correct_items += 1 if res.correct else 0
        st.failure_items += 1 if (res.pred is None or res.error is not None) else 0
        st.running_latency_sum += max(0.0, res.latency_s)
        st.completed_indices.add(res.index)
        st.in_flight.pop(res.index, None)
        if res.error:
            st.last_event = f"#{res.index:02d} error={res.error}"
            st.last_error_by_item[res.index] = res.error
            st.recent_events.append(f"#{res.index:02d} FAIL ({res.error})")
        else:
            st.last_event = f"#{res.index:02d} pred={res.pred} truth={res.truth}"
            st.last_error_by_item.pop(res.index, None)
            st.recent_events.append(f"#{res.index:02d} ok pred={res.pred} truth={res.truth}")
        st.recent_events = st.recent_events[-10:]
        self._mark_dirty()

    def mark_item_in_flight(self, model_idx: int, item_idx: int, attempt: int = 1) -> None:
        st = self.states[model_idx]
        st.in_flight[item_idx] = (attempt, time.time())
        self._mark_dirty()

    def clear_item_in_flight(self, model_idx: int, item_idx: int) -> None:
        st = self.states[model_idx]
        if item_idx in st.in_flight:
            st.in_flight.pop(item_idx, None)
            self._mark_dirty()

    def on_retry(self, model_idx: int, item_idx: int, failed_attempt: int, err: str) -> None:
        st = self.states[model_idx]
        st.last_error_by_item[item_idx] = err
        st.last_event = f"#{item_idx:02d} retry a{failed_attempt}"
        st.recent_events.append(f"#{item_idx:02d} retry a{failed_attempt} ({err[:42]})")
        st.recent_events = st.recent_events[-10:]
        self._mark_dirty()

    def finish_model(self, model_idx: int, failed: bool, summary_path: str, metrics: Dict[str, Any]) -> None:
        st = self.states[model_idx]
        st.status = "failed" if failed else "done"
        st.end_ts = time.time()
        st.summary_path = summary_path
        st.last_event = (
            f"acc={metrics.get('accuracy_exact', 0.0) * 100.0:.1f}% "
            f"failures={metrics.get('failures', 0)}"
        )
        self._mark_dirty()

    def finish_all(self) -> None:
        self.render(final=True)
        if self._enabled and self._cursor_hidden:
            sys.stdout.write("\x1b[?25h")
            self._cursor_hidden = False
        if self._enabled and self._screen_active:
            # Leave alternate screen and return to normal terminal buffer.
            sys.stdout.write("\x1b[?1049l")
            self._screen_active = False
        sys.stdout.flush()

    def _c(self, code: str) -> str:
        if not self._use_color:
            return ""
        return f"\x1b[{code}m"

    def _status_symbol(self, status: str) -> str:
        return {
            "queued": "Q",
            "waiting": "W",
            "running": "R",
            "done": "D",
            "failed": "F",
        }.get(status, "?")

    def _status_text(self, status: str) -> str:
        if status == "running":
            return f"{self._c('96')}RUN{self._c('0')}"
        if status == "done":
            return f"{self._c('92')}DONE{self._c('0')}"
        if status == "failed":
            return f"{self._c('91')}FAIL{self._c('0')}"
        if status == "waiting":
            return f"{self._c('93')}WAIT{self._c('0')}"
        return f"{self._c('90')}QUEUED{self._c('0')}"

    def _bar(self, done: int, total: int, width: int = 22) -> str:
        if total <= 0:
            return "." * width
        ratio = max(0.0, min(1.0, done / total))
        fill = int(round(ratio * width))
        left = "#" * fill
        right = "-" * (width - fill)
        if self._use_color and fill > 0:
            left = f"{self._c('92')}{left}{self._c('0')}"
        return left + right

    def _elapsed(self, st: ModelRunState) -> float:
        if st.start_ts is None:
            return 0.0
        end = st.end_ts if st.end_ts is not None else time.time()
        return max(0.0, end - st.start_ts)

    def _fmt_s(self, sec: float) -> str:
        sec = int(max(0, round(sec)))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _active_model_idx(self) -> Optional[int]:
        for i, st in enumerate(self.states):
            if st.status == "running":
                return i
        return None

    def _overall_pct(self) -> float:
        total_items_done = sum(st.completed_items for st in self.states)
        total_items = self.items_per_model * len(self.states)
        if total_items <= 0:
            return 0.0
        return (100.0 * total_items_done) / total_items

    def render(self, final: bool = False) -> None:
        if not self._enabled and not final:
            return
        if not final and not self._dirty:
            return
        lines: List[str] = []
        done_models = sum(1 for st in self.states if st.status in ("done", "failed"))
        active_models = sum(1 for st in self.states if st.status == "running")
        total_items_done = sum(st.completed_items for st in self.states)
        total_items = self.items_per_model * len(self.states)
        elapsed = self._fmt_s(time.time() - self.start_ts)
        active_idx = self._active_model_idx()
        overall_bar = self._bar(total_items_done, max(1, total_items), width=32)
        lines.append(
            f"{self._c('1')}EyeBench-V2 live{self._c('0')}  "
            f"models {done_models}/{len(self.states)}  "
            f"items {total_items_done}/{total_items}  "
            f"active {active_models}  elapsed {elapsed}"
        )
        lines.append(f"overall {overall_bar} {self._overall_pct():5.1f}%")
        if active_idx is not None:
            st = self.states[active_idx]
            avg_lat = (st.running_latency_sum / st.completed_items) if st.completed_items else 0.0
            done = st.completed_items
            total = max(1, st.total_items or self.items_per_model)
            rem = max(0, total - done)
            eta_s = (avg_lat * rem) / self.concurrency
            model_disp = f"{st.provider}:{st.model}"
            if st.reasoning_effort:
                model_disp += f" ({st.reasoning_effort})"
            lines.append(
                f"now   {active_idx + 1}/{len(self.states)} {model_disp} | "
                f"item {done}/{total} | ok {st.correct_items} | fail {st.failure_items} | eta~{self._fmt_s(eta_s)}"
            )

            # Show the currently in-flight questions (typically matches concurrency, e.g. 4)
            active_items = sorted(st.in_flight.items(), key=lambda kv: kv[0])[: self.concurrency]
            if active_items:
                active_parts: List[str] = []
                for item_idx, (attempt, start_ts) in active_items:
                    elapsed_s = int(max(0.0, time.time() - start_ts))
                    active_parts.append(f"#{item_idx:02d}(a{attempt},{elapsed_s}s)")
                lines.append("live  in-flight: " + "  ".join(active_parts))
            else:
                lines.append("live  in-flight: -")

            # Show the next queued questions for the active model.
            pending: List[int] = []
            for i in range(1, total + 1):
                if i in st.completed_indices or i in st.in_flight:
                    continue
                pending.append(i)
                if len(pending) >= self.concurrency:
                    break
            if pending:
                pending_str = "  ".join(f"#{i:02d}" for i in pending)
                lines.append(f"live  next-up : {pending_str}")
            else:
                lines.append("live  next-up : -")

            if self._verbose:
                lines.append("live  details:")
                if active_items:
                    for item_idx, (attempt, start_ts) in active_items:
                        elapsed_s = int(max(0.0, time.time() - start_ts))
                        last_err = st.last_error_by_item.get(item_idx, "none")
                        if len(last_err) > 58:
                            last_err = last_err[:55] + "..."
                        lines.append(
                            f"      q#{item_idx:02d} attempt={attempt} elapsed={elapsed_s:>3}s last_err={last_err}"
                        )
                else:
                    lines.append("      (no in-flight questions)")
                if st.recent_events:
                    lines.append("live  recent:")
                    for ev in st.recent_events[-6:]:
                        lines.append(f"      - {ev}")
        else:
            lines.append("now   idle")

        sep = "-" * max(50, len(lines[0]))
        lines.append(sep)
        lines.append("id  status  model                                   prog%   acc%  fail  avg_s  run_t    event")
        for idx, st in enumerate(self.states, start=1):
            model_name = f"{st.provider}:{st.model}"
            if st.reasoning_effort:
                model_name += f" ({st.reasoning_effort})"
            if len(model_name) > 39:
                model_name = model_name[:36] + "..."
            bar = self._bar(st.completed_items, max(1, st.total_items or self.items_per_model))
            pct = (
                (100.0 * st.completed_items / st.total_items)
                if st.total_items > 0 else 0.0
            )
            acc = (100.0 * st.correct_items / st.completed_items) if st.completed_items else 0.0
            avg_lat = (st.running_latency_sum / st.completed_items) if st.completed_items else 0.0
            event = st.last_event
            if st.status == "waiting" and st.wait_until_ts:
                rem = max(0.0, st.wait_until_ts - time.time())
                event = f"waiting {rem:.0f}s"
            if len(event) > 24:
                event = event[:21] + "..."
            lines.append(
                f"{idx:02d}  {self._status_text(st.status):<7} {model_name:<39} "
                f"{pct:6.1f}  {acc:5.1f}  {st.failure_items:4d}  {avg_lat:5.1f}  {self._fmt_s(self._elapsed(st)):<8} {event}"
            )
            if st.status == "running":
                lines.append(f"    {bar}")

        if final:
            lines.append(sep)
            lines.append(f"{self._c('92')}Run completed.{self._c('0')}")

        text = "\n".join(lines)
        if self._enabled:
            if not self._screen_active:
                # Use alternate screen so terminal scrollback stays clean.
                sys.stdout.write("\x1b[?1049h")
                self._screen_active = True
            if not self._cursor_hidden:
                sys.stdout.write("\x1b[?25l")
                self._cursor_hidden = True
            # Hard repaint: clear full screen + home cursor, then draw one frame.
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(text)
            sys.stdout.write("\x1b[J")
            if final:
                sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_render_lines = len(lines)
            self._dirty = False
        else:
            print(text, flush=True)


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


def build_models_url(base_url: Optional[str]) -> str:
    """Return the OpenRouter-style models listing endpoint for a given base URL."""
    base = (base_url or OPENROUTER_DEFAULT_BASE).rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return f"{base}/models"


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


def detect_reasoning_unsupported(text_body: str) -> Optional[str]:
    """Return a user-friendly message when the API says reasoning is unsupported."""
    candidates: List[str] = []
    try:
        parsed = json.loads(text_body)
        if isinstance(parsed, dict):
            err_obj = parsed.get("error")
            if isinstance(err_obj, dict):
                msg = err_obj.get("message") or err_obj.get("error")
                if isinstance(msg, str):
                    candidates.append(msg)
            # Some providers return {"message": "..."}
            if isinstance(parsed.get("message"), str):
                candidates.append(parsed["message"])
    except json.JSONDecodeError:
        pass

    candidates.append(text_body)

    for msg in candidates:
        if not isinstance(msg, str):
            continue
        lower = msg.lower()
        if "reasoning" in lower and any(key in lower for key in ("not support", "unsupported", "disabled", "unavailable")):
            return msg.strip() or None
    return None


def _coerce_float(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _normalize_model_id(model: str) -> str:
    """OpenRouter sometimes omits a leading 'openrouter/' prefix in model IDs."""
    m = (model or "").strip()
    if m.lower().startswith("openrouter/"):
        return m[len("openrouter/") :]
    return m


async def fetch_openrouter_price_pair(
    session: aiohttp.ClientSession,
    models_url: str,
    headers: Dict[str, str],
    model: str,
) -> Optional[Tuple[float, float]]:
    """Fetch (prompt, completion) USD-per-million prices for a model from /models.

    Returns None if the endpoint is unavailable or the model/pricing is missing.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with session.get(models_url, headers=headers, timeout=timeout) as resp:
            body = await resp.text()
            if resp.status >= 400:
                logger.debug("Price fetch failed HTTP %s: %s", resp.status, body[:200])
                return None
            doc = json.loads(body)
    except Exception as exc:
        logger.debug("Price fetch exception: %s", exc)
        return None

    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, list):
        return None

    target_ids = {
        (model or "").strip().lower(),
        _normalize_model_id(model).lower(),
    }

    for item in data:
        if not isinstance(item, dict):
            continue
        item_id_raw = item.get("id") or item.get("model") or item.get("name")
        if not isinstance(item_id_raw, str):
            continue
        item_id = item_id_raw.strip().lower()
        if item_id not in target_ids and _normalize_model_id(item_id).lower() not in target_ids:
            continue

        pricing = item.get("pricing") or {}
        if not isinstance(pricing, dict):
            return None
        prompt_val = _coerce_float(
            pricing.get("prompt")
            or pricing.get("input")
            or pricing.get("prompt_per_million")
            or pricing.get("input_per_million")
        )
        completion_val = _coerce_float(
            pricing.get("completion")
            or pricing.get("output")
            or pricing.get("completion_per_million")
            or pricing.get("output_per_million")
        )
        if prompt_val is None and completion_val is None:
            return None
        return (prompt_val or 0.0, completion_val or 0.0)

    return None


async def openrouter_completion(
    session: aiohttp.ClientSession,
    api_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: float,
    reasoning_requested: bool = False,
) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_s) if timeout_s and timeout_s > 0 else aiohttp.ClientTimeout(total=None)
    try:
        async with session.post(api_url, json=payload, headers=headers, timeout=timeout) as resp:
            text_body = await resp.text()
            if resp.status == 429:
                raise RateLimitError(f"429 rate limit: {text_body[:200]}")
            reasoning_hint = detect_reasoning_unsupported(text_body) if reasoning_requested else None
            if resp.status >= 400:
                if reasoning_requested and reasoning_hint:
                    raise UnsupportedReasoningModelError(reasoning_hint)
                raise OpenRouterError(f"HTTP {resp.status}: {text_body[:200]}")
            if not text_body:
                raise OpenRouterError("Empty response body")
            try:
                parsed = json.loads(text_body)
            except json.JSONDecodeError as exc:
                raise OpenRouterError(f"Invalid JSON response: {text_body[:200]}") from exc
            if isinstance(parsed, dict):
                err_obj = parsed.get("error")
                if err_obj:
                    err_code: Any = None
                    err_msg: str
                    if isinstance(err_obj, dict):
                        err_code = err_obj.get("code")
                        raw_msg = err_obj.get("message") or err_obj.get("error") or str(err_obj)
                        err_msg = str(raw_msg)
                    else:
                        err_msg = str(err_obj)
                    lower_msg = err_msg.lower()
                    if err_code == 429 or "rate limit" in lower_msg:
                        raise RateLimitError(f"rate_limited_payload: code={err_code} msg={err_msg[:200]}")
                    raise OpenRouterError(f"API error payload: code={err_code} msg={err_msg[:200]}")

                # Some upstream errors return a 200 with an empty completion object.
                choices = parsed.get("choices")
                if choices is None or (isinstance(choices, list) and len(choices) == 0):
                    raise OpenRouterError("Malformed completion payload: missing/empty choices")
            return parsed
    except asyncio.TimeoutError as exc:
        raise OpenRouterError(f"Request timed out after {timeout_s}s") from exc
    except aiohttp.ClientError as exc:
        raise OpenRouterError(f"HTTP client error: {exc}") from exc


def parse_first_int(text: str) -> Optional[int]:
    if not text:
        return None
    raw = text.strip()

    # Best case: clean numeric output.
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)

    # Accept simple JSON wrappers.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            candidate = parsed.get("answer")
            if isinstance(candidate, int):
                return candidate
            if isinstance(candidate, str) and re.fullmatch(r"-?\d+", candidate.strip()):
                return int(candidate.strip())
        if isinstance(parsed, int):
            return parsed
    except Exception:
        pass

    # Prefer final answer-like line near end.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in reversed(lines[-12:]):
        m = re.search(r"(?i)(?:final\\s*answer|answer|total)\\D*(-?\\d+)", ln)
        if m:
            return int(m.group(1))
        if re.fullmatch(r"-?\\d+", ln):
            return int(ln)

    # Fallback: choose the last integer, not the first.
    all_nums = INT_RE.findall(raw)
    return int(all_nums[-1]) if all_nums else None


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


def extract_usage_counts(usage: Any) -> Tuple[int, int, int]:
    def _get_first(d: Dict[str, Any], keys: List[str]) -> int:
        for key in keys:
            val = d.get(key)
            if isinstance(val, int) and val >= 0:
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
        return 0

    if isinstance(usage, dict):
        prompt = _get_first(usage, ["prompt_tokens", "input_tokens", "input_token_count"])
        completion = _get_first(usage, ["completion_tokens", "output_tokens", "output_token_count"])
        total = _get_first(usage, ["total_tokens", "token_count"])
        if total == 0 and (prompt or completion):
            total = prompt + completion
        return prompt, completion, total
    return 0, 0, 0


def extract_reasoning_tokens(usage: Any) -> int:
    """Best-effort extraction of reasoning-token counts across providers.

    OpenRouter may surface reasoning usage as a top-level field (e.g., reasoning_tokens)
    or nested under completion/output token details.
    """
    if not isinstance(usage, dict):
        return 0

    def _coerce_int(val: Any) -> Optional[int]:
        if isinstance(val, int) and val >= 0:
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
        return None

    for key in ("reasoning_tokens", "reasoning_token_count"):
        v = _coerce_int(usage.get(key))
        if v is not None:
            return v

    # OpenAI-style details blocks
    for detail_key in (
        "completion_tokens_details",
        "output_tokens_details",
        "completion_details",
        "output_details",
        "token_details",
        "details",
    ):
        detail = usage.get(detail_key)
        if isinstance(detail, dict):
            for key in ("reasoning_tokens", "reasoning_token_count"):
                v = _coerce_int(detail.get(key))
                if v is not None:
                    return v

    return 0


def sanitize_model_slug(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model.strip())
    slug = slug.strip("_.") or "model"
    return slug


def build_run_slug(model: str, reasoning_effort: Optional[str]) -> str:
    base = sanitize_model_slug(model)
    if reasoning_effort:
        return f"{base}__reasoning+{reasoning_effort}"
    return base


def build_model_label(model: str, reasoning_effort: Optional[str]) -> str:
    if reasoning_effort:
        return f"{model} ({reasoning_effort} reasoning)"
    return model


def short_model_name(model_id: str) -> str:
    """Strip provider prefixes and return the bare model name."""
    raw = (model_id or "").strip()
    if not raw:
        return raw
    raw = raw.split(" (", 1)[0]
    lowered = raw.lower()
    if lowered.startswith("openrouter/"):
        raw = raw[len("openrouter/") :]
    parts = [p for p in raw.replace("\\", "/").split("/") if p]
    return parts[-1] if parts else raw


def build_benchmark_label(model: str, reasoning_effort: Optional[str]) -> str:
    """Build the display label used by graph rendering."""
    display = short_model_name(model)
    if reasoning_effort:
        display = f"{display} ({reasoning_effort} reasoning)"
    return display


KNOWN_PROVIDERS = {"openrouter"}
DEFAULT_PROVIDER = "openrouter"


def normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    """Normalize reasoning effort value."""
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"none", "off", "false", "0", "no"}:
        return "none"
    if v in {"minimal", "low", "medium", "high", "xhigh"}:
        return v
    raise ValueError(
        f"Invalid reasoning effort '{value}' "
        f"(expected minimal|low|medium|high|xhigh|none)"
    )


def parse_model_entry(raw: str, default_reasoning: Optional[str]) -> Tuple[str, str, Optional[str]]:
    """Parse model string with optional provider and reasoning override.

    Supported formats:
        - 'openai/o3' -> (openrouter, openai/o3, default_reasoning)
        - 'openrouter:openai/o4-mini:reasoning=high' -> (openrouter, openai/o4-mini, high)

    Returns:
        Tuple of (provider, model, reasoning_effort)
    """
    provider = DEFAULT_PROVIDER
    model = raw
    reasoning = normalize_reasoning_effort(default_reasoning)

    parts = raw.split(":")
    remaining_parts = parts

    # Check if first part is a known provider
    if len(parts) >= 2 and parts[0].lower() in KNOWN_PROVIDERS:
        provider = parts[0].lower()
        remaining_parts = parts[1:]
    elif len(parts) >= 2 and parts[0].lower() == "google":
        raise ValueError("Google provider has been removed. Use OpenRouter model IDs only.")

    # Rejoin remaining parts and look for reasoning= suffix
    rejoined = ":".join(remaining_parts)

    # Check for reasoning= in the last part
    if remaining_parts and "=" in remaining_parts[-1]:
        last_part = remaining_parts[-1]
        if last_part.lower().startswith("reasoning="):
            # Extract reasoning value
            val = last_part.split("=", 1)[1].strip().lower()
            reasoning = normalize_reasoning_effort(val)
            # Model is everything except the last part
            model = ":".join(remaining_parts[:-1])
        else:
            model = rejoined
    else:
        model = rejoined

    model = model.strip() or "model"
    return provider, model, reasoning


def aggregate_token_usage(results: List[ItemResult]) -> Dict[str, int]:
    prompt = sum(r.prompt_tokens for r in results)
    completion = sum(r.completion_tokens for r in results)
    total = sum(r.total_tokens for r in results)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def lookup_price_override(model: str, slug: str) -> Optional[Tuple[float, float]]:
    if not MODEL_PRICE_FILE.exists():
        return None
    try:
        data = json.loads(MODEL_PRICE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read %s: %s", MODEL_PRICE_FILE, exc)
        return None

    def _coerce_number(val: Any) -> Optional[float]:
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
            prompt = _coerce_number(entry.get("input") or entry.get("prompt") or entry.get("prompt_per_million"))
            completion = _coerce_number(entry.get("output") or entry.get("completion") or entry.get("completion_per_million"))
            if prompt is None and completion is None:
                return None
            prompt_val = prompt if prompt is not None else 0.0
            completion_val = completion if completion is not None else 0.0
            return prompt_val, completion_val
        coerced = _coerce_number(entry)
        if coerced is not None:
            return coerced, coerced
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            first = _coerce_number(entry[0])
            second = _coerce_number(entry[1])
            if first is not None or second is not None:
                return first or 0.0, second or 0.0
        return None

    for key in (model, slug):
        if key in data:
            coerced_pair = _coerce_price_entry(data[key])
            if coerced_pair is not None:
                return coerced_pair
    return None


def load_price_info_local(model: str, run_slug: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Load price info from local sources (model_prices.json or previous summaries)."""
    base_slug = sanitize_model_slug(model)

    override = lookup_price_override(model, base_slug)
    if override is not None:
        return override

    slug_candidates = []
    if run_slug and run_slug != base_slug:
        slug_candidates.append(run_slug)
    slug_candidates.append(base_slug)

    for slug in slug_candidates:
        prev_summary = RUNS_DIR / slug / "summary.json"
        if prev_summary.exists():
            try:
                doc = json.loads(prev_summary.read_text(encoding="utf-8"))
                price_obj = doc.get("price", {}) if isinstance(doc, dict) else {}
                prompt_val = price_obj.get("input_per_million")
                completion_val = price_obj.get("output_per_million")
                if isinstance(prompt_val, (int, float)) or isinstance(completion_val, (int, float)):
                    return (
                        float(prompt_val) if isinstance(prompt_val, (int, float)) else 0.0,
                        float(completion_val) if isinstance(completion_val, (int, float)) else 0.0,
                    )
                legacy = price_obj.get("per_million")
                if isinstance(legacy, (int, float)):
                    val = float(legacy)
                    return val, val
            except Exception as exc:
                logger.debug("Unable to parse previous summary for price: %s", exc)

    return None


async def load_price_info_async(
    model: str,
    run_slug: Optional[str],
    session: aiohttp.ClientSession,
    models_url: str,
    headers: Dict[str, str],
) -> Tuple[float, float]:
    """Load price info, falling back to OpenRouter /models when needed."""
    local = load_price_info_local(model, run_slug=run_slug)
    if local is not None:
        return local

    fetched = await fetch_openrouter_price_pair(
        session=session,
        models_url=models_url,
        headers=headers,
        model=model,
    )
    if fetched is not None:
        return fetched
    return DEFAULT_PROMPT_PRICE, DEFAULT_COMPLETION_PRICE


def write_model_summary(
    model: str,
    model_label: str,
    run_slug: str,
    reasoning_effort: Optional[str],
    args: argparse.Namespace,
    results: List[ItemResult],
    metrics: Dict[str, Any],
    token_usage: Dict[str, int],
    price_info: Tuple[float, float],
    estimated_cost: float,
) -> Path:
    out_dir = RUNS_DIR / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_price, completion_price = price_info
    summary_effort_raw = metrics.get("effective_reasoning_effort")
    summary_effort = summary_effort_raw if isinstance(summary_effort_raw, str) and summary_effort_raw else None
    benchmark_label = build_benchmark_label(model, summary_effort)
    payload = {
        "model": model,
        "model_label": model_label,
        "benchmark_label": benchmark_label,
        "run_slug": run_slug,
        "base_url": args.base_url,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "params": {
            "n": metrics["n"],
            "concurrency": args.concurrency,
            "max_retries": args.max_retries,
            "reasoning_effort": reasoning_effort,
            "model_delay": args.model_delay,
        },
        "summary": metrics,
        "token_usage": token_usage,
        "price": {
            "input_per_million": prompt_price,
            "output_per_million": completion_price,
            "estimated_cost": estimated_cost,
        },
        "items": [
            {
                "index": r.index,
                "truth": r.truth,
                "pred": r.pred,
                "correct": r.correct,
                "latency_s": r.latency_s,
                "error": r.error,
                "llm_output": r.llm_output,
            }
            for r in sorted(results, key=lambda x: x.index)
        ],
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


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
    rate_limit_backoff: float,
    reasoning_effort: Optional[str],
    dashboard: Optional[RunDashboard] = None,
    model_idx: Optional[int] = None,
) -> ItemResult:
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            choice0 = None
            logger.debug("Item %02d attempt %d starting", idx, attempt + 1)
            async with semaphore:
                if dashboard is not None and model_idx is not None:
                    dashboard.mark_item_in_flight(model_idx, idx, attempt + 1)
                try:
                    t0 = time.perf_counter()

                    payload = {
                        "model": model,
                        "messages": build_messages(data_uri),
                        "temperature": 0,
                    }
                    if reasoning_effort is not None:
                        payload["reasoning"] = {"effort": reasoning_effort, "exclude": True}
                        # Legacy flag retained for broader provider compatibility.
                        payload["include_reasoning"] = False
                    resp = await openrouter_completion(
                        session=session,
                        api_url=api_url,
                        headers=headers,
                        payload=payload,
                        timeout_s=request_timeout_s,
                        reasoning_requested=bool(reasoning_effort and reasoning_effort != "none"),
                    )
                    t1 = time.perf_counter()
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Item %02d raw response: %s", idx, safe_preview(resp))
                    content = extract_text(resp)
                    if not content:
                        # Treat empty assistant text as transient API/gateway failure so retries apply.
                        raise OpenRouterError("No textual assistant content in completion payload")
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
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0
                    reasoning_tokens = 0
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
                        if isinstance(usage, dict):
                            prompt_tokens, completion_tokens, total_tokens = extract_usage_counts(usage)
                            reasoning_tokens = extract_reasoning_tokens(usage)
                        # content shape
                        msg_content = None
                        if msg is not None:
                            msg_content = getattr(msg, "content", None)
                            if msg_content is None and isinstance(msg, dict):
                                msg_content = msg.get("content")
                        content_type = type(msg_content).__name__ if msg_content is not None else "None"
                        content_len = (len(msg_content) if isinstance(msg_content, str) else (len(msg_content) if isinstance(msg_content, list) else 0))
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
                            f"reasoning_tokens={reasoning_tokens} "
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
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        reasoning_tokens=reasoning_tokens,
                        llm_output=content,
                    )

                    if logger.isEnabledFor(logging.DEBUG):
                        summary_parts = [
                            f"pred={item.pred}",
                            f"truth={item.truth}",
                            f"error={item.error or 'none'}",
                            f"latency_s={item.latency_s:.3f}",
                        ]
                        if item.debug:
                            summary_parts.append(item.debug)
                        logger.debug("Item %02d result: %s", idx, " | ".join(summary_parts))
                    if logger.isEnabledFor(logging.DEBUG):
                        if content:
                            logger.debug("Item %02d content preview: %s", idx, preview_text(content, limit=800))
                        if choice0 is not None:
                            logger.debug("Item %02d choice0 preview: %s", idx, safe_preview(choice0))

                    return item
                finally:
                    if dashboard is not None and model_idx is not None:
                        dashboard.clear_item_in_flight(model_idx, idx)
        except UnsupportedReasoningModelError as e:
            last_err = f"reasoning_not_supported: {e}"
            logger.error(
                "Item %02d reasoning unsupported by model '%s': %s (disable --reasoning-effort or choose a reasoning-capable model)",
                idx,
                model,
                e,
            )
            return ItemResult(index=idx, truth=truth, pred=None, correct=False, latency_s=0.0, error=last_err)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if dashboard is not None and model_idx is not None:
                dashboard.on_retry(model_idx, idx, attempt + 1, last_err)
            delay: float
            if isinstance(e, RateLimitError):
                delay = rate_limit_backoff * (attempt + 1)
                logger.debug(
                    "Item %02d attempt %d hit rate limit; sleeping %.2fs (%s)",
                    idx,
                    attempt + 1,
                    delay,
                    last_err,
                )
            else:
                delay = 0.4 * (2**attempt)
                logger.debug("Item %02d attempt %d failed with %s", idx, attempt + 1, last_err)
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


async def run_model_once_openrouter(
    model: str,
    reasoning_effort: Optional[str],
    args: argparse.Namespace,
    truths: Dict[int, int],
    img_indices: List[int],
    data_uris: Dict[int, str],
    api_url: str,
    headers: Dict[str, str],
    dashboard: Optional[RunDashboard] = None,
    model_idx: Optional[int] = None,
) -> None:
    """Run benchmark for a single model using OpenRouter API."""
    run_slug = build_run_slug(model, reasoning_effort)
    model_label = build_model_label(model, reasoning_effort)
    sem = asyncio.Semaphore(args.concurrency)
    models_url = build_models_url(args.base_url)
    price_info: Tuple[float, float]

    t_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        price_info = await load_price_info_async(
            model=model,
            run_slug=run_slug,
            session=session,
            models_url=models_url,
            headers=headers,
        )
        tasks = [
            asyncio.create_task(
                eval_one(
                    idx=i,
                    model=model,
                    data_uri=data_uris[i],
                    truth=truths.get(i),
                    semaphore=sem,
                    session=session,
                    api_url=api_url,
                    headers=headers,
                    request_timeout_s=args.request_timeout,
                    max_retries=args.max_retries,
                    rate_limit_backoff=args.rate_limit_backoff,
                    reasoning_effort=reasoning_effort,
                    dashboard=dashboard,
                    model_idx=model_idx,
                )
            )
            for i in img_indices
        ]

        results: List[ItemResult] = []
        completed = 0
        total = len(tasks)
        if dashboard is not None and model_idx is not None:
            dashboard.start_model(model_idx, total)
        if args.progress and dashboard is None:
            print(f"[progress] {completed}/{total} completed", flush=True)

        for finished in asyncio.as_completed(tasks):
            res = await finished
            results.append(res)
            completed += 1
            if dashboard is not None and model_idx is not None:
                dashboard.on_item(model_idx, res)
            if args.progress and dashboard is None:
                print(
                    f"[progress] {completed}/{total} completed (last={res.index:02d})",
                    flush=True,
                )
    t_end = time.perf_counter()

    # Per-item lines only when explicitly requested.
    if args.progress and dashboard is None:
        for r in sorted(results, key=lambda x: x.index):
            print(
                f"{r.index:02d}.png -> pred={r.pred} truth={r.truth} "
                f"correct={int(r.correct)} latency_s={r.latency_s:.3f}"
                + (f" error={r.error}" if r.error else "")
            )
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
    token_usage = aggregate_token_usage(results)
    input_price, output_price = price_info
    reasoning_tokens_total = sum(r.reasoning_tokens for r in results)
    reasoning_observed = reasoning_tokens_total > 0
    effective_reasoning_effort = reasoning_effort or ("medium" if reasoning_observed else None)
    estimated_cost = (
        (token_usage["prompt_tokens"] / 1_000_000.0) * input_price
        + (token_usage["completion_tokens"] / 1_000_000.0) * output_price
    ) if (input_price or output_price) else 0.0
    metrics = {
        "n": total,
        "total_time_s": total_time,
        "throughput_ips": throughput,
        "accuracy_exact": acc,
        "mae": mae,
        "latency_avg_s": avg_lat,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "failures": failures,
        "percent_correct": acc * 100.0,
        "num_correct": num_correct,
        "estimated_cost": estimated_cost,
        "reasoning_observed": reasoning_observed,
        "reasoning_tokens_total": reasoning_tokens_total,
        "effective_reasoning_effort": effective_reasoning_effort,
    }
    if dashboard is None:
        print("\n--- summary ---")
        print(f"model={model}")
        if args.base_url:
            print(f"base_url={args.base_url}")
        print(f"model_label={model_label}")
        if reasoning_effort:
            print(f"reasoning_effort={reasoning_effort}")
        elif effective_reasoning_effort:
            print(f"reasoning_effort={effective_reasoning_effort} (implicit)")
        print(f"n={len(results)} concurrency={args.concurrency} total_time_s={total_time:.3f} throughput_ips={throughput:.2f}")
        print(f"accuracy_exact={acc:.4f}  mae={mae:.4f}")
        print(f"latency_avg_s={avg_lat:.3f}  p50_s={p50:.3f}  p95_s={p95:.3f}")
        print(f"failures={failures}")
        if total > 0:
            print(f"percent_correct={acc * 100:.1f}% ({num_correct}/{total})")
        print(
            "token_usage="
            f"prompt={token_usage['prompt_tokens']} completion={token_usage['completion_tokens']} total={token_usage['total_tokens']}"
        )
        if input_price or output_price:
            print(
                "estimated_cost=${:.4f} (input_per_million={} output_per_million={})".format(
                    estimated_cost,
                    input_price,
                    output_price,
                )
            )

    summary_path = write_model_summary(
        model=model,
        model_label=model_label,
        run_slug=run_slug,
        reasoning_effort=reasoning_effort,
        args=args,
        results=results,
        metrics=metrics,
        token_usage=token_usage,
        price_info=price_info,
        estimated_cost=estimated_cost,
    )
    if dashboard is not None and model_idx is not None:
        dashboard.finish_model(model_idx, failed=bool(failures), summary_path=str(summary_path), metrics=metrics)
    else:
        print(f"summary_saved={summary_path}")


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

    img_indices = list(range(1, n + 1))
    data_uris: Dict[int, str] = {}
    for i in img_indices:
        p = imgs_dir / f"{i}.png"
        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")
        data_uris[i] = to_data_uri(p)

    default_reasoning = normalize_reasoning_effort(args.reasoning_effort)
    raw_models = args.models if args.models else [args.model]
    # model_specs: List of (provider, model, reasoning_effort)
    model_specs: List[Tuple[str, str, Optional[str]]] = []
    try:
        for raw in raw_models:
            provider, model_name, reasoning_override = parse_model_entry(raw, default_reasoning)
            model_specs.append((provider, model_name, reasoning_override))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set")
    api_url = build_api_url(args.base_url)
    headers = build_openrouter_headers(openrouter_api_key)

    if args.models and not args.tui:
        def fmt(p: str, m: str, r: Optional[str]) -> str:
            reasoning_str = f" ({r} reasoning)" if r else ""
            return f"{p}:{m}{reasoning_str}"

        desc = ", ".join([fmt(p, m, r) for p, m, r in model_specs])
        print(f"Queued models ({len(model_specs)}): {desc}")

    dashboard: Optional[RunDashboard] = None
    dashboard_stop_event: Optional[asyncio.Event] = None
    dashboard_task: Optional[asyncio.Task] = None
    prev_ibench_level: Optional[int] = None
    if args.tui and len(model_specs) > 0:
        dashboard = RunDashboard(
            model_specs,
            items_per_model=len(img_indices),
            concurrency=args.concurrency,
            use_color=args.color,
            verbose=args.verbose,
            force_live=False,
        )
        # Keep terminal output dedicated to the live dashboard.
        prev_ibench_level = logging.getLogger("ibench").level
        logging.getLogger("ibench").setLevel(logging.CRITICAL)
        dashboard.render()
        dashboard_stop_event = asyncio.Event()

        async def _dashboard_tick() -> None:
            # Keep elapsed timers moving even when no item finishes.
            while dashboard_stop_event is not None and not dashboard_stop_event.is_set():
                await asyncio.sleep(1.0)
                if dashboard is not None:
                    dashboard._mark_dirty()
                    dashboard.render()

        dashboard_task = asyncio.create_task(_dashboard_tick())

    try:
        for idx, (provider, model, reasoning_effort) in enumerate(model_specs):
            if idx > 0 and args.model_delay > 0:
                delay = args.model_delay
                if dashboard is not None:
                    dashboard.set_waiting(idx, delay)
                    remaining = delay
                    while remaining > 0:
                        await asyncio.sleep(min(1.0, remaining))
                        remaining -= 1.0
                        dashboard.render()
                else:
                    print(f"Waiting {delay:.0f}s before next model...", flush=True)
                    await asyncio.sleep(delay)

            run_label = f"{provider}:{model}"
            if reasoning_effort:
                run_label += f" ({reasoning_effort} reasoning)"
            if dashboard is None:
                print(f"\n=== Running model {run_label} ({idx + 1}/{len(model_specs)}) ===")

            await run_model_once_openrouter(
                model=model,
                reasoning_effort=reasoning_effort,
                args=args,
                truths=truths,
                img_indices=img_indices,
                data_uris=data_uris,
                api_url=api_url,
                headers=headers,
                dashboard=dashboard,
                model_idx=idx,
            )
    finally:
        if dashboard is not None:
            dashboard.finish_all()
        if prev_ibench_level is not None:
            logging.getLogger("ibench").setLevel(prev_ibench_level)
        if dashboard_stop_event is not None:
            dashboard_stop_event.set()
        if dashboard_task is not None:
            dashboard_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await dashboard_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision model benchmark for counting line intersections")
    p.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-vl-235b-a22b-instruct",
        help="Model name with optional openrouter prefix (default: qwen/qwen3-vl-235b-a22b-instruct). "
             "Format: [openrouter:]model[:reasoning=LEVEL]. Examples: 'openai/gpt-4o', 'openrouter:openai/o3:reasoning=high'.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Queue multiple OpenRouter models to run sequentially. "
             "Example: --models openai/gpt-5 openai/gpt-5.1 openrouter:openai/o3:reasoning=high",
    )
    p.add_argument("--imgs", type=str, default="public/imgs", help="Directory containing images named 1.png..N.png (default: ./public/imgs)")
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
        "--model-delay",
        type=float,
        default=60.0,
        help="Seconds to wait between sequential models when --models is used (default: 60)",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=OPENROUTER_DEFAULT_BASE,
        help="OpenRouter-compatible base URL (default: https://openrouter.ai/api/v1)",
    )
    p.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["minimal", "low", "medium", "high", "xhigh", "none"],
        default=None,
        help="Default reasoning effort for all models (minimal|low|medium|high|xhigh|none). Use none to disable.",
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
    p.add_argument(
        "--progress",
        action="store_true",
        help="Print streaming progress updates as items complete",
    )
    p.add_argument(
        "--no-tui",
        dest="tui",
        action="store_false",
        help="Disable live terminal dashboard and print classic logs",
    )
    p.add_argument(
        "--no-color",
        dest="color",
        action="store_false",
        help="Disable ANSI colors in terminal output",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show expanded live per-question telemetry in the TUI",
    )
    p.set_defaults(tui=True)
    p.set_defaults(color=True)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        # Clean silent exit on Ctrl-C.
        pass


if __name__ == "__main__":
    main()
