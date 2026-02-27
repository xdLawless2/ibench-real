#!/usr/bin/env python3
"""
Google Gemini API provider for ibench.

Supports direct calls to Google's Gemini API using the google-genai SDK.
"""

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger("ibench.google")

INT_RE = re.compile(r"\d+")

QUESTION = "How many distinct intersections between different shapes are in this image? Count each distinct crossing point once. Reply with ONLY a number and nothing else."

SYSTEM_MSG = (
    "You are a precise vision assistant. For the given image, return ONLY a single "
    "non-negative INTEGER: the count of distinct intersections formed by different shapes. "
    "No words, no punctuation, no preamble, just the number."
)


class GoogleProviderError(RuntimeError):
    """Raised when the Google API returns an error."""


class UnsupportedReasoningError(GoogleProviderError):
    """Raised when an invalid reasoning effort is specified for a model."""


@dataclass
class ItemResult:
    """Result from evaluating a single image."""
    index: int
    truth: Optional[int]
    pred: Optional[int]
    correct: bool
    latency_s: float
    error: Optional[str] = None
    debug: Optional[str] = None
    raw_text_preview: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_len: int = 0
    thinking_blocks: int = 0
    raw_response: Optional[str] = None


def check_google_genai_available() -> None:
    """Raise an error if google-genai is not installed."""
    if genai is None:
        raise ImportError(
            "google-genai package is required for Google provider. "
            "Install it with: pip install google-genai"
        )


def is_gemini_3(model: str) -> bool:
    """Check if model is a Gemini 3 family model."""
    lower = model.lower()
    return "gemini-3" in lower or "gemini3" in lower


def is_gemini_25(model: str) -> bool:
    """Check if model is a Gemini 2.5 family model."""
    lower = model.lower()
    return "gemini-2.5" in lower or "gemini-2-5" in lower or "gemini2.5" in lower


def validate_reasoning_effort(model: str, reasoning_effort: Optional[str]) -> None:
    """Validate reasoning effort is compatible with the model.

    Gemini 3 only supports 'low' or 'high' thinking levels.
    Gemini 2.5 ignores reasoning effort (uses dynamic thinking by default).
    """
    if reasoning_effort is None:
        return

    if is_gemini_3(model):
        if reasoning_effort not in ("low", "high"):
            raise UnsupportedReasoningError(
                f"Gemini 3 models only support 'low' or 'high' reasoning effort, "
                f"got '{reasoning_effort}'. Use --reasoning-effort low or --reasoning-effort high"
            )


def build_thinking_config(model: str, reasoning_effort: Optional[str]) -> Optional[Any]:
    """Build the appropriate thinking config based on model and reasoning effort.

    - Gemini 2.5: Don't set thinking_budget (let Google decide with dynamic thinking)
    - Gemini 3: Use thinking_level if reasoning_effort is specified
    """
    if types is None:
        return None

    if is_gemini_3(model) and reasoning_effort:
        # Gemini 3 uses thinking_level: "low" or "high"
        return types.ThinkingConfig(thinking_level=reasoning_effort)

    # For Gemini 2.5 and others, don't set thinking config (use defaults)
    return None


def parse_first_int(text: str) -> Optional[int]:
    """Extract the first integer from text."""
    if not text:
        return None
    m = INT_RE.search(text)
    return int(m.group()) if m else None


def extract_text_from_response(response: Any) -> str:
    """Extract text content from a Gemini API response."""
    try:
        # Try the simple .text property first
        if hasattr(response, 'text') and response.text:
            return response.text

        # Fall back to parsing candidates
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts') and content.parts:
                    texts = []
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            texts.append(part.text)
                    return "\n".join(texts)

        return ""
    except Exception as e:
        logger.debug("Error extracting text from response: %s", e)
        return ""


def extract_usage_from_response(response: Any) -> Tuple[int, int, int, int]:
    """Extract token usage from a Gemini API response.

    Returns: (prompt_tokens, completion_tokens, total_tokens, thinking_tokens)
    """
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    thinking_tokens = 0

    try:
        usage = getattr(response, 'usage_metadata', None)
        if usage is None and hasattr(response, 'usage'):
            usage = response.usage

        if usage:
            prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
            completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
            total_tokens = getattr(usage, 'total_token_count', 0) or 0
            thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0

            # Fallback field names
            if prompt_tokens == 0:
                prompt_tokens = getattr(usage, 'input_tokens', 0) or 0
            if completion_tokens == 0:
                completion_tokens = getattr(usage, 'output_tokens', 0) or 0
            if total_tokens == 0 and (prompt_tokens or completion_tokens):
                total_tokens = prompt_tokens + completion_tokens
    except Exception as e:
        logger.debug("Error extracting usage from response: %s", e)

    return prompt_tokens, completion_tokens, total_tokens, thinking_tokens


def serialize_raw_response(response: Any) -> str:
    """Best-effort full raw response serialization for run summaries."""
    try:
        if isinstance(response, (dict, list)):
            return json.dumps(response, ensure_ascii=False)
        if hasattr(response, "model_dump_json"):
            return response.model_dump_json(exclude_none=False)
        if hasattr(response, "to_json"):
            as_json = response.to_json()
            if isinstance(as_json, str):
                return as_json
        if hasattr(response, "__dict__"):
            return json.dumps(response.__dict__, default=str, ensure_ascii=False)
    except Exception:
        pass
    return repr(response)


def base64_data_uri_to_bytes(data_uri: str) -> Tuple[bytes, str]:
    """Convert a base64 data URI to raw bytes and mime type.

    Args:
        data_uri: A data URI like "data:image/png;base64,..."

    Returns:
        Tuple of (image_bytes, mime_type)
    """
    # Parse data URI: data:image/png;base64,<data>
    if data_uri.startswith("data:"):
        header, b64_data = data_uri.split(",", 1)
        # Extract mime type from header like "data:image/png;base64"
        mime_part = header.split(";")[0]  # "data:image/png"
        mime_type = mime_part.replace("data:", "")  # "image/png"
    else:
        # Assume raw base64 PNG if no header
        b64_data = data_uri
        mime_type = "image/png"

    image_bytes = base64.b64decode(b64_data)
    return image_bytes, mime_type


class GoogleProvider:
    """Provider for Google's Gemini API."""

    def __init__(self, api_key: str, api_version: str = "v1beta"):
        """Initialize the Google provider.

        Args:
            api_key: The GEMINI_API_KEY for authentication.
            api_version: API version to use ("v1beta", "v1", etc.).
        """
        check_google_genai_available()
        # Use v1beta API (default for SDK, supports preview features)
        http_options = types.HttpOptions(api_version=api_version)
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.api_key = api_key
        self.api_version = api_version

    def _generate_content_sync(
        self,
        model: str,
        image_bytes: bytes,
        mime_type: str,
        reasoning_effort: Optional[str],
    ) -> Any:
        """Synchronous call to generate_content."""
        # Build the content with image
        # Use inline_data dict format which is more compatible
        image_part = {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(image_bytes).decode("ascii"),
            }
        }

        contents = [
            {"text": QUESTION},
            image_part,
        ]

        # Build config
        thinking_config = build_thinking_config(model, reasoning_effort)

        config_kwargs: Dict[str, Any] = {
            "system_instruction": SYSTEM_MSG,
            "temperature": 0,
        }
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config

        config = types.GenerateContentConfig(**config_kwargs)

        # Make the API call
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        return response

    async def eval_one(
        self,
        idx: int,
        model: str,
        data_uri: str,
        truth: Optional[int],
        reasoning_effort: Optional[str],
        max_retries: int = 5,
        rate_limit_backoff: float = 5.0,
    ) -> ItemResult:
        """Evaluate a single image using the Google Gemini API.

        Args:
            idx: Image index (1-based).
            model: Model name (e.g., "gemini-2.5-flash").
            data_uri: Base64 data URI of the image.
            truth: Ground truth intersection count.
            reasoning_effort: Optional reasoning effort level.
            max_retries: Maximum retry attempts.
            rate_limit_backoff: Base backoff time for rate limits.

        Returns:
            ItemResult with prediction and metadata.
        """
        # Validate reasoning effort for this model
        try:
            validate_reasoning_effort(model, reasoning_effort)
        except UnsupportedReasoningError as e:
            return ItemResult(
                index=idx,
                truth=truth,
                pred=None,
                correct=False,
                latency_s=0.0,
                error=f"unsupported_reasoning: {e}",
            )

        # Convert data URI to bytes
        try:
            image_bytes, mime_type = base64_data_uri_to_bytes(data_uri)
        except Exception as e:
            return ItemResult(
                index=idx,
                truth=truth,
                pred=None,
                correct=False,
                latency_s=0.0,
                error=f"image_decode_error: {e}",
            )

        last_err: Optional[str] = None

        for attempt in range(max_retries):
            try:
                logger.debug("Item %02d attempt %d starting (Google)", idx, attempt + 1)

                t0 = time.perf_counter()

                # Run sync call in thread pool to not block event loop
                response = await asyncio.to_thread(
                    self._generate_content_sync,
                    model,
                    image_bytes,
                    mime_type,
                    reasoning_effort,
                )

                t1 = time.perf_counter()
                latency = t1 - t0

                # Extract text and usage
                content = extract_text_from_response(response)
                prompt_tokens, completion_tokens, total_tokens, thinking_tokens = extract_usage_from_response(response)
                raw_response = serialize_raw_response(response)

                logger.debug("Item %02d extracted text: %s", idx, content[:200] if content else "(empty)")

                # Parse prediction
                pred = parse_first_int(content)

                # Classify errors
                err: Optional[str] = None
                if not content:
                    err = "empty_content"
                elif pred is None:
                    err = "no_digit"

                # Build debug string
                text_preview = (content[:160] + "..." if len(content) > 160 else content) if content else None
                debug_str = (
                    f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
                    f"thinking_tokens={thinking_tokens}"
                )

                correct = (pred == truth) if (pred is not None and truth is not None) else False

                result = ItemResult(
                    index=idx,
                    truth=truth,
                    pred=pred,
                    correct=correct,
                    latency_s=latency,
                    error=err,
                    debug=debug_str,
                    raw_text_preview=text_preview,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    reasoning_tokens=thinking_tokens,
                    reasoning_len=0,
                    thinking_blocks=0,
                    raw_response=raw_response,
                )

                if logger.isEnabledFor(logging.INFO):
                    summary_parts = [
                        f"pred={result.pred}",
                        f"truth={result.truth}",
                        f"error={result.error or 'none'}",
                        f"latency_s={result.latency_s:.3f}",
                    ]
                    if result.debug:
                        summary_parts.append(result.debug)
                    logger.info("Item %02d result (Google): %s", idx, " | ".join(summary_parts))

                return result

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                err_str = str(e).lower()

                # Check for rate limit errors
                is_rate_limit = any(x in err_str for x in ("429", "rate limit", "quota", "resource exhausted"))

                if is_rate_limit:
                    delay = rate_limit_backoff * (attempt + 1)
                    logger.warning(
                        "Item %02d attempt %d hit rate limit; sleeping %.2fs (%s)",
                        idx, attempt + 1, delay, last_err
                    )
                else:
                    delay = 0.4 * (2 ** attempt)
                    logger.warning("Item %02d attempt %d failed with %s", idx, attempt + 1, last_err)

                # Add jitter
                import random
                delay += random.uniform(0.0, 0.5)
                await asyncio.sleep(delay)

        logger.error("Item %02d failed after %d attempts: %s", idx, max_retries, last_err)
        return ItemResult(
            index=idx,
            truth=truth,
            pred=None,
            correct=False,
            latency_s=0.0,
            error=last_err,
        )


# Default Google model pricing (USD per 1M tokens) - as of 2025
# These are fallbacks; prefer model_prices.json
GOOGLE_MODEL_PRICES: Dict[str, Tuple[float, float]] = {
    # Gemini 2.5 Flash
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-flash-preview": (0.15, 0.60),
    # Gemini 2.5 Pro
    "gemini-2.5-pro": (1.25, 5.00),
    "gemini-2.5-pro-preview": (1.25, 5.00),
    # Gemini 2.5 Flash Lite
    "gemini-2.5-flash-lite": (0.075, 0.30),
    "gemini-2.5-flash-lite-preview": (0.075, 0.30),
    # Gemini 3 (placeholder - update when pricing is available)
    "gemini-3-pro": (2.50, 10.00),
    "gemini-3-pro-preview": (2.50, 10.00),
}


def get_google_model_price(model: str) -> Tuple[float, float]:
    """Get default pricing for a Google model.

    Returns: (input_price_per_million, output_price_per_million)
    """
    model_lower = model.lower()

    # Try exact match first
    if model_lower in GOOGLE_MODEL_PRICES:
        return GOOGLE_MODEL_PRICES[model_lower]

    # Try partial match
    for key, price in GOOGLE_MODEL_PRICES.items():
        if key in model_lower or model_lower in key:
            return price

    # Default fallback
    return (0.0, 0.0)
