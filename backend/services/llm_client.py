"""Shared Gemini API client with exponential-backoff retry and persistent JSON cache.

All three LLM modules (outreach_generator, risk_tier, promise_extractor) import
from here.  No other module should call the Gemini API directly.

Retry policy
------------
429 (rate-limit) and any 5xx → wait 2 s then 4 s then 8 s, then give up and
raise so the caller can fall back to its deterministic template.  Every retry
and every final fallback is logged at WARNING level so it's visible in test
output, not silent.

Cache policy
------------
A plain JSON file per use-case under cache/ (gitignored, persists across runs).
Key is always a tuple serialised as a JSON-array string.  Cache is read before
every Gemini call and written after every successful (non-fallback) response.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))


import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)

# ----- Gemini constants -------------------------------------------------------

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-flash-latest"

# ----- Retry policy -----------------------------------------------------------

_RETRY_DELAYS = (2.0, 4.0, 8.0)  # seconds between attempts


def post_gemini_with_retry(
    api_key: str,
    payload: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """POST to Gemini generateContent, retrying on 429/5xx with exponential backoff.

    Raises the last HTTPError / URLError / ValueError on exhaustion so the caller
    can fall back to its deterministic template.  Never swallows the error silently.
    """
    url = f"{GEMINI_BASE_URL}/{model}:generateContent"
    body = json.dumps(payload).encode("utf-8")
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    last_exc: Exception | None = None
    attempts = len(_RETRY_DELAYS) + 1  # initial attempt + retries

    for attempt in range(attempts):
        req = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            last_exc = exc
            if exc.code in (429,) or 500 <= exc.code < 600:
                if attempt < len(_RETRY_DELAYS):
                    delay = _RETRY_DELAYS[attempt]
                    LOGGER.warning(
                        "Gemini API HTTP %s on attempt %d/%d; retrying in %.0f s",
                        exc.code,
                        attempt + 1,
                        attempts,
                        delay,
                    )
                    time.sleep(delay)
                    continue
            # Non-retryable HTTP error (4xx other than 429): raise immediately
            LOGGER.warning(
                "Gemini API non-retryable HTTP %s: %s; falling back to template",
                exc.code,
                exc.reason,
            )
            raise
        except (URLError, json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            if attempt < len(_RETRY_DELAYS):
                delay = _RETRY_DELAYS[attempt]
                LOGGER.warning(
                    "Gemini API transient error on attempt %d/%d (%s); retrying in %.0f s",
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                LOGGER.warning(
                    "Gemini API error after %d attempts: %s; falling back to template",
                    attempts,
                    exc,
                )
                raise

    # Exhausted all retries
    assert last_exc is not None
    LOGGER.warning(
        "Gemini API exhausted %d retries; falling back to template. Last error: %s",
        len(_RETRY_DELAYS),
        last_exc,
    )
    raise last_exc


# ----- Persistent JSON cache --------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


def _cache_path(name: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"{name}.json"


def _key_str(key: tuple[Any, ...]) -> str:
    return json.dumps(key, sort_keys=True, ensure_ascii=True)


def read_cache(name: str, key: tuple[Any, ...]) -> Any | None:
    """Return cached value or None if absent/unreadable."""
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        return data.get(_key_str(key))
    except (OSError, json.JSONDecodeError):
        return None


def write_cache(name: str, key: tuple[Any, ...], value: Any) -> None:
    """Persist a value to the named cache file.  Silently ignores write errors."""
    path = _cache_path(name)
    try:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data: dict[str, Any] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            data = {}
        data[_key_str(key)] = value
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=True, indent=2)
    except OSError as exc:
        LOGGER.warning("Cache write failed (%s): %s", path, exc)
