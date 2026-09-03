"""Outreach message generator — LLM-backed, with deterministic fallback.

Public API
----------
generate_outreach(invoice, debtor, action_type, prior_actions, prior_replies) -> str
    Single-item wrapper kept for backward compatibility (tests, API layer).
    Internally delegates to generate_outreach_batch.

generate_outreach_batch(items, batch_size=10) -> list[str]
    Accepts a list of (invoice, debtor, action_type, context_str) tuples.
    Groups them into batches of at most `batch_size`, sends ONE Gemini request
    per batch asking for a JSON array of exactly N strings in the same order.
    If a batch response cannot be parsed into exactly N strings the fallback
    template is used for every item in that batch — no partial or shifted
    results are acceptable because a shifted array would attach the wrong
    message to the wrong invoice.

LLM call details
----------------
URL:     https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
AUTH:    x-goog-api-key header
MODEL:   gemini-flash-latest (Google-maintained alias, not a dated version)
PAYLOAD: system_instruction / contents / generationConfig
PARSE:   candidates[0]["content"]["parts"][0]["text"]

Cache
-----
Key: (invoice_id, action_type, contact_count)
Stored in: cache/outreach.json (gitignored, survives process restarts)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from typing import Any, Final

from backend.models import ActionType, RiskTier
from backend.services.llm_client import (
    DEFAULT_MODEL,
    post_gemini_with_retry,
    read_cache,
    write_cache,
)

import os

LOGGER = logging.getLogger(__name__)

_CACHE_NAME: Final = "outreach"
MAX_WORDS: Final = 120


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_outreach(
    invoice: Any,
    debtor: Any,
    action_type: str | ActionType,
    prior_actions: list[Any] | None,
    prior_replies: list[Any] | None,
) -> str:
    """Single-invoice wrapper — kept for backward compatibility.

    All test mocks patch `_post_gemini_generate_content` on this module,
    which is re-exported via `generate_outreach_batch` → `_call_gemini_batch`.
    """
    action = _action_value(action_type)
    context = _build_context(prior_actions or [], prior_replies or [])
    results = generate_outreach_batch([(invoice, debtor, action, context)], batch_size=1)
    return results[0]


def generate_outreach_batch(
    items: list[tuple[Any, Any, str, str]],
    batch_size: int = 40,
) -> list[str]:
    """Generate outreach messages for a list of (invoice, debtor, action_type, context) tuples.

    Returns one message per input, in the same order.  Never returns fewer or
    more messages than requested — on any batch failure the deterministic
    fallback is used for every item in that batch.
    """
    results: list[str] = []
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]
        results.extend(_process_batch(batch))
    return results


# ---------------------------------------------------------------------------
# Internal — batch processing
# ---------------------------------------------------------------------------

def _process_batch(batch: list[tuple[Any, Any, str, str]]) -> list[str]:
    """Process one batch; fall back to templates for the whole batch on any error."""
    n = len(batch)

    # 1. Check cache for all items — if every item is cached, skip the API call.
    messages: list[str | None] = []
    uncached_indices: list[int] = []
    for i, (invoice, debtor, action, context) in enumerate(batch):
        cache_key = _cache_key(invoice, action)
        cached = read_cache(_CACHE_NAME, cache_key)
        if cached is not None:
            messages.append(str(cached))
        else:
            messages.append(None)
            uncached_indices.append(i)

    if not uncached_indices:
        # Every item was cached
        return [m for m in messages]  # type: ignore[return-value]

    # 2. Build uncached sub-batch
    uncached_batch = [batch[i] for i in uncached_indices]
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        llm_results = _call_gemini_batch(uncached_batch, api_key)
    else:
        LOGGER.warning(
            "GEMINI_API_KEY not set; using fallback templates for %d outreach items",
            len(uncached_batch),
        )
        llm_results = [_fallback_message(inv, deb, act, ctx) for inv, deb, act, ctx in uncached_batch]

    # 3. Merge cached + fresh results and populate cache
    for idx, (invoice, debtor, action, context) in zip(uncached_indices, uncached_batch):
        msg = llm_results[uncached_indices.index(idx)]
        messages[idx] = msg
        if api_key and not msg.startswith("Dear "):
            # Only cache LLM-generated messages, not fallback templates
            # (Heuristic: fallback always starts with "Dear"; LLM output varies)
            pass
        # Cache unconditionally — even fallback results avoid duplicate calls
        cache_key = _cache_key(invoice, action)
        write_cache(_CACHE_NAME, cache_key, msg)

    return [m for m in messages]  # type: ignore[return-value]


def _call_gemini_batch(
    batch: list[tuple[Any, Any, str, str]],
    api_key: str,
) -> list[str]:
    """Send ONE Gemini request for the entire batch; return exactly len(batch) strings.

    On any failure (HTTP error, parse error, wrong count) returns deterministic
    fallback templates for every item in the batch.
    """
    n = len(batch)
    model = os.getenv("GEMINI_OUTREACH_MODEL", DEFAULT_MODEL)

    system_prompt = (
        "You are a professional, firm-but-respectful B2B collections correspondent "
        "for an Indian SME. "
        "You will receive a JSON array of invoice contexts. "
        "For each item, write exactly one natural collections message. "
        "Use INR and Indian business language where natural. "
        "Never threaten, abuse, shame, or make legal claims. "
        "LOW risk = friendly reminder, MED risk = firm but understanding, "
        "HIGH risk = clear and direct about consequences while remaining professional. "
        "Return ONLY a valid JSON array of exactly "
        + str(n)
        + " plain-text strings, one per input, in the same order. "
        "Each string must be under 120 words. No markdown, no subject line, "
        "no JSON keys, no extra commentary — just the array."
    )

    inputs = []
    for invoice, debtor, action, context in batch:
        tier = _value(_field(invoice, "risk_tier"), "MED")
        amount = _amount(invoice)
        due_date = _format_date(_value(_field(invoice, "due_date"), "not available"))
        debtor_name = _value(
            _field(debtor, "name") if debtor is not None else None,
            _value(_field(invoice, "debtor_name"), "your team"),
        )
        action_guidance = {
            "REMINDER": "Give a simple nudge with the invoice amount and due date, and invite an update.",
            "FOLLOWUP": "Reference that no response was received, restate the amount, and ask for a specific commitment.",
            "NEGOTIATION": "Explicitly ask for a payment date and amount commitment; you may offer a short extension.",
            "ESCALATION": "Write a formal notice prior to escalation. Start with 'DRAFT - REQUIRES HUMAN APPROVAL:' and make the approval requirement clear.",
        }
        inputs.append({
            "debtor_name": debtor_name,
            "invoice_id": _value(_field(invoice, "id"), "the invoice"),
            "amount": amount,
            "due_date": due_date,
            "risk_tier": tier,
            "action_type": action,
            "action_guidance": action_guidance.get(action, ""),
            "prior_context": context or "No prior contact context is available.",
        })

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": json.dumps(inputs, ensure_ascii=True)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 200 * n,
        },
    }

    try:
        response = _post_gemini_generate_content(api_key, payload, model)
        raw_text = response["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list) or len(parsed) != n:
            raise ValueError(
                f"Expected JSON array of {n} strings, got {type(parsed).__name__} "
                f"with length {len(parsed) if isinstance(parsed, list) else 'N/A'}"
            )
        result = [_limit_words(_clean_plain_text(str(item))) for item in parsed]
        LOGGER.info("Gemini outreach batch: %d messages generated successfully", n)
        return result
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Gemini outreach batch of %d failed, using fallback for all items: %s",
            n,
            exc,
        )
        return [_fallback_message(inv, deb, act, ctx) for inv, deb, act, ctx in batch]


# ---------------------------------------------------------------------------
# Internal — helpers
# ---------------------------------------------------------------------------

def _post_gemini_generate_content(
    api_key: str,
    payload: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Thin wrapper so tests can patch this exact symbol."""
    return post_gemini_with_retry(api_key, payload, model)


def _cache_key(invoice: Any, action: str) -> tuple[str, str, int]:
    invoice_id = str(_value(_field(invoice, "id"), "unknown"))
    contact_count = int(_field(invoice, "contact_count") or 0)
    return (invoice_id, action, contact_count)


def _build_context(prior_actions: list[Any], prior_replies: list[Any]) -> str:
    entries: list[str] = []
    for action in prior_actions[-3:]:
        action_type = _value(_field(action, "type"), "contact")
        timestamp = _format_date(_value(_field(action, "timestamp"), ""))
        entries.append(f"Previous {action_type} on {timestamp}" if timestamp else f"Previous {action_type}")
    for reply in prior_replies[-3:]:
        text = _value(_field(reply, "raw_text"), "")
        timestamp = _format_date(_value(_field(reply, "timestamp"), ""))
        if text:
            text = " ".join(str(text).split())[:240]
            entries.append(f"Debtor reply on {timestamp}: {text}" if timestamp else f"Debtor reply: {text}")
    return "; ".join(entries)


def _fallback_message(invoice: Any, debtor: Any, action: str, context: str) -> str:
    name = _value(
        _field(debtor, "name") if debtor is not None else None,
        _value(_field(invoice, "debtor_name"), "Team"),
    )
    invoice_id = _value(_field(invoice, "id"), "your invoice")
    amount = _amount(invoice)
    due_date = _format_date(_value(_field(invoice, "due_date"), "the due date"))
    tier = _value(_field(invoice, "risk_tier"), "MED")
    prior_date = re.search(r"on (\d{4}-\d{2}-\d{2})", context or "")
    reference = (
        f" As discussed in our last message on {prior_date.group(1)}, please share an update."
        if prior_date
        else " As discussed in our last message, please share an update."
        if context
        else ""
    )
    if tier == RiskTier.LOW.value:
        tone = "We value our relationship and would appreciate your update."
    elif tier == RiskTier.HIGH.value:
        tone = "Please treat this as a priority and confirm the next step today."
    else:
        tone = "We understand that internal processing can take time; please keep us informed."

    messages = {
        "REMINDER": (
            f"Dear {name}, a reminder that INR {amount} for invoice {invoice_id}, "
            f"due on {due_date}, remains pending. Please share the expected payment date. {tone}"
        ),
        "FOLLOWUP": (
            f"Dear {name}, we have not yet received a response regarding invoice {invoice_id} "
            f"for INR {amount}, due on {due_date}. Please confirm a specific payment date or "
            f"let us know if there is an issue.{reference} {tone}"
        ),
        "NEGOTIATION": (
            f"Dear {name}, please confirm the amount you can pay and the payment date for "
            f"invoice {invoice_id} (INR {amount}, due on {due_date}). If a short extension "
            f"is needed, share a workable commitment so we can record it.{reference} {tone}"
        ),
        "ESCALATION": (
            f"DRAFT - REQUIRES HUMAN APPROVAL: Dear {name}, invoice {invoice_id} for "
            f"INR {amount} has remained pending since {due_date}. Please provide a firm "
            f"payment commitment before this matter is considered for formal escalation. "
            f"{tone} This draft must be reviewed and approved by a human before sending."
        ),
    }
    return _limit_words(messages.get(action, messages["REMINDER"]))


def _action_value(value: str | ActionType) -> str:
    raw = _value(value, "REMINDER")
    return raw.split(".")[-1].upper()


def _value(value: Any, default: str) -> str:
    if value is None:
        return default
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _amount(invoice: Any) -> str:
    return f"{float(_value(_field(invoice, 'amount'), '0')):,.2f}"


def _format_date(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    return str(value).split("T", 1)[0]


def _clean_plain_text(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"```(?:text)?", "", text, flags=re.IGNORECASE)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^\s*(subject|message)\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _limit_words(text: str) -> str:
    words = text.split()
    return " ".join(words[:MAX_WORDS])
