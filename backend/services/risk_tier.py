"""Risk tiering module — deterministic scoring with a narrow LLM sentiment signal.

Design principle: deterministic-first.  The base score uses only a weighted
formula over invoice/debtor fields.  The LLM is called ONLY for the
sentiment-adjustment step, and ONLY when a debtor reply is present.

Public API
----------
compute_base_risk_score(invoice, debtor) -> float
score_to_tier(score) -> "LOW" | "MED" | "HIGH"
adjust_for_reply_sentiment(base_score, latest_reply_text) -> float
    Single-item wrapper kept for backward compatibility.
tier_invoice(invoice, debtor, session, latest_reply_text=None) -> (float, str)

adjust_for_reply_sentiment_batch(items) -> list[SentimentAnalysis]
    Batch API: accepts list of (invoice, reply_text) pairs, returns one
    SentimentAnalysis per input in the same order.  Sends ONE Gemini request
    per batch.  On batch failure, returns base_score unchanged for every item.

LLM call details
----------------
URL:     https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
AUTH:    x-goog-api-key header
MODEL:   gemini-flash-latest
PAYLOAD: system_instruction / contents / generationConfig (responseMimeType json)
PARSE:   candidates[0]["content"]["parts"][0]["text"]

Cache
-----
Key: (invoice_id, reply_id_or_hash)
Stored in: cache/sentiment.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from backend.db import log_and_commit
from backend.models import AuditActor, AuditLog, Debtor, Invoice, RiskTier
from backend.services.llm_client import (
    DEFAULT_MODEL,
    post_gemini_with_retry,
    read_cache,
    write_cache,
)

LOGGER = logging.getLogger(__name__)

_CACHE_NAME = "sentiment"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SentimentAnalysis:
    label: str
    confidence: float
    adjusted_score: float
    reason: str


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


# ---------------------------------------------------------------------------
# Deterministic scoring
# ---------------------------------------------------------------------------

def compute_base_risk_score(invoice: Any, debtor: Any) -> float:
    """Return a deterministic base risk score on [0, 1].

    Weighting rationale:
    - amount: 30% because larger receivables have higher recovery impact and
      often merit more attention.
    - days_overdue: 35% because aging is the clearest operational signal.
    - historical_promise_kept_rate: 20% because a lower keep-rate strongly
      predicts follow-through risk.
    - historical_avg_days_late: 15% because chronic lateness is useful, but
      secondary to aging and amount.

    The amount feature is log-scaled so the score changes meaningfully across
    small and large invoices without letting the largest invoices dominate.
    """
    amount = float(getattr(invoice, "amount", 0.0) or 0.0)
    days_overdue = float(getattr(invoice, "days_overdue", 0.0) or 0.0)
    kept_rate = float(getattr(debtor, "historical_promise_kept_rate", 0.5) or 0.5) if debtor is not None else 0.5
    avg_days_late = float(getattr(debtor, "historical_avg_days_late", 0.0) or 0.0) if debtor is not None else 0.0

    amount_score = 0.0
    if amount > 0:
        min_amount = math.log10(5_000)
        max_amount = math.log10(500_000)
        amount_score = _clamp((math.log10(amount) - min_amount) / (max_amount - min_amount))

    days_score = _clamp(days_overdue / 90.0)
    keep_risk_score = _clamp(1.0 - kept_rate)
    lateness_score = _clamp(avg_days_late / 60.0)

    score = (
        0.30 * amount_score
        + 0.35 * days_score
        + 0.20 * keep_risk_score
        + 0.15 * lateness_score
    )
    return _clamp(score)


def score_to_tier(score: float) -> str:
    if score < 0.35:
        return RiskTier.LOW.value
    if score <= 0.65:
        return RiskTier.MED.value
    return RiskTier.HIGH.value


# ---------------------------------------------------------------------------
# Single-item wrappers (backward-compat)
# ---------------------------------------------------------------------------

def adjust_for_reply_sentiment(base_score: float, latest_reply_text: str) -> float:
    """Single-item wrapper — used in tests and by tier_invoice."""
    analysis = _analyze_reply_sentiment_single(base_score, latest_reply_text)
    return analysis.adjusted_score


def tier_invoice(
    invoice: Invoice,
    debtor: Debtor | None,
    session: Session,
    latest_reply_text: str | None = None,
) -> tuple[float, str]:
    base_score = compute_base_risk_score(invoice, debtor)
    session.add(
        AuditLog(
            invoice_id=invoice.id,
            actor=AuditActor.SYSTEM,
            event="risk_tier_base_score",
            reason=(
                "base score computed from amount, days overdue, "
                "historical keep rate, and historical lateness "
                f"(score={base_score:.3f})"
            ),
        )
    )

    final_score = base_score
    if latest_reply_text:
        analysis = _analyze_reply_sentiment_single(base_score, latest_reply_text)
        if analysis.reason.startswith("used sentiment signal"):
            session.add(
                AuditLog(
                    invoice_id=invoice.id,
                    actor=AuditActor.LLM,
                    event="risk_tier_sentiment_adjustment",
                    reason=(
                        f"label={analysis.label}, confidence={analysis.confidence:.2f}, "
                        f"delta={analysis.adjusted_score - base_score:+.2f}"
                    ),
                )
            )
        else:
            session.add(
                AuditLog(
                    invoice_id=invoice.id,
                    actor=AuditActor.SYSTEM,
                    event="risk_tier_sentiment_skipped",
                    reason=analysis.reason,
                )
            )
        final_score = analysis.adjusted_score

    final_tier = score_to_tier(final_score)
    invoice.risk_tier = RiskTier(final_tier)
    session.add(
        AuditLog(
            invoice_id=invoice.id,
            actor=AuditActor.SYSTEM,
            event="risk_tier_finalized",
            reason=f"final tier={final_tier}, score={final_score:.3f}",
        )
    )

    log_and_commit(
        session,
        AuditLog(
            invoice_id=invoice.id,
            actor=AuditActor.SYSTEM,
            event="risk_tier_commit",
            reason=f"persisted risk_tier={final_tier}, score={final_score:.3f}",
        ),
    )
    return final_score, final_tier


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def adjust_for_reply_sentiment_batch(
    items: list[tuple[Any, str]],
) -> list[SentimentAnalysis]:
    """Batch sentiment classification for (invoice, reply_text) pairs.

    Returns one SentimentAnalysis per input in the same order.
    On batch failure, returns base_score unchanged for all items.
    """
    if not items:
        return []

    results: list[SentimentAnalysis | None] = [None] * len(items)
    uncached_indices: list[int] = []

    # Pre-compute base scores and check cache
    base_scores: list[float] = []
    for i, (invoice, reply_text) in enumerate(items):
        # base_score needs a debtor — not available in batch context, use 0.5 default
        # (the caller should have already computed the base score; we just need it for fallback)
        base_score = _base_score_from_invoice_only(invoice)
        base_scores.append(base_score)

        cache_key = _sentiment_cache_key(invoice, reply_text)
        cached = read_cache(_CACHE_NAME, cache_key)
        if cached is not None:
            try:
                results[i] = _deserialize_sentiment(cached, base_score)
            except (KeyError, TypeError, ValueError):
                uncached_indices.append(i)
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return results  # type: ignore[return-value]

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        reason = "Gemini API key missing, using base score without sentiment adjustment"
        LOGGER.info(reason)
        for i in uncached_indices:
            results[i] = SentimentAnalysis("NEUTRAL", 1.0, _clamp(base_scores[i]), reason)
        return results  # type: ignore[return-value]

    uncached_items = [items[i] for i in uncached_indices]
    uncached_base_scores = [base_scores[i] for i in uncached_indices]
    
    llm_results = []
    batch_size = 40
    for batch_start in range(0, len(uncached_items), batch_size):
        batch = uncached_items[batch_start : batch_start + batch_size]
        base_batch = uncached_base_scores[batch_start : batch_start + batch_size]
        llm_results.extend(_call_gemini_sentiment_batch(batch, base_batch, key))

    for idx, analysis in zip(uncached_indices, llm_results):
        results[idx] = analysis
        invoice, reply_text = items[idx]
        cache_key = _sentiment_cache_key(invoice, reply_text)
        write_cache(_CACHE_NAME, cache_key, _serialize_sentiment(analysis))

    return results  # type: ignore[return-value]


def _call_gemini_sentiment_batch(
    items: list[tuple[Any, str]],
    base_scores: list[float],
    api_key: str,
) -> list[SentimentAnalysis]:
    """Send ONE Gemini request for sentiment classification of N reply texts.

    Returns exactly len(items) SentimentAnalysis objects.
    On failure, returns base_score unchanged for all items.
    """
    n = len(items)
    model = os.getenv("GEMINI_RISK_TIER_MODEL", DEFAULT_MODEL)

    system_prompt = (
        "You are a debtor-reply tone classifier. "
        "You will receive a JSON array of reply texts. "
        "For each, classify the tone as exactly one of: COOPERATIVE, NEUTRAL, EVASIVE, HOSTILE. "
        "Return ONLY a valid JSON array of exactly "
        + str(n)
        + " objects, each with keys 'label' (string) and 'confidence' (float 0-1), "
        "in the same order as the inputs. No extra keys, no commentary."
    )

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": json.dumps(
                            [{"reply_text": reply_text} for _, reply_text in items],
                            ensure_ascii=True,
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 50 * n,
        },
    }

    try:
        response = _post_gemini_generate_content(api_key, payload, model)
        raw_text = response["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list) or len(parsed) != n:
            raise ValueError(
                f"Expected JSON array of {n} objects, got {type(parsed).__name__} "
                f"len={len(parsed) if isinstance(parsed, list) else 'N/A'}"
            )
        return [_make_sentiment(item, base) for item, base in zip(parsed, base_scores)]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Gemini sentiment batch of %d failed, using base scores: %s", n, exc
        )
        reason = f"sentiment batch failed, using base score unchanged: {exc}"
        return [
            SentimentAnalysis("NEUTRAL", 0.0, _clamp(base), reason)
            for base in base_scores
        ]


def _make_sentiment(parsed_item: Any, base_score: float) -> SentimentAnalysis:
    label = str(parsed_item.get("label", "NEUTRAL")).upper()
    confidence = float(parsed_item.get("confidence", 0.0) or 0.0)

    if confidence < 0.5:
        reason = (
            f"sentiment confidence below threshold ({confidence:.2f}), "
            "using base score unchanged"
        )
        return SentimentAnalysis(label, confidence, _clamp(base_score), reason)

    adjustment_map = {
        "COOPERATIVE": -0.10,
        "NEUTRAL": 0.00,
        "EVASIVE": 0.15,
        "HOSTILE": 0.25,
    }
    if label not in adjustment_map:
        reason = f"unknown sentiment label {label!r}, using base score unchanged"
        return SentimentAnalysis(label, confidence, _clamp(base_score), reason)

    delta = adjustment_map[label]
    adjusted = _clamp(base_score + delta)
    reason = f"used sentiment signal label={label} confidence={confidence:.2f} delta={delta:+.2f}"
    return SentimentAnalysis(label, confidence, adjusted, reason)


# ---------------------------------------------------------------------------
# Single-item internal (used by tier_invoice and adjust_for_reply_sentiment)
# ---------------------------------------------------------------------------

def _analyze_reply_sentiment_single(base_score: float, latest_reply_text: str) -> SentimentAnalysis:
    """Single-item sentiment analysis; used by the single-item public API."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        reason = "Gemini API key missing, using base score without sentiment adjustment"
        LOGGER.info(reason)
        return SentimentAnalysis("NEUTRAL", 1.0, _clamp(base_score), reason)

    payload = {
        "system_instruction": {
            "parts": [{"text": "Classify the debtor reply tone only. Return exactly one JSON object with label and confidence. Do not add extra keys."}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": (
                    "Reply text:\n"
                    f"{latest_reply_text}\n\n"
                    "Choose one label: COOPERATIVE, NEUTRAL, EVASIVE, HOSTILE. "
                    "Confidence must reflect how clear the tone is."
                )}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        model = os.getenv("GEMINI_RISK_TIER_MODEL", DEFAULT_MODEL)
        response = _post_gemini_generate_content(key, payload, model)
        content = response["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(content)
        label = str(parsed["label"]).upper()
        confidence = float(parsed["confidence"])
    except Exception as exc:  # noqa: BLE001
        reason = f"sentiment analysis failed, using base score unchanged: {exc}"
        LOGGER.warning(reason)
        return SentimentAnalysis("NEUTRAL", 0.0, _clamp(base_score), reason)

    return _make_sentiment({"label": label, "confidence": confidence}, base_score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_gemini_generate_content(
    api_key: str,
    payload: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Thin wrapper so tests can patch this exact symbol."""
    return post_gemini_with_retry(api_key, payload, model)


def _base_score_from_invoice_only(invoice: Any) -> float:
    """Compute a base score using only invoice fields (debtor not available in batch context)."""
    amount = float(getattr(invoice, "amount", 0.0) or 0.0)
    days_overdue = float(getattr(invoice, "days_overdue", 0.0) or 0.0)
    amount_score = 0.0
    if amount > 0:
        min_amount = math.log10(5_000)
        max_amount = math.log10(500_000)
        amount_score = _clamp((math.log10(amount) - min_amount) / (max_amount - min_amount))
    days_score = _clamp(days_overdue / 90.0)
    return _clamp(0.30 * amount_score + 0.35 * days_score + 0.20 * 0.5 + 0.15 * 0.0)


def _sentiment_cache_key(invoice: Any, reply_text: str) -> tuple[str, str]:
    invoice_id = str(getattr(invoice, "id", "unknown"))
    # Use a hash of the reply text as a stable key (reply row id not always available here)
    text_hash = hashlib.sha256(reply_text.encode()).hexdigest()[:16]
    return (invoice_id, text_hash)


def _serialize_sentiment(analysis: SentimentAnalysis) -> dict[str, Any]:
    return {
        "label": analysis.label,
        "confidence": analysis.confidence,
        "adjusted_score": analysis.adjusted_score,
        "reason": analysis.reason,
    }


def _deserialize_sentiment(data: dict[str, Any], base_score: float) -> SentimentAnalysis:
    return SentimentAnalysis(
        label=data["label"],
        confidence=float(data["confidence"]),
        adjusted_score=float(data["adjusted_score"]),
        reason=data["reason"],
    )
