from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy.orm import Session

from backend.db import log_and_commit
from backend.models import AuditActor, AuditLog, Debtor, Invoice, RiskTier


LOGGER = logging.getLogger(__name__)

GEMINI_CHAT_COMPLETIONS_URL: Final = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEFAULT_MODEL: Final = "gemini-1.5-flash"


@dataclass(frozen=True)
class SentimentAnalysis:
    label: str
    confidence: float
    adjusted_score: float
    reason: str


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


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
    kept_rate = float(getattr(debtor, "historical_promise_kept_rate", 0.5) or 0.0)
    avg_days_late = float(getattr(debtor, "historical_avg_days_late", 0.0) or 0.0)

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


def adjust_for_reply_sentiment(base_score: float, latest_reply_text: str) -> float:
    """Adjust a base score using a narrowly-scoped sentiment signal.

    This helper is intentionally forgiving: if the OpenAI API is unavailable or
    the structured response is not trustworthy enough, it returns the base score
    unchanged rather than inventing a risky adjustment.
    """

    analysis = _analyze_reply_sentiment(base_score, latest_reply_text)
    return analysis.adjusted_score


def score_to_tier(score: float) -> str:
    if score < 0.35:
        return RiskTier.LOW.value
    if score <= 0.65:
        return RiskTier.MED.value
    return RiskTier.HIGH.value


def tier_invoice(
    invoice: Invoice,
    debtor: Debtor,
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
        analysis = _analyze_reply_sentiment(base_score, latest_reply_text)
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


def _analyze_reply_sentiment(base_score: float, latest_reply_text: str) -> SentimentAnalysis:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        reason = "OpenAI API key missing, using base score without sentiment adjustment"
        LOGGER.info(reason)
        return SentimentAnalysis("NEUTRAL", 1.0, _clamp(base_score), reason)

    payload = {
        "model": os.getenv("GEMINI_RISK_TIER_MODEL", DEFAULT_MODEL),
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "risk_tier_sentiment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["COOPERATIVE", "NEUTRAL", "EVASIVE", "HOSTILE"],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["label", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the debtor reply tone only. Return exactly one JSON object "
                    "with label and confidence. Do not add extra keys."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Reply text:\n"
                    f"{latest_reply_text}\n\n"
                    "Choose one label: COOPERATIVE, NEUTRAL, EVASIVE, HOSTILE. "
                    "Confidence must reflect how clear the tone is."
                ),
            },
        ],
    }

    try:
        response = _post_openai_chat_completions(key, payload)
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        label = str(parsed["label"]).upper()
        confidence = float(parsed["confidence"])
    except (HTTPError, URLError, KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError) as exc:
        reason = f"sentiment analysis failed, using base score unchanged: {exc}"
        LOGGER.warning(reason)
        return SentimentAnalysis("NEUTRAL", 0.0, _clamp(base_score), reason)

    if confidence < 0.5:
        reason = (
            f"sentiment confidence below threshold ({confidence:.2f}), "
            "using base score unchanged"
        )
        LOGGER.info(reason)
        return SentimentAnalysis(label, confidence, _clamp(base_score), reason)

    adjustment_map = {
        "COOPERATIVE": -0.10,
        "NEUTRAL": 0.00,
        "EVASIVE": 0.15,
        "HOSTILE": 0.25,
    }
    delta = adjustment_map.get(label, 0.0)
    if label not in adjustment_map:
        reason = f"unknown sentiment label {label!r}, using base score unchanged"
        LOGGER.info(reason)
        return SentimentAnalysis(label, confidence, _clamp(base_score), reason)

    adjusted_score = _clamp(base_score + delta)
    reason = f"used sentiment signal label={label} confidence={confidence:.2f} delta={delta:+.2f}"
    LOGGER.info(reason)
    return SentimentAnalysis(label, confidence, adjusted_score, reason)


def _post_openai_chat_completions(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        GEMINI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))
