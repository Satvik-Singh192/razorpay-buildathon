from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime
from typing import Any, Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.models import ActionType, RiskTier


LOGGER = logging.getLogger(__name__)

GEMINI_CHAT_COMPLETIONS_URL: Final = "https://generativelanguage.googleapis.com/v1beta/gemini/chat/completions"
DEFAULT_MODEL: Final = "gemini-1.5-flash"
MAX_WORDS: Final = 120


def generate_outreach(
    invoice: Any,
    debtor: Any,
    action_type: str | ActionType,
    prior_actions: list[Any] | None,
    prior_replies: list[Any] | None,
) -> str:
    """Generate a send-ready collections message, with a safe local fallback."""

    action = _action_value(action_type)
    context = _build_context(prior_actions or [], prior_replies or [])
    payload = _build_payload(invoice, debtor, action, context)

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            response = _post_gemini_chat_completions(api_key, payload)
            content = response["choices"][0]["message"]["content"]
            message = _clean_plain_text(content)
            if message:
                return _limit_words(message)
        except (HTTPError, URLError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("outreach generation failed, using fallback: %s", exc)
        except Exception as exc:  # Network/client failures should not break the demo.
            LOGGER.warning("unexpected outreach generation failure, using fallback: %s", exc)

    return _fallback_message(invoice, debtor, action, context)


def _build_payload(invoice: Any, debtor: Any, action: str, context: str) -> dict[str, Any]:
    tier = _value(_field(invoice, "risk_tier"), "MED")
    amount = _amount(invoice)
    due_date = _format_date(_value(_field(invoice, "due_date"), "not available"))
    debtor_name = _value(_field(debtor, "name"), _value(_field(invoice, "debtor_name"), "your team"))

    system_prompt = (
        "You are a professional, firm-but-respectful B2B collections correspondent for an Indian SME. "
        "Write one natural message ready to send to a business debtor. "
        "Use INR and Indian business language where natural. Never threaten, abuse, shame, or make legal claims. "
        "LOW risk is friendly, MED risk is firm but understanding, and HIGH risk is clear and direct about consequences while remaining professional. "
        "Return plain text only, no markdown, subject line, quotation marks, or commentary. Keep it under 120 words."
    )
    action_guidance = {
        "REMINDER": "Give a simple nudge with the invoice amount and due date, and invite an update.",
        "FOLLOWUP": "Reference that no response was received, restate the amount, and ask for a specific commitment.",
        "NEGOTIATION": "Explicitly ask for a payment date and amount commitment; you may offer a short extension.",
        "ESCALATION": "Write a formal notice prior to escalation. Start with 'DRAFT - REQUIRES HUMAN APPROVAL:' and make the approval requirement clear.",
    }
    user_prompt = {
        "debtor_name": debtor_name,
        "invoice_id": _value(_field(invoice, "id"), "the invoice"),
        "amount": amount,
        "due_date": due_date,
        "risk_tier": tier,
        "action_type": action,
        "action_guidance": action_guidance[action],
        "prior_context": context or "No prior contact context is available.",
    }
    return {
        "model": os.getenv("GEMINI_OUTREACH_MODEL", DEFAULT_MODEL),
        "temperature": 0.7,
        "max_tokens": 180,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
        ],
    }


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
    name = _value(_field(debtor, "name"), _value(_field(invoice, "debtor_name"), "Team"))
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
        "REMINDER": f"Dear {name}, a reminder that INR {amount} for invoice {invoice_id}, due on {due_date}, remains pending. Please share the expected payment date. {tone}",
        "FOLLOWUP": f"Dear {name}, we have not yet received a response regarding invoice {invoice_id} for INR {amount}, due on {due_date}. Please confirm a specific payment date or let us know if there is an issue.{reference} {tone}",
        "NEGOTIATION": f"Dear {name}, please confirm the amount you can pay and the payment date for invoice {invoice_id} (INR {amount}, due on {due_date}). If a short extension is needed, share a workable commitment so we can record it.{reference} {tone}",
        "ESCALATION": f"DRAFT - REQUIRES HUMAN APPROVAL: Dear {name}, invoice {invoice_id} for INR {amount} has remained pending since {due_date}. Please provide a firm payment commitment before this matter is considered for formal escalation. {tone} This draft must be reviewed and approved by a human before sending.",
    }
    if tier == RiskTier.LOW.value and action == "REMINDER":
        return _limit_words(messages[action])
    return _limit_words(messages[action])


def _post_gemini_chat_completions(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        GEMINI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


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
