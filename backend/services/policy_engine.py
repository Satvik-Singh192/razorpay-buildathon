from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

from sqlalchemy.orm import Session

from backend.db import log_and_commit
from backend.models import ActionType, AuditActor, AuditLog, InvoiceState, PromiseStatus, RiskTier


CONTACT_FREQUENCY_DAYS = 3
MAX_AUTOMATED_CONTACTS = 5
PROMISE_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    requires_human_approval: bool = False


def evaluate_action(invoice: Any, proposed_action_type: str | ActionType, session: Session) -> PolicyDecision:
    action_type = _normalize_action_type(proposed_action_type)
    requires_human_approval = False

    if invoice.opted_out is True:
        decision = PolicyDecision(
            allowed=False,
            reason="debtor has opted out, no further contact permitted",
        )
    elif _contacted_within_frequency_cap(invoice.last_contacted_at):
        decision = PolicyDecision(
            allowed=False,
            reason="contact frequency cap: must wait 3 days between contacts",
        )
    elif invoice.contact_count >= MAX_AUTOMATED_CONTACTS and action_type != ActionType.ESCALATION:
        decision = PolicyDecision(
            allowed=False,
            reason="contact cap reached, must escalate to human review instead",
        )
    elif action_type == ActionType.ESCALATION:
        requires_human_approval = True
        decision = PolicyDecision(
            allowed=True,
            reason=(
                "ESCALATION may be drafted per escalation ladder, but requires human "
                "approval before any legal or collections handoff"
            ),
            requires_human_approval=requires_human_approval,
        )
    else:
        decision = PolicyDecision(
            allowed=True,
            reason=_allow_reason_for(invoice, action_type),
        )

    _audit_policy_decision(session, invoice.id, action_type, decision)
    return decision


def determine_next_rung(invoice: Any, promise_history: Iterable[Any]) -> ActionType:
    broken_count = _broken_promise_count(promise_history)

    if invoice.state == InvoiceState.PROMISED or str(invoice.state) == InvoiceState.PROMISED.value:
        if broken_count >= 2:
            return ActionType.ESCALATION
        if broken_count >= 1 and _enum_value(invoice.risk_tier) == RiskTier.HIGH.value:
            return ActionType.ESCALATION
        if broken_count >= 1:
            return ActionType.NEGOTIATION

    if _unresolved(invoice) and invoice.days_overdue >= 15 and invoice.contact_count > 0:
        return ActionType.ESCALATION

    if _enum_value(invoice.state) == InvoiceState.NEW.value and invoice.contact_count == 0:
        return ActionType.REMINDER

    if _enum_value(invoice.state) == InvoiceState.CONTACTED.value and _has_no_reply(invoice):
        days_since_contact = _days_since(invoice.last_contacted_at)
        if days_since_contact >= 8:
            return ActionType.NEGOTIATION
        if days_since_contact >= 4:
            return ActionType.FOLLOWUP

    return ActionType.REMINDER


def should_auto_apply_promise(extraction_confidence: float | None) -> PolicyDecision:
    if extraction_confidence is None or extraction_confidence < PROMISE_CONFIDENCE_THRESHOLD:
        return PolicyDecision(
            allowed=False,
            reason=(
                "promise extraction confidence below 0.6, route to human-review "
                "exception queue before state transition"
            ),
        )
    return PolicyDecision(
        allowed=True,
        reason="promise extraction confidence meets threshold, auto-transition permitted",
    )


def _normalize_action_type(value: str | ActionType) -> ActionType:
    if isinstance(value, ActionType):
        return value
    return ActionType[str(value).upper()]


def _contacted_within_frequency_cap(last_contacted_at: datetime | None) -> bool:
    if last_contacted_at is None:
        return False
    return datetime.now() - last_contacted_at < timedelta(days=CONTACT_FREQUENCY_DAYS)


def _days_since(value: datetime | None) -> int:
    if value is None:
        return 10_000
    return (datetime.now() - value).days


def _has_no_reply(invoice: Any) -> bool:
    replies = getattr(invoice, "replies", None)
    return not replies


def _unresolved(invoice: Any) -> bool:
    return _enum_value(invoice.state) not in {
        InvoiceState.KEPT.value,
        InvoiceState.CLOSED.value,
        InvoiceState.ESCALATED.value,
    }


def _broken_promise_count(promise_history: Iterable[Any]) -> int:
    count = 0
    for promise in promise_history:
        status = _enum_value(getattr(promise, "status", None))
        if status is None and isinstance(promise, dict):
            status = _enum_value(promise.get("status"))
        if status == PromiseStatus.BROKEN.value:
            count += 1
    return count


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    raw = str(value)
    return raw.split(".")[-1] if "." in raw else raw


def _allow_reason_for(invoice: Any, action_type: ActionType) -> str:
    if action_type == ActionType.REMINDER:
        return "NEW invoice with 0 contacts, proceeding to REMINDER per escalation ladder"
    if action_type == ActionType.FOLLOWUP:
        return "day 4, no reply yet, proceeding to FOLLOWUP per escalation ladder"
    if action_type == ActionType.NEGOTIATION:
        return "day 8 or broken-promise follow-up, proceeding to NEGOTIATION per escalation ladder"
    return f"policy allows {action_type.value} for invoice state {_enum_value(invoice.state)}"


def _audit_policy_decision(
    session: Session,
    invoice_id: str,
    action_type: ActionType,
    decision: PolicyDecision,
) -> None:
    event_status = "allowed" if decision.allowed else "blocked"
    approval_suffix = "; requires_human_approval=True" if decision.requires_human_approval else ""
    audit_entry = AuditLog(
        invoice_id=invoice_id,
        actor=AuditActor.POLICY_ENGINE,
        event=f"policy_decision_{event_status}:{action_type.value}",
        reason=f"{decision.reason}{approval_suffix}",
    )
    log_and_commit(session, audit_entry)

