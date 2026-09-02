from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session, selectinload

from backend.db import log_and_commit
from backend.models import AuditActor, AuditLog, Invoice, InvoiceState, Promise, PromiseStatus


ABSOLUTE_PAYMENT_TOLERANCE_INR = 10.0
RELATIVE_PAYMENT_TOLERANCE = 0.01


@dataclass(frozen=True)
class PaymentSnapshot:
    paid: bool
    amount_paid: float


def verify_promises(session: Session, payment_feed_simulator: Any, as_of_date: date | datetime | str) -> dict[str, int]:
    """Verify due pending promises against the deterministic payment feed."""
    as_of = _coerce_date(as_of_date)
    summary = {"kept": 0, "broken": 0, "partial": 0, "pending": 0}

    due_promises = (
        session.query(Promise)
        .options(selectinload(Promise.invoice))
        .filter(Promise.status == PromiseStatus.PENDING, Promise.promised_date <= as_of)
        .order_by(Promise.promised_date, Promise.id)
        .all()
    )

    for promise in due_promises:
        invoice = promise.invoice
        snapshot = _check_payment(payment_feed_simulator, promise.invoice_id, as_of)
        tolerance = _payment_tolerance(promise.promised_amount)
        verified_at = _verified_at(as_of)

        if snapshot.amount_paid + tolerance >= promise.promised_amount:
            promise.status = PromiseStatus.KEPT
            promise.verified_at = verified_at
            if invoice is not None:
                invoice.state = InvoiceState.CLOSED
            summary["kept"] += 1
            _commit_verification(
                session,
                promise.invoice_id,
                "promise kept",
                (
                    f"Payment of INR {snapshot.amount_paid:,.2f} received by {as_of.isoformat()}, "
                    f"meeting promised amount INR {promise.promised_amount:,.2f} within tolerance "
                    f"INR {tolerance:,.2f}; invoice marked KEPT then CLOSED."
                ),
            )
        elif snapshot.amount_paid > 0:
            promise.status = PromiseStatus.PARTIAL
            promise.verified_at = verified_at
            if invoice is not None and invoice.state != InvoiceState.CLOSED:
                invoice.state = InvoiceState.PROMISED
            summary["partial"] += 1
            _commit_verification(
                session,
                promise.invoice_id,
                "promise partially paid",
                (
                    f"Partial payment of INR {snapshot.amount_paid:,.2f} received by {as_of.isoformat()} "
                    f"against promised amount INR {promise.promised_amount:,.2f}; invoice remains PROMISED."
                ),
            )
        else:
            promise.status = PromiseStatus.BROKEN
            promise.verified_at = verified_at
            if invoice is not None and invoice.state != InvoiceState.CLOSED:
                invoice.state = InvoiceState.BROKEN
            summary["broken"] += 1
            days_late = max(0, (as_of - promise.promised_date).days)
            _commit_verification(
                session,
                promise.invoice_id,
                "promise broken",
                (
                    f"No payment received by promised date {promise.promised_date.isoformat()}; "
                    f"promise is {days_late} day(s) late as of {as_of.isoformat()}."
                ),
            )

    summary["pending"] = (
        session.query(Promise)
        .filter(Promise.status == PromiseStatus.PENDING)
        .count()
    )
    return summary


def close_invoice_if_fully_paid(
    session: Session,
    payment_feed_simulator: Any,
    as_of_date: date | datetime | str,
) -> dict[str, int | float]:
    """Close non-closed invoices whose cumulative payment feed covers the invoice amount."""
    as_of = _coerce_date(as_of_date)
    summary: dict[str, int | float] = {"closed": 0, "amount_closed": 0.0}

    invoices = (
        session.query(Invoice)
        .filter(Invoice.state != InvoiceState.CLOSED)
        .order_by(Invoice.id)
        .all()
    )
    for invoice in invoices:
        snapshot = _check_payment(payment_feed_simulator, invoice.id, as_of)
        tolerance = _payment_tolerance(invoice.amount)
        if snapshot.amount_paid + tolerance < invoice.amount:
            continue

        invoice.state = InvoiceState.CLOSED
        summary["closed"] = int(summary["closed"]) + 1
        summary["amount_closed"] = float(summary["amount_closed"]) + invoice.amount
        _commit_verification(
            session,
            invoice.id,
            "payment received",
            (
                f"Payment sweep found INR {snapshot.amount_paid:,.2f} paid by {as_of.isoformat()}, "
                f"covering invoice amount INR {invoice.amount:,.2f} within tolerance INR {tolerance:,.2f}; "
                "invoice closed."
            ),
        )

    return summary


def _check_payment(payment_feed_simulator: Any, invoice_id: str, as_of: date) -> PaymentSnapshot:
    raw = payment_feed_simulator.check_payment(invoice_id, as_of)
    if isinstance(raw, dict):
        amount = raw.get("amount_paid", raw.get("amount", 0.0))
        paid = bool(raw.get("paid", float(amount or 0.0) > 0.0))
        return PaymentSnapshot(paid=paid, amount_paid=float(amount or 0.0))
    if isinstance(raw, tuple) and len(raw) >= 2:
        return PaymentSnapshot(paid=bool(raw[0]), amount_paid=float(raw[1] or 0.0))
    if isinstance(raw, (int, float)):
        return PaymentSnapshot(paid=float(raw) > 0.0, amount_paid=float(raw))
    return PaymentSnapshot(paid=bool(raw), amount_paid=0.0)


def _payment_tolerance(expected_amount: float) -> float:
    return max(ABSOLUTE_PAYMENT_TOLERANCE_INR, float(expected_amount) * RELATIVE_PAYMENT_TOLERANCE)


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).split("T", 1)[0])


def _verified_at(as_of: date) -> datetime:
    return datetime.combine(as_of, datetime.min.time())


def _commit_verification(session: Session, invoice_id: str, event: str, reason: str) -> None:
    log_and_commit(
        session,
        AuditLog(
            invoice_id=invoice_id,
            actor=AuditActor.SYSTEM,
            event=event,
            reason=reason,
        ),
    )
