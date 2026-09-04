"""Orchestrator — the central state-machine loop.

run_batch_step: processes one simulated day for all active invoices.
run_full_simulation: iterates run_batch_step over num_days days.

Bugs fixed vs. original build-session draft
--------------------------------------------
1. Debtor is loaded from the DB before tier_invoice (was passing None → wrong scores).
2. action.policy_decision uses PolicyDecision enum, not a raw string "ALLOWED".
3. Escalation branch: Action + AuditLog committed atomically via log_and_commit.
4. Normal outreach branch: same — no dangling session.add before the commit.
5. Relative 'data/synthetic_invoices.json' path replaced with absolute via Path(__file__).
6. LLM calls batched:
   - All unprocessed replies → extract_promise_batch (one Gemini call per run)
   - All invoices needing sentiment re-tier → adjust_for_reply_sentiment_batch (one call)
   - All invoices needing outreach → generate_outreach_batch (one call per ≤10 invoices)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, selectinload

from backend.db import log_and_commit
from backend.models import (
    Action,
    ActionType,
    AuditActor,
    AuditLog,
    Debtor,
    Invoice,
    InvoiceState,
    PolicyDecision as PolicyDecisionEnum,
    Promise,
    PromiseStatus,
    Reply,
    RiskTier,
)
from backend.services.outreach_generator import generate_outreach_batch, _build_context, _action_value
from backend.services.policy_engine import determine_next_rung, evaluate_action
from backend.services.promise_extractor import extract_promise_batch, should_auto_apply
from backend.services.risk_tier import (
    adjust_for_reply_sentiment_batch,
    compute_base_risk_score,
    score_to_tier,
    tier_invoice,
)
from backend.services.verification import close_invoice_if_fully_paid, verify_promises

LOGGER = logging.getLogger(__name__)

# Absolute path to synthetic_invoices.json — robust regardless of cwd
_INVOICES_JSON = Path(__file__).resolve().parent.parent / "data" / "synthetic_invoices.json"

_invoices_data_cache: list[dict[str, Any]] | None = None


def _load_invoices_data() -> list[dict[str, Any]]:
    global _invoices_data_cache
    if _invoices_data_cache is None:
        try:
            with _INVOICES_JSON.open("r") as fh:
                _invoices_data_cache = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not load synthetic_invoices.json: %s", exc)
            _invoices_data_cache = []
    return _invoices_data_cache


def run_batch_step(
    session: Session,
    as_of_date: date,
    payment_feed_simulator: Any,
    reply_simulator: Any,
) -> None:
    """Run one simulated day across all active invoices."""
    current_datetime = datetime.combine(as_of_date, datetime.min.time())

    # -----------------------------------------------------------------------
    # PHASE 1 — Batch promise extraction for all invoices with new replies
    # -----------------------------------------------------------------------
    extraction_batch: list[tuple[str, Invoice, int | None, Reply]] = []
    all_invoices = session.query(Invoice).options(
        selectinload(Invoice.replies), selectinload(Invoice.debtor)
    ).all()
    for invoice in all_invoices:
        unprocessed = [r for r in invoice.replies if r.extraction_confidence is None]
        if unprocessed:
            latest_reply = unprocessed[-1]
            extraction_batch.append(
                (latest_reply.raw_text, invoice, latest_reply.id, latest_reply)
            )

    if extraction_batch:
        extraction_triples = [(text, inv, rid) for text, inv, rid, _ in extraction_batch]
        extractions = extract_promise_batch(extraction_triples)

        for (reply_text, invoice, reply_id, latest_reply), extraction in zip(
            extraction_batch, extractions
        ):
            unprocessed = [r for r in invoice.replies if r.extraction_confidence is None]

            latest_reply.extracted_promise_amount = extraction.amount
            latest_reply.extracted_promise_date = extraction.date
            latest_reply.extraction_confidence = extraction.confidence

            if extraction.is_opt_out:
                invoice.opted_out = True
                log_and_commit(
                    session,
                    AuditLog(
                        invoice_id=invoice.id,
                        actor=AuditActor.SYSTEM,
                        event="opt_out_processed",
                        reason="opt-out extracted from reply; invoice opted_out set to True",
                    ),
                )
                for r in unprocessed[:-1]:
                    r.extraction_confidence = 0.0

            elif should_auto_apply(extraction):
                promise = Promise(
                    invoice_id=invoice.id,
                    reply_id=latest_reply.id,
                    promised_amount=extraction.amount,
                    promised_date=extraction.date,
                    status=PromiseStatus.PENDING,
                )
                session.add(promise)
                invoice.state = InvoiceState.PROMISED
                log_and_commit(
                    session,
                    AuditLog(
                        invoice_id=invoice.id,
                        actor=AuditActor.SYSTEM,
                        event="promise_extracted_and_applied",
                        reason=f"applied promise of {extraction.amount} by {extraction.date}",
                    ),
                )
                for r in unprocessed[:-1]:
                    r.extraction_confidence = 0.0
            else:
                log_and_commit(
                    session,
                    AuditLog(
                        invoice_id=invoice.id,
                        actor=AuditActor.SYSTEM,
                        event="low-confidence extraction routed to exception queue",
                        reason=(
                            f"extraction confidence {extraction.confidence:.2f} below threshold "
                            "or missing amount/date"
                        ),
                    ),
                )
                for r in unprocessed[:-1]:
                    r.extraction_confidence = 0.0

    # Reload after opt-out changes so opted-out invoices are excluded
    session.expire_all()
    active_invoices = (
        session.query(Invoice)
        .options(
            selectinload(Invoice.replies),
            selectinload(Invoice.actions),
            selectinload(Invoice.promises),
            selectinload(Invoice.debtor),
        )
        .filter(Invoice.state.notin_([InvoiceState.CLOSED, InvoiceState.ESCALATED]))
        .all()
    )

    # -----------------------------------------------------------------------
    # PHASE 2 — Risk tiering (batch sentiment for those with new replies,
    #           deterministic-only for those with no reply yet)
    # -----------------------------------------------------------------------
    needs_sentiment: list[tuple[Invoice, str]] = []
    needs_base_only: list[Invoice] = []

    for invoice in active_invoices:
        if invoice.opted_out:
            continue
        # Find if there's a newly-processed reply for this step
        new_reply_text: str | None = None
        processed = [
            r for r in invoice.replies
            if r.extraction_confidence is not None and r.raw_text
        ]
        if processed:
            latest = max(processed, key=lambda r: r.timestamp or datetime.min)
            new_reply_text = latest.raw_text

        if invoice.risk_tier is None:
            if new_reply_text:
                needs_sentiment.append((invoice, new_reply_text))
            else:
                needs_base_only.append(invoice)
        elif new_reply_text:
            needs_sentiment.append((invoice, new_reply_text))

    if needs_sentiment:
        sentiment_results = adjust_for_reply_sentiment_batch(needs_sentiment)
        for (invoice, reply_text), sentiment in zip(needs_sentiment, sentiment_results):
            debtor = invoice.debtor
            base_score = compute_base_risk_score(invoice, debtor)
            final_score = sentiment.adjusted_score
            final_tier = score_to_tier(final_score)
            invoice.risk_tier = RiskTier(final_tier)
            log_and_commit(
                session,
                AuditLog(
                    invoice_id=invoice.id,
                    actor=AuditActor.LLM if sentiment.reason.startswith("used sentiment") else AuditActor.SYSTEM,
                    event="risk_tier_updated",
                    reason=(
                        f"base_score={base_score:.3f}, sentiment_label={sentiment.label}, "
                        f"final_score={final_score:.3f}, tier={final_tier}"
                    ),
                ),
            )

    for invoice in needs_base_only:
        debtor = invoice.debtor
        tier_invoice(invoice, debtor, session, latest_reply_text=None)

    # -----------------------------------------------------------------------
    # PHASE 3 — Policy evaluation + batch outreach generation
    # -----------------------------------------------------------------------
    outreach_queue: list[tuple[Invoice, ActionType, bool]] = []

    for invoice in active_invoices:
        if invoice.opted_out:
            continue
        proposed_action_type = determine_next_rung(
            invoice, invoice.promises, current_date=current_datetime
        )
        decision = evaluate_action(
            invoice, proposed_action_type, session, current_date=current_datetime
        )
        if not decision.allowed:
            continue
        outreach_queue.append((invoice, proposed_action_type, decision.requires_human_approval))

    if outreach_queue:
        batch_inputs = []
        for invoice, action_type, _ in outreach_queue:
            action_str = _action_value(action_type)
            context = _build_context(invoice.actions or [], invoice.replies or [])
            debtor = invoice.debtor
            batch_inputs.append((invoice, debtor, action_str, context))

        messages = generate_outreach_batch(batch_inputs)

        for (invoice, action_type, requires_human_approval), content in zip(
            outreach_queue, messages
        ):
            if requires_human_approval:
                # Escalation draft — commit action + AuditLog atomically
                is_fallback = content.startswith("Dear ")
                action = Action(
                    invoice_id=invoice.id,
                    type=action_type,
                    content=content,
                    generated_by="FALLBACK" if is_fallback else "LLM",
                    policy_decision=PolicyDecisionEnum.ALLOWED,
                    policy_reason="Escalation draft — requires human approval before sending.",
                    timestamp=current_datetime,
                )
                session.add(action)
                invoice.state = InvoiceState.ESCALATED
                log_and_commit(
                    session,
                    AuditLog(
                        invoice_id=invoice.id,
                        actor=AuditActor.SYSTEM,
                        event="escalation_drafted",
                        reason="drafted escalation requires human approval",
                    ),
                )
            else:
                # Normal outreach — commit action + AuditLog atomically
                is_fallback = content.startswith("Dear ")
                action = Action(
                    invoice_id=invoice.id,
                    type=action_type,
                    content=content,
                    generated_by="FALLBACK" if is_fallback else "LLM",
                    policy_decision=PolicyDecisionEnum.ALLOWED,
                    policy_reason=f"Policy allowed {action_type.value} outreach.",
                    timestamp=current_datetime,
                )
                session.add(action)
                invoice.contact_count += 1
                invoice.last_contacted_at = current_datetime
                if invoice.state == InvoiceState.NEW:
                    invoice.state = InvoiceState.CONTACTED
                log_and_commit(
                    session,
                    AuditLog(
                        invoice_id=invoice.id,
                        actor=AuditActor.SYSTEM,
                        event=f"action_sent_{action_type.value}",
                        reason=f"sent {action_type.value} outreach",
                    ),
                )

                # Simulate debtor reply for the next step
                invoice_dict = _get_invoice_dict(invoice)
                reply_text = reply_simulator.generate_reply(
                    invoice_dict, action_type, invoice.contact_count
                )
                if reply_text:
                    import random
                    from datetime import timedelta
                    delay = timedelta(hours=random.randint(1, 8), minutes=random.randint(1, 59))
                    reply = Reply(
                        invoice_id=invoice.id,
                        raw_text=reply_text,
                        timestamp=current_datetime + delay,
                    )
                    session.add(reply)
                    session.commit()

    # -----------------------------------------------------------------------
    # PHASE 4 — Verification
    # -----------------------------------------------------------------------
    verify_promises(session, payment_feed_simulator, as_of_date)
    close_invoice_if_fully_paid(session, payment_feed_simulator, as_of_date)


def run_full_simulation(
    session: Session,
    payment_feed_simulator: Any,
    reply_simulator: Any,
    num_days: int = 30,
) -> None:
    start_date = date.today()
    for day in range(num_days):
        current_date = date.fromordinal(start_date.toordinal() + day)
        print(f"--- Simulating Day {day + 1} ({current_date}) ---")
        run_batch_step(session, current_date, payment_feed_simulator, reply_simulator)

        states = session.query(Invoice.state).all()
        state_counts: dict[str, int] = {}
        for (s,) in states:
            state_val = s.value if hasattr(s, "value") else str(s)
            state_counts[state_val] = state_counts.get(state_val, 0) + 1

        recovered = 0.0
        for inv in session.query(Invoice).all():
            raw = payment_feed_simulator.check_payment(inv.id, current_date)
            if isinstance(raw, dict):
                recovered += float(raw.get("amount_paid", raw.get("amount", 0.0)) or 0.0)
            elif isinstance(raw, tuple) and len(raw) >= 2:
                recovered += float(raw[1] or 0.0)
            elif isinstance(raw, (int, float)):
                recovered += float(raw)

        print(f"State Distribution: {state_counts}")
        print(f"₹ Recovered so far: {recovered:,.2f}")
        print()


def _get_invoice_dict(invoice: Invoice) -> dict[str, Any]:
    """Return a dict for the reply simulator, preferring the original JSON data."""
    invoices_data = _load_invoices_data()
    matched = next((item for item in invoices_data if item.get("id") == invoice.id), None)
    if matched:
        return matched
    return {
        "id": invoice.id,
        "amount": invoice.amount,
        "debtor_name": invoice.debtor_name,
        "days_overdue": invoice.days_overdue,
        "state": invoice.state.value if hasattr(invoice.state, "value") else str(invoice.state),
        "ground_truth_behavior": "pays_on_reminder",
    }
