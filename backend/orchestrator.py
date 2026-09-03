from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session, selectinload

from backend.db import log_and_commit
from backend.models import Action, ActionType, AuditActor, AuditLog, Invoice, InvoiceState, Promise, PromiseStatus
from backend.services.outreach_generator import generate_outreach
from backend.services.policy_engine import determine_next_rung, evaluate_action
from backend.services.promise_extractor import extract_promise, should_auto_apply
from backend.services.risk_tier import tier_invoice
from backend.services.verification import close_invoice_if_fully_paid, verify_promises

LOGGER = logging.getLogger(__name__)


def run_batch_step(
    session: Session,
    as_of_date: date,
    payment_feed_simulator: Any,
    reply_simulator: Any,
) -> None:
    """Run one simulated day across all active invoices."""
    active_invoices = (
        session.query(Invoice)
        .options(
            selectinload(Invoice.replies),
            selectinload(Invoice.actions),
            selectinload(Invoice.promises),
        )
        .filter(Invoice.state.notin_([InvoiceState.CLOSED, InvoiceState.ESCALATED]))
        .all()
    )

    for invoice in active_invoices:
        # 1. Process new reply
        replies = invoice.replies
        unprocessed_replies = [r for r in replies if r.extraction_confidence is None]
        new_reply_text = None

        if unprocessed_replies:
            latest_reply = unprocessed_replies[-1]
            new_reply_text = latest_reply.raw_text
            
            extraction = extract_promise(new_reply_text, invoice)
            
            # Save extraction results to the reply
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
                # Mark others as processed too so they don't get re-processed
                for r in unprocessed_replies[:-1]:
                    r.extraction_confidence = 0.0
                continue
            
            if should_auto_apply(extraction):
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
            else:
                log_and_commit(
                    session,
                    AuditLog(
                        invoice_id=invoice.id,
                        actor=AuditActor.SYSTEM,
                        event="low-confidence extraction routed to exception queue",
                        reason="extraction confidence too low or missing amount/date",
                    ),
                )

            # Mark any other older unprocessed replies as processed (skipped)
            for r in unprocessed_replies[:-1]:
                r.extraction_confidence = 0.0

        # 2. Risk tiering
        if invoice.risk_tier is None or new_reply_text:
            tier_invoice(invoice, None, session, latest_reply_text=new_reply_text)

        current_datetime = datetime.combine(as_of_date, datetime.min.time())
        # 3. Determine proposed next action
        proposed_action_type = determine_next_rung(invoice, invoice.promises, current_date=current_datetime)

        # 4. Evaluate action via policy engine
        decision = evaluate_action(invoice, proposed_action_type, session, current_date=current_datetime)

        # 5. If blocked, take no action
        if not decision.allowed:
            continue

        # 6. If allowed but requires human approval (escalation)
        if decision.requires_human_approval:
            draft_content = generate_outreach(
                invoice, None, proposed_action_type, invoice.actions, invoice.replies
            )
            action = Action(
                invoice_id=invoice.id,
                type=proposed_action_type,
                content=draft_content,
                policy_decision="ALLOWED",
                policy_reason=decision.reason,
                timestamp=datetime.combine(as_of_date, datetime.min.time())
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
            continue

        # 7. Normal allowed case
        content = generate_outreach(
            invoice, None, proposed_action_type, invoice.actions, invoice.replies
        )
        action = Action(
            invoice_id=invoice.id,
            type=proposed_action_type,
            content=content,
            policy_decision="ALLOWED",
            policy_reason=decision.reason,
            timestamp=datetime.combine(as_of_date, datetime.min.time())
        )
        session.add(action)
        
        invoice.contact_count += 1
        invoice.last_contacted_at = datetime.combine(as_of_date, datetime.min.time())
        if invoice.state == InvoiceState.NEW:
            invoice.state = InvoiceState.CONTACTED
            
        log_and_commit(
            session,
            AuditLog(
                invoice_id=invoice.id,
                actor=AuditActor.SYSTEM,
                event=f"action_sent_{proposed_action_type.value}",
                reason=f"sent {proposed_action_type.value} outreach",
            ),
        )

        # Simulate reply
        invoice_dict = invoice.__dict__.copy()
        if hasattr(reply_simulator, 'invoices_data'):
             # fallback if we have access to raw data
             pass
        else:
            # Let's see if we can find it in synthetic invoices
            try:
                import json
                with open('data/synthetic_invoices.json', 'r') as f:
                    invoices_data = json.load(f)
                matched = next((item for item in invoices_data if item["id"] == invoice.id), None)
                if matched:
                    invoice_dict = matched
            except Exception:
                pass
                
        reply_text = reply_simulator.generate_reply(invoice_dict, proposed_action_type, invoice.contact_count)
        if reply_text:
            from backend.models import Reply
            reply = Reply(
                invoice_id=invoice.id,
                raw_text=reply_text,
                timestamp=datetime.combine(as_of_date, datetime.min.time())
            )
            session.add(reply)
            session.commit()

    # 8. Verification
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
        
        # Print summary
        states = session.query(Invoice.state).all()
        state_counts = {}
        for s in states:
            state_val = s[0].value if hasattr(s[0], "value") else str(s[0])
            state_counts[state_val] = state_counts.get(state_val, 0) + 1
            
        recovered = sum(
            payment_feed_simulator.check_payment(i.id, current_date)[1] 
            if isinstance(payment_feed_simulator.check_payment(i.id, current_date), tuple) 
            else payment_feed_simulator.check_payment(i.id, current_date).get("amount_paid", 0) 
            if isinstance(payment_feed_simulator.check_payment(i.id, current_date), dict)
            else float(payment_feed_simulator.check_payment(i.id, current_date) or 0.0)
            for i in session.query(Invoice).all()
        )
        
        print(f"State Distribution: {state_counts}")
        print(f"₹ Recovered so far: {recovered:,.2f}")
        print()
