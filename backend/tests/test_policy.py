from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import (
    AuditActor,
    AuditLog,
    Base,
    Debtor,
    Invoice,
    InvoiceState,
    PromiseStatus,
    RiskTier,
)
from backend.services.policy_engine import (
    PolicyDecision,
    determine_next_rung,
    evaluate_action,
    should_auto_apply_promise,
)


class PolicyEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=engine)
        self.Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
        self.session = self.Session()
        self.debtor = Debtor(id="DEB-TEST", name="Test Enterprises")
        self.session.add(self.debtor)
        self.session.commit()

    def tearDown(self) -> None:
        self.session.close()

    def test_opted_out_blocks_contact(self) -> None:
        invoice = self._invoice(opted_out=True)
        decision = evaluate_action(invoice, "REMINDER", self.session)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "debtor has opted out, no further contact permitted")
        self._assert_policy_audit(invoice.id, "blocked")

    def test_frequency_cap_blocks_recent_contact(self) -> None:
        invoice = self._invoice(last_contacted_at=datetime.now() - timedelta(days=1))
        decision = evaluate_action(invoice, "FOLLOWUP", self.session)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "contact frequency cap: must wait 3 days between contacts")
        self._assert_policy_audit(invoice.id, "blocked")

    def test_contact_count_cap_blocks_non_escalation(self) -> None:
        invoice = self._invoice(contact_count=5, last_contacted_at=datetime.now() - timedelta(days=4))
        decision = evaluate_action(invoice, "NEGOTIATION", self.session)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "contact cap reached, must escalate to human review instead")
        self._assert_policy_audit(invoice.id, "blocked")

    def test_escalation_requires_human_approval(self) -> None:
        invoice = self._invoice(contact_count=5, last_contacted_at=datetime.now() - timedelta(days=4))
        decision = evaluate_action(invoice, "ESCALATION", self.session)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_human_approval)
        self.assertIn("requires human approval", decision.reason)
        self._assert_policy_audit(invoice.id, "allowed")

    def test_allows_reminder_rung_for_new_invoice(self) -> None:
        invoice = self._invoice(state=InvoiceState.NEW, contact_count=0)
        rung = determine_next_rung(invoice, [])
        decision = evaluate_action(invoice, rung, self.session)
        self.assertEqual(rung.value, "REMINDER")
        self.assertTrue(decision.allowed)
        self.assertIn("REMINDER", decision.reason)

    def test_allows_followup_rung_after_four_days_without_reply(self) -> None:
        invoice = self._invoice(
            state=InvoiceState.CONTACTED,
            contact_count=1,
            days_overdue=10,
            last_contacted_at=datetime.now() - timedelta(days=5),
        )
        rung = determine_next_rung(invoice, [])
        decision = evaluate_action(invoice, rung, self.session)
        self.assertEqual(rung.value, "FOLLOWUP")
        self.assertTrue(decision.allowed)
        self.assertIn("FOLLOWUP", decision.reason)

    def test_allows_negotiation_rung_after_eight_days_without_reply(self) -> None:
        invoice = self._invoice(
            state=InvoiceState.CONTACTED,
            contact_count=1,
            days_overdue=12,
            last_contacted_at=datetime.now() - timedelta(days=9),
        )
        rung = determine_next_rung(invoice, [])
        decision = evaluate_action(invoice, rung, self.session)
        self.assertEqual(rung.value, "NEGOTIATION")
        self.assertTrue(decision.allowed)
        self.assertIn("NEGOTIATION", decision.reason)

    def test_high_risk_broken_promise_escalates(self) -> None:
        invoice = self._invoice(
            state=InvoiceState.PROMISED,
            risk_tier=RiskTier.HIGH,
            contact_count=2,
            last_contacted_at=datetime.now() - timedelta(days=4),
        )
        rung = determine_next_rung(invoice, [{"status": PromiseStatus.BROKEN}])
        decision = evaluate_action(invoice, rung, self.session)
        self.assertEqual(rung.value, "ESCALATION")
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_human_approval)

    def test_two_broken_promises_escalate_any_tier(self) -> None:
        invoice = self._invoice(
            state=InvoiceState.PROMISED,
            risk_tier=RiskTier.LOW,
            contact_count=3,
            last_contacted_at=datetime.now() - timedelta(days=4),
        )
        rung = determine_next_rung(
            invoice,
            [{"status": PromiseStatus.BROKEN}, {"status": PromiseStatus.BROKEN}],
        )
        self.assertEqual(rung.value, "ESCALATION")

    def test_overdue_unresolved_invoice_escalates_after_fifteen_days(self) -> None:
        invoice = self._invoice(
            state=InvoiceState.CONTACTED,
            contact_count=2,
            days_overdue=16,
            last_contacted_at=datetime.now() - timedelta(days=4),
        )
        rung = determine_next_rung(invoice, [])
        decision = evaluate_action(invoice, rung, self.session)
        self.assertEqual(rung.value, "ESCALATION")
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_human_approval)

    def test_low_confidence_promise_blocks_auto_transition(self) -> None:
        decision = should_auto_apply_promise(0.42)
        self.assertIsInstance(decision, PolicyDecision)
        self.assertFalse(decision.allowed)
        self.assertIn("confidence below 0.6", decision.reason)

    def _invoice(
        self,
        *,
        invoice_id: str | None = None,
        state: InvoiceState = InvoiceState.NEW,
        contact_count: int = 0,
        days_overdue: int = 7,
        last_contacted_at: datetime | None = None,
        opted_out: bool = False,
        risk_tier: RiskTier | None = None,
    ) -> Invoice:
        invoice = Invoice(
            id=invoice_id or f"INV-TEST-{len(self.session.identity_map) + datetime.now().microsecond}",
            merchant_id="MER-TEST",
            debtor_id=self.debtor.id,
            debtor_name=self.debtor.name,
            amount=10000.0,
            due_date=datetime.now().date() - timedelta(days=days_overdue),
            issued_date=datetime.now().date() - timedelta(days=days_overdue + 30),
            days_overdue=days_overdue,
            risk_tier=risk_tier,
            state=state,
            contact_count=contact_count,
            last_contacted_at=last_contacted_at,
            opted_out=opted_out,
        )
        self.session.add(invoice)
        self.session.commit()
        return invoice

    def _assert_policy_audit(self, invoice_id: str, status: str) -> None:
        audit = (
            self.session.query(AuditLog)
            .filter(AuditLog.invoice_id == invoice_id)
            .order_by(AuditLog.id.desc())
            .first()
        )
        self.assertIsNotNone(audit)
        self.assertEqual(audit.actor, AuditActor.POLICY_ENGINE)
        self.assertIn(status, audit.event)
        self.assertTrue(audit.reason)


if __name__ == "__main__":
    unittest.main()
