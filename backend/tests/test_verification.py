from __future__ import annotations

import unittest
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import (
    AuditActor,
    AuditLog,
    Base,
    Debtor,
    Invoice,
    InvoiceState,
    Promise,
    PromiseStatus,
    Reply,
)
from backend.services.verification import close_invoice_if_fully_paid, verify_promises


class FakePaymentFeed:
    def __init__(self, payments: dict[str, float]):
        self.payments = payments
        self.calls: list[tuple[str, date]] = []

    def check_payment(self, invoice_id: str, as_of_date: date):
        self.calls.append((invoice_id, as_of_date))
        amount = self.payments.get(invoice_id, 0.0)
        return {
            "invoice_id": invoice_id,
            "as_of_date": as_of_date.isoformat(),
            "paid": amount > 0,
            "amount_paid": amount,
            "payments": [],
        }


class VerificationTest(unittest.TestCase):
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

    def test_kept_promise_closes_invoice_with_tolerance(self) -> None:
        invoice = self._invoice("INV-KEPT", amount=10_000)
        promise = self._promise(invoice, amount=10_000, promised_date=date(2026, 9, 5))

        summary = verify_promises(self.session, FakePaymentFeed({invoice.id: 9_910}), date(2026, 9, 5))

        self.assertEqual(summary["kept"], 1)
        self.assertEqual(summary["pending"], 0)
        self.assertEqual(promise.status, PromiseStatus.KEPT)
        self.assertEqual(invoice.state, InvoiceState.CLOSED)
        self._assert_audit(invoice.id, "promise kept")

    def test_partial_payment_does_not_close_invoice(self) -> None:
        invoice = self._invoice("INV-PARTIAL", amount=20_000)
        promise = self._promise(invoice, amount=20_000, promised_date=date(2026, 9, 5))

        summary = verify_promises(self.session, FakePaymentFeed({invoice.id: 5_000}), date(2026, 9, 6))

        self.assertEqual(summary["partial"], 1)
        self.assertEqual(promise.status, PromiseStatus.PARTIAL)
        self.assertEqual(invoice.state, InvoiceState.PROMISED)
        self._assert_audit(invoice.id, "promise partially paid")

    def test_broken_promise_marks_invoice_broken(self) -> None:
        invoice = self._invoice("INV-BROKEN", amount=15_000)
        promise = self._promise(invoice, amount=15_000, promised_date=date(2026, 9, 5))

        summary = verify_promises(self.session, FakePaymentFeed({}), date(2026, 9, 8))

        self.assertEqual(summary["broken"], 1)
        self.assertEqual(promise.status, PromiseStatus.BROKEN)
        self.assertEqual(invoice.state, InvoiceState.BROKEN)
        audit = self._assert_audit(invoice.id, "promise broken")
        self.assertIn("3 day(s) late", audit.reason)

    def test_future_pending_promise_is_not_checked(self) -> None:
        invoice = self._invoice("INV-PENDING", amount=12_000)
        self._promise(invoice, amount=12_000, promised_date=date(2026, 9, 20))
        feed = FakePaymentFeed({invoice.id: 12_000})

        summary = verify_promises(self.session, feed, date(2026, 9, 10))

        self.assertEqual(summary, {"kept": 0, "broken": 0, "partial": 0, "pending": 1})
        self.assertEqual(feed.calls, [])
        self.assertEqual(invoice.state, InvoiceState.PROMISED)

    def test_general_sweep_closes_fully_paid_invoice_without_promise(self) -> None:
        invoice = self._invoice("INV-SWEEP", amount=7_500, state=InvoiceState.CONTACTED)

        summary = close_invoice_if_fully_paid(self.session, FakePaymentFeed({invoice.id: 7_500}), date(2026, 9, 7))

        self.assertEqual(summary["closed"], 1)
        self.assertEqual(summary["amount_closed"], 7_500)
        self.assertEqual(invoice.state, InvoiceState.CLOSED)
        self._assert_audit(invoice.id, "payment received")

    def test_general_sweep_does_not_double_close(self) -> None:
        invoice = self._invoice("INV-CLOSED", amount=7_500, state=InvoiceState.CLOSED)
        feed = FakePaymentFeed({invoice.id: 7_500})

        summary = close_invoice_if_fully_paid(self.session, feed, date(2026, 9, 7))

        self.assertEqual(summary["closed"], 0)
        self.assertEqual(feed.calls, [])
        self.assertEqual(
            self.session.query(AuditLog).filter(AuditLog.invoice_id == invoice.id).count(),
            0,
        )

    def _invoice(
        self,
        invoice_id: str,
        *,
        amount: float,
        state: InvoiceState = InvoiceState.PROMISED,
    ) -> Invoice:
        invoice = Invoice(
            id=invoice_id,
            merchant_id="MER-TEST",
            debtor_id=self.debtor.id,
            debtor_name=self.debtor.name,
            amount=amount,
            due_date=date(2026, 8, 1),
            issued_date=date(2026, 7, 1),
            days_overdue=32,
            state=state,
        )
        self.session.add(invoice)
        self.session.commit()
        return invoice

    def _promise(self, invoice: Invoice, *, amount: float, promised_date: date) -> Promise:
        reply = Reply(
            invoice_id=invoice.id,
            raw_text="We will pay by the promised date.",
            extracted_promise_amount=amount,
            extracted_promise_date=promised_date,
            extraction_confidence=0.9,
        )
        self.session.add(reply)
        self.session.commit()
        promise = Promise(
            invoice_id=invoice.id,
            reply_id=reply.id,
            promised_amount=amount,
            promised_date=promised_date,
            status=PromiseStatus.PENDING,
        )
        self.session.add(promise)
        self.session.add(
            AuditLog(
                invoice_id=invoice.id,
                actor=AuditActor.SYSTEM,
                event="promise created",
                reason="test setup",
            )
        )
        self.session.commit()
        return promise

    def _assert_audit(self, invoice_id: str, event: str) -> AuditLog:
        audit = (
            self.session.query(AuditLog)
            .filter(AuditLog.invoice_id == invoice_id, AuditLog.event == event)
            .order_by(AuditLog.id.desc())
            .first()
        )
        self.assertIsNotNone(audit)
        self.assertEqual(audit.actor, AuditActor.SYSTEM)
        self.assertTrue(audit.reason)
        return audit


if __name__ == "__main__":
    unittest.main()
