from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import AuditLog, Base, Debtor, Invoice, RiskTier
from backend.services.risk_tier import (
    SentimentAnalysis,
    adjust_for_reply_sentiment,
    compute_base_risk_score,
    score_to_tier,
    tier_invoice,
)


class RiskTierTest(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=engine)
        self.Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
        self.session = self.Session()
        self.debtor = Debtor(
            id="DEB-TEST",
            name="Test Enterprises",
            historical_promise_kept_rate=0.8,
            historical_avg_days_late=5,
        )
        self.session.add(self.debtor)
        self.session.commit()

    def tearDown(self) -> None:
        self.session.close()

    def test_base_score_rises_with_amount_and_age(self) -> None:
        low = self._invoice(amount=10_000, days_overdue=8)
        high = self._invoice(amount=250_000, days_overdue=45)

        low_score = compute_base_risk_score(low, self.debtor)
        high_score = compute_base_risk_score(high, self.debtor)

        self.assertLess(low_score, high_score)
        self.assertGreaterEqual(low_score, 0.0)
        self.assertLessEqual(high_score, 1.0)

    def test_score_to_tier_boundaries(self) -> None:
        self.assertEqual(score_to_tier(0.10), RiskTier.LOW.value)
        self.assertEqual(score_to_tier(0.35), RiskTier.MED.value)
        self.assertEqual(score_to_tier(0.65), RiskTier.MED.value)
        self.assertEqual(score_to_tier(0.66), RiskTier.HIGH.value)

    def test_adjust_for_reply_sentiment_falls_back_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            score = adjust_for_reply_sentiment(0.44, "Please give us till Friday, we will pay in full.")

        self.assertEqual(score, 0.44)

    def test_adjust_for_reply_sentiment_uses_structured_label(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), patch(
            "backend.services.risk_tier._post_openai_chat_completions",
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": '{"label":"EVASIVE","confidence":0.93}',
                        }
                    }
                ]
            },
        ):
            score = adjust_for_reply_sentiment(0.40, "Will update you shortly, please check again next week.")

        self.assertAlmostEqual(score, 0.55, places=6)

    def test_tier_invoice_without_reply_logs_and_persists(self) -> None:
        invoice = self._invoice(amount=30_000, days_overdue=12)
        score, tier = tier_invoice(invoice, self.debtor, self.session)

        self.assertEqual(invoice.risk_tier, RiskTier(tier))
        self.assertGreater(score, 0.0)
        self.assertGreaterEqual(
            self.session.query(AuditLog).filter(AuditLog.invoice_id == invoice.id).count(),
            3,
        )

    def test_tier_invoice_reacts_to_reply_tone(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), patch(
            "backend.services.risk_tier._post_openai_chat_completions"
        ) as mock_openai:
            mock_openai.return_value = {
                "choices": [{"message": {"content": '{"label":"COOPERATIVE","confidence":0.95}'}}]
            }
            cooperative_invoice = self._invoice(amount=40_000, days_overdue=18)
            cooperative_score, cooperative_tier = tier_invoice(
                cooperative_invoice,
                self.debtor,
                self.session,
                latest_reply_text="Thanks for the reminder, we will pay by Friday.",
            )

            mock_openai.return_value = {
                "choices": [{"message": {"content": '{"label":"EVASIVE","confidence":0.92}'}}]
            }
            evasive_invoice = self._invoice(amount=40_000, days_overdue=18)
            evasive_score, evasive_tier = tier_invoice(
                evasive_invoice,
                self.debtor,
                self.session,
                latest_reply_text="We are checking internally, please follow up next week.",
            )

            mock_openai.return_value = {
                "choices": [{"message": {"content": '{"label":"HOSTILE","confidence":0.88}'}}]
            }
            hostile_invoice = self._invoice(amount=40_000, days_overdue=18)
            hostile_score, hostile_tier = tier_invoice(
                hostile_invoice,
                self.debtor,
                self.session,
                latest_reply_text="Stop chasing us, we will deal with it later.",
            )

        self.assertLess(cooperative_score, evasive_score)
        self.assertLess(evasive_score, hostile_score)
        self.assertIn(cooperative_tier, {RiskTier.MED.value, RiskTier.HIGH.value, RiskTier.LOW.value})
        self.assertIn(hostile_tier, {RiskTier.MED.value, RiskTier.HIGH.value, RiskTier.LOW.value})

    def _invoice(self, *, amount: float, days_overdue: int) -> Invoice:
        invoice = Invoice(
            id=f"INV-{amount}-{days_overdue}-{len(self.session.identity_map)}",
            merchant_id="MER-TEST",
            debtor_id=self.debtor.id,
            debtor_name=self.debtor.name,
            amount=amount,
            due_date=datetime.now().date() - timedelta(days=days_overdue),
            issued_date=datetime.now().date() - timedelta(days=days_overdue + 30),
            days_overdue=days_overdue,
        )
        self.session.add(invoice)
        self.session.commit()
        return invoice


if __name__ == "__main__":
    unittest.main()
