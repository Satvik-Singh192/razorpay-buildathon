from __future__ import annotations

import os
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from backend.models import ActionType, RiskTier
from backend.services.outreach_generator import generate_outreach


def _invoice(tier: RiskTier) -> SimpleNamespace:
    return SimpleNamespace(
        id="INV-DEMO-001",
        debtor_name="Surat Packaging",
        amount=20000.0,
        due_date=date(2026, 8, 1),
        risk_tier=tier,
    )


def _debtor() -> SimpleNamespace:
    return SimpleNamespace(name="Surat Packaging")


class OutreachGeneratorTest(unittest.TestCase):
    def test_fallback_covers_each_action_type(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            for action in ActionType:
                message = generate_outreach(_invoice(RiskTier.MED), _debtor(), action, [], [])
                self.assertTrue(message)
                self.assertLessEqual(len(message.split()), 120)
                if action == ActionType.ESCALATION:
                    self.assertIn("DRAFT - REQUIRES HUMAN APPROVAL", message)

    def test_fallback_uses_prior_context(self) -> None:
        prior = [SimpleNamespace(type=ActionType.REMINDER, timestamp=date(2026, 8, 5))]
        replies = [SimpleNamespace(raw_text="We will confirm by Friday.", timestamp=date(2026, 8, 6))]
        with patch.dict(os.environ, {}, clear=True):
            message = generate_outreach(_invoice(RiskTier.LOW), _debtor(), ActionType.FOLLOWUP, prior, replies)
        self.assertIn("last message", message)

    def test_api_message_is_cleaned_and_limited(self) -> None:
        response = {"choices": [{"message": {"content": "**Subject:** Please pay\n\nKindly update us."}}]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), patch(
            "backend.services.outreach_generator._post_openai_chat_completions",
            return_value=response,
        ) as call:
            message = generate_outreach(_invoice(RiskTier.HIGH), _debtor(), ActionType.REMINDER, [], [])
        call.assert_called_once()
        self.assertEqual(message, "Please pay Kindly update us.")


def run_harness() -> None:
    for tier in RiskTier:
        for action in ActionType:
            message = generate_outreach(
                _invoice(tier),
                _debtor(),
                action,
                [SimpleNamespace(type=ActionType.REMINDER, timestamp=date(2026, 8, 5))],
                [SimpleNamespace(raw_text="We are reviewing this internally.", timestamp=date(2026, 8, 6))],
            )
            print(f"[{tier.value} / {action.value}]\n{message}\n")


if __name__ == "__main__":
    run_harness()
