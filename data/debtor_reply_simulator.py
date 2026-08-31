from __future__ import annotations

import hashlib
import random
from datetime import date, datetime, timedelta
from typing import Any


FIXED_TODAY = date(2026, 8, 31)


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _format_human_date(value: str | date, today: date = FIXED_TODAY) -> str:
    target = _parse_date(value)
    delta = (target - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta == 2:
        return "day after tomorrow"
    if 3 <= delta <= 8:
        return target.strftime("%A")
    return target.strftime("%d %b")


def _stable_rng(*parts: Any) -> random.Random:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


class DebtorReplySimulator:
    """Generate varied debtor replies consistent with hidden invoice behavior."""

    def __init__(self, today: date = FIXED_TODAY):
        self.today = today

    def generate_reply(
        self,
        invoice: dict[str, Any],
        action_type: str,
        contact_count: int,
    ) -> str | None:
        behavior = invoice["ground_truth_behavior"]
        variant = invoice.get("_simulation_truth", {}).get("variant")
        rng = _stable_rng(invoice["id"], action_type.upper(), contact_count, behavior)

        if behavior == "ghost_or_opt_out":
            if variant == "opt_out":
                if contact_count <= 1 or action_type.upper() in {"REMINDER", "FOLLOWUP"}:
                    return rng.choice(
                        [
                            "Please stop contacting me, I will settle this directly with your team.",
                            "Kindly do not send automated reminders. I am speaking to your accounts team directly and will settle there.",
                            "Stop following up on this number please. We will close it directly with your finance person.",
                            "Please remove us from these reminders. Payment discussion is already happening directly with your team.",
                        ]
                    )
                return None
            return None

        if behavior == "quick_payer":
            return self._quick_payer_reply(invoice, action_type, contact_count, rng)

        if behavior == "negotiates_keeps_promise":
            return self._kept_promise_reply(invoice, action_type, contact_count, rng)

        if behavior == "breaks_promise_then_pays":
            return self._broken_promise_reply(invoice, action_type, contact_count, rng)

        raise ValueError(f"Unknown ground_truth_behavior: {behavior}")

    def _quick_payer_reply(
        self,
        invoice: dict[str, Any],
        action_type: str,
        contact_count: int,
        rng: random.Random,
    ) -> str:
        pay_date = invoice["_simulation_truth"]["payment_events"][0]["paid_at"]
        date_text = _format_human_date(pay_date, self.today)
        return rng.choice(
            [
                f"Sure, thanks for the reminder. We are releasing it {date_text}.",
                f"Got it, payment is being processed from our side {date_text}.",
                f"Yes noted. Sending now, should reflect by {date_text}.",
                f"Sorry, missed this in approvals. I have asked accounts to clear it {date_text}.",
                f"Ha ji, reminder received. We will pay {date_text}, please check once it reflects.",
                f"Already initiated from bank side. It should hit your account by {date_text}.",
            ]
        )

    def _kept_promise_reply(
        self,
        invoice: dict[str, Any],
        action_type: str,
        contact_count: int,
        rng: random.Random,
    ) -> str:
        promise = invoice["_simulation_truth"]["promises"][0]
        promised_date = _format_human_date(promise["promised_date"], self.today)
        amount = f"Rs {invoice['amount']:,}"
        return rng.choice(
            [
                f"Cash flow is a bit tight this week. Can I clear the full {amount} by {promised_date}? Confirming we will pay in full.",
                f"We are waiting for one customer receipt. I can commit to paying the complete amount by {promised_date}.",
                f"Thoda cash crunch chal raha hai, but I will close this invoice by {promised_date}. Full payment only, no dispute.",
                f"Please give us till {promised_date}. I am confirming here that we will release the full pending amount.",
                f"Approval is done, funds are expected shortly. We will settle {amount} on or before {promised_date}.",
                f"Can you allow a few days? Pakka we will clear the full invoice by {promised_date}.",
            ]
        )

    def _broken_promise_reply(
        self,
        invoice: dict[str, Any],
        action_type: str,
        contact_count: int,
        rng: random.Random,
    ) -> str:
        promises = invoice["_simulation_truth"]["promises"]
        first = promises[0]
        second = promises[1]

        if contact_count <= 1 or action_type.upper() in {"REMINDER", "NEGOTIATION"}:
            promised_date = _format_human_date(first["promised_date"], self.today)
            return rng.choice(
                [
                    f"We can clear this by {promised_date}. Funds are expected, please bear with us.",
                    f"I will arrange payment by {promised_date}; currently collections are delayed at our end.",
                    f"Give us till {promised_date} please. I am committing to close the invoice then.",
                    f"Sir cash flow is stuck, but we should be able to release the full amount by {promised_date}.",
                    f"Not ignoring this. Please mark a promise from our side for {promised_date}, payment will be done.",
                ]
            )

        promised_date = _format_human_date(second["promised_date"], self.today)
        return rng.choice(
            [
                f"Sorry for the delay, payment did not go as planned. Need a few more days, will definitely clear by {promised_date}.",
                f"Apologies, bank approval got pushed. Please give us till {promised_date}; I will make sure it is released.",
                f"I know we committed earlier. One receivable slipped, so we need time until {promised_date}. It will be cleared then.",
                f"Sorry yaar, could not arrange funds on the promised date. Please extend to {promised_date}, full payment will go.",
                f"We missed the earlier date. Not disputing the invoice, just need until {promised_date} to pay.",
            ]
        )

