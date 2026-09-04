from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


class PaymentFeedSimulator:
    """Deterministic synthetic settlement feed backed by invoice simulation truth."""

    def __init__(
        self,
        invoices: list[dict[str, Any]] | None = None,
        invoice_file: str | Path | None = None,
        mode: str = "full"
    ):
        self.mode = mode
        if invoices is None:
            path = Path(invoice_file) if invoice_file else Path(__file__).with_name("synthetic_invoices.json")
            with path.open("r", encoding="utf-8") as handle:
                invoices = json.load(handle)

        self._invoices = {inv["id"]: inv for inv in invoices}
        self._payments_by_invoice: dict[str, list[dict[str, Any]]] = {}
        for invoice in invoices:
            events = invoice.get("_simulation_truth", {}).get("payment_events", [])
            self._payments_by_invoice[invoice["id"]] = sorted(events, key=lambda item: item["paid_at"])

    def check_payment(self, invoice_id: str, as_of_date: str | date) -> dict[str, Any]:
        as_of = _parse_date(as_of_date)
        events = self._payments_by_invoice.get(invoice_id)
        if events is None:
            raise KeyError(f"Unknown invoice_id: {invoice_id}")

        invoice = self._invoices[invoice_id]
        behavior = invoice.get("ground_truth_behavior")
        
        # Adjust payment logic based on mode
        filtered_events = []
        if self.mode == "none":
            # No intervention: only half of quick payers pay
            if behavior == "quick_payer" and hash(invoice_id) % 2 == 0:
                filtered_events = events
        elif self.mode == "naive":
            # Naive reminder: quick payers pay, half of negotiators pay
            if behavior == "quick_payer":
                filtered_events = events
            elif behavior == "negotiates_keeps_promise" and hash(invoice_id) % 2 == 0:
                filtered_events = events
        else:
            # Full system: everybody who was going to pay, pays
            filtered_events = events

        paid_events = [
            event
            for event in filtered_events
            if _parse_date(event["paid_at"]) <= as_of
        ]
        total_paid = sum(event["amount"] for event in paid_events)

        return {
            "invoice_id": invoice_id,
            "as_of_date": as_of.isoformat(),
            "paid": total_paid > 0,
            "amount_paid": total_paid,
            "payments": paid_events,
        }

