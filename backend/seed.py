from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.db import SessionLocal, create_all
from backend.models import Debtor, Invoice


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "synthetic_invoices.json"


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def seed_invoices(data_file: Path = DATA_FILE) -> int:
    with data_file.open("r", encoding="utf-8") as handle:
        invoice_rows: list[dict[str, Any]] = json.load(handle)

    create_all()
    with SessionLocal() as session:
        inserted = 0
        for row in invoice_rows:
            debtor = session.get(Debtor, row["debtor_id"])
            if debtor is None:
                debtor = Debtor(
                    id=row["debtor_id"],
                    name=row["debtor_name"],
                )
                session.add(debtor)

            invoice = session.get(Invoice, row["id"])
            if invoice is None:
                invoice = Invoice(
                    id=row["id"],
                    merchant_id=row["merchant_id"],
                    debtor_id=row["debtor_id"],
                    debtor_name=row["debtor_name"],
                    amount=float(row["amount"]),
                    due_date=_parse_date(row["due_date"]),
                    issued_date=_parse_date(row["issued_date"]),
                    days_overdue=int(row["days_overdue"]),
                )
                session.add(invoice)
                inserted += 1

        session.commit()
        total = session.query(Invoice).count()

    print(f"Seeded {inserted} new invoices from {data_file}")
    print(f"Invoice table count: {total}")
    return total


if __name__ == "__main__":
    seed_invoices()

