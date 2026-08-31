from __future__ import annotations

import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

try:
    from faker import Faker
except ModuleNotFoundError:  # Keeps the generator runnable in minimal Python sandboxes.
    Faker = None


FIXED_TODAY = date(2026, 8, 31)
DEFAULT_SEED = 20260831

BEHAVIOR_LABELS = {
    "quick_payer": "pays quickly after reminder",
    "negotiates_keeps_promise": "negotiates and keeps promise",
    "breaks_promise_then_pays": "breaks at least one promise before paying",
    "ghost_or_opt_out": "ghosts or opts out",
}

INDUSTRIES = [
    "Textiles & Garments",
    "Industrial Supplies",
    "Pharma Distribution",
    "Electronics Trading",
    "Food & Beverages",
    "Construction Materials",
    "Packaging",
    "Logistics",
    "Auto Components",
    "IT Services",
    "Facility Management",
    "Printing & Signage",
]

CITY_NAMES = [
    "Aarav",
    "Vivaan",
    "Ishaan",
    "Kabir",
    "Aditi",
    "Neha",
    "Meera",
    "Rohan",
    "Nikhil",
    "Saanvi",
    "Kavya",
    "Anaya",
]

INDIAN_CITY_WORDS = [
    "Ahmedabad",
    "Bengaluru",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Indore",
    "Jaipur",
    "Kochi",
    "Kolkata",
    "Lucknow",
    "Mumbai",
    "Nagpur",
    "Pune",
    "Surat",
    "Vadodara",
]

SME_SUFFIXES = [
    "Traders",
    "Enterprises",
    "Industries",
    "Distributors",
    "Agencies",
    "Foods",
    "Packaging",
    "Logistics",
    "Solutions",
    "Exports",
    "Electricals",
    "Fabrics",
]


def _behavior_counts(n: int) -> dict[str, int]:
    weights = [
        ("quick_payer", 0.40),
        ("negotiates_keeps_promise", 0.30),
        ("breaks_promise_then_pays", 0.20),
        ("ghost_or_opt_out", 0.10),
    ]
    counts = {behavior: int(n * weight) for behavior, weight in weights}
    remainder = n - sum(counts.values())
    fractions = sorted(
        ((n * weight - counts[behavior], behavior) for behavior, weight in weights),
        reverse=True,
    )
    for _, behavior in fractions[:remainder]:
        counts[behavior] += 1
    return counts


def _amount(rng: random.Random) -> int:
    raw = int(rng.lognormvariate(10.15, 1.05))
    clamped = max(5_000, min(500_000, raw))
    return int(round(clamped / 100) * 100)


def _fake_indian_sme_name(fake: Any, rng: random.Random) -> str:
    if rng.random() < 0.55:
        prefix = rng.choice(CITY_NAMES)
    elif fake is None:
        prefix = rng.choice(INDIAN_CITY_WORDS)
    else:
        prefix = fake.city().split()[0]
    return f"{prefix} {rng.choice(SME_SUFFIXES)}"


def _make_promises_and_payments(
    behavior: str,
    amount: int,
    due_date: date,
    rng: random.Random,
) -> dict[str, Any]:
    if behavior == "quick_payer":
        paid_at = FIXED_TODAY + timedelta(days=rng.choice([0, 1, 1, 2]))
        return {
            "promises": [],
            "payment_events": [{"paid_at": paid_at.isoformat(), "amount": amount}],
        }

    if behavior == "negotiates_keeps_promise":
        promised_date = FIXED_TODAY + timedelta(days=rng.randint(5, 18))
        paid_at = promised_date - timedelta(days=rng.choice([0, 0, 1, 2]))
        return {
            "promises": [
                {
                    "sequence": 1,
                    "promised_date": promised_date.isoformat(),
                    "promised_amount": amount,
                    "expected_status": "kept",
                }
            ],
            "payment_events": [{"paid_at": paid_at.isoformat(), "amount": amount}],
        }

    if behavior == "breaks_promise_then_pays":
        first_date = FIXED_TODAY + timedelta(days=rng.randint(3, 10))
        second_date = first_date + timedelta(days=rng.randint(5, 12))
        paid_at = second_date + timedelta(days=rng.choice([-1, 0, 0, 1]))
        return {
            "promises": [
                {
                    "sequence": 1,
                    "promised_date": first_date.isoformat(),
                    "promised_amount": amount,
                    "expected_status": "broken",
                },
                {
                    "sequence": 2,
                    "promised_date": second_date.isoformat(),
                    "promised_amount": amount,
                    "expected_status": "kept",
                },
            ],
            "payment_events": [{"paid_at": paid_at.isoformat(), "amount": amount}],
        }

    natural_payment = rng.random() < 0.20
    payment_events = []
    if natural_payment:
        payment_events.append(
            {
                "paid_at": (due_date + timedelta(days=rng.randint(75, 130))).isoformat(),
                "amount": amount,
            }
        )
    return {"promises": [], "payment_events": payment_events}


def generate_invoices(n: int = 60, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    fake = Faker("en_IN") if Faker else None
    if Faker:
        Faker.seed(seed)
    rng = random.Random(seed)

    counts = _behavior_counts(n)
    behaviors = [
        behavior
        for behavior, count in counts.items()
        for _ in range(count)
    ]
    rng.shuffle(behaviors)

    ghost_indexes = [idx for idx, behavior in enumerate(behaviors) if behavior == "ghost_or_opt_out"]
    opt_out_indexes = set(rng.sample(ghost_indexes, min(2, len(ghost_indexes))))

    invoices: list[dict[str, Any]] = []
    for idx, behavior in enumerate(behaviors, start=1):
        payment_terms = rng.choice([30, 60, 90])
        days_overdue = rng.randint(7, 95)
        due_date = FIXED_TODAY - timedelta(days=days_overdue)
        issued_date = due_date - timedelta(days=payment_terms)
        amount = _amount(rng)
        simulation = _make_promises_and_payments(behavior, amount, due_date, rng)
        simulation["variant"] = "opt_out" if idx - 1 in opt_out_indexes else "ghost"
        simulation["payment_terms_days"] = payment_terms

        invoice = {
            "id": f"INV-{idx:04d}",
            "merchant_id": f"MER-{rng.randint(1, 10):03d}",
            "debtor_id": f"DEB-{idx:04d}",
            "debtor_name": _fake_indian_sme_name(fake, rng),
            "amount": amount,
            "currency": "INR",
            "issued_date": issued_date.isoformat(),
            "due_date": due_date.isoformat(),
            "payment_terms_days": payment_terms,
            "days_overdue": (FIXED_TODAY - due_date).days,
            "industry_category": rng.choice(INDUSTRIES),
            "ground_truth_behavior": behavior,
            "_simulation_truth": simulation,
        }
        invoices.append(invoice)

    return invoices


def _print_summary(invoices: list[dict[str, Any]]) -> None:
    counts = Counter(invoice["ground_truth_behavior"] for invoice in invoices)
    total = len(invoices)
    print(f"Generated {total} synthetic invoices using fixed today = {FIXED_TODAY.isoformat()}")
    print()
    print(f"{'Ground-truth behavior':<38} {'Count':>5} {'Share':>8}")
    print("-" * 55)
    for behavior in [
        "quick_payer",
        "negotiates_keeps_promise",
        "breaks_promise_then_pays",
        "ghost_or_opt_out",
    ]:
        count = counts[behavior]
        share = count / total if total else 0
        print(f"{BEHAVIOR_LABELS[behavior]:<38} {count:>5} {share:>7.0%}")
    opt_outs = sum(
        1
        for invoice in invoices
        if invoice.get("_simulation_truth", {}).get("variant") == "opt_out"
    )
    print("-" * 55)
    print(f"{'opt-out invoices inside ghost bucket':<38} {opt_outs:>5}")


def main() -> None:
    out_path = Path(__file__).with_name("synthetic_invoices.json")
    invoices = generate_invoices(60)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(invoices, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _print_summary(invoices)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
