from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

from backend.models import Action, AuditLog, Base, Invoice, Promise


DATABASE_URL = f"sqlite:///{Path(__file__).with_name('receivables.db')}"

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def create_all() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()


def log_and_commit(session: Session, audit_entry: AuditLog) -> None:
    """Commit business writes and their AuditLog atomically.

    Future modules must use this helper for every Invoice.state transition and every
    Action or Promise write. Do not commit those writes directly: add the business
    objects to the session, pass the matching AuditLog here, and let this function
    commit the whole transaction together.
    """
    session.add(audit_entry)
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise


@event.listens_for(Session, "before_flush")
def _require_audit_for_sensitive_writes(session: Session, flush_context: object, instances: object) -> None:
    required_invoice_ids: set[str] = set()

    for obj in session.new:
        if isinstance(obj, (Action, Promise)) and obj.invoice_id:
            required_invoice_ids.add(obj.invoice_id)

    for obj in session.dirty:
        if isinstance(obj, Invoice):
            state_history = inspect(obj).attrs.state.history
            if state_history.has_changes():
                required_invoice_ids.add(obj.id)

    if not required_invoice_ids:
        return

    audited_invoice_ids = {
        obj.invoice_id
        for obj in session.new
        if isinstance(obj, AuditLog) and obj.invoice_id is not None
    }
    missing = required_invoice_ids - audited_invoice_ids
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise RuntimeError(
            "Sensitive write requires an AuditLog in the same transaction. "
            f"Use log_and_commit(session, audit_entry). Missing invoice_id(s): {missing_text}"
        )

