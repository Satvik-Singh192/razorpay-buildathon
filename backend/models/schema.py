from __future__ import annotations

import enum

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RiskTier(str, enum.Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


class InvoiceState(str, enum.Enum):
    NEW = "NEW"
    CONTACTED = "CONTACTED"
    PROMISED = "PROMISED"
    KEPT = "KEPT"
    BROKEN = "BROKEN"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class ActionType(str, enum.Enum):
    REMINDER = "REMINDER"
    FOLLOWUP = "FOLLOWUP"
    NEGOTIATION = "NEGOTIATION"
    ESCALATION = "ESCALATION"


class PolicyDecision(str, enum.Enum):
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"


class PromiseStatus(str, enum.Enum):
    PENDING = "PENDING"
    KEPT = "KEPT"
    BROKEN = "BROKEN"
    PARTIAL = "PARTIAL"


class AuditActor(str, enum.Enum):
    SYSTEM = "SYSTEM"
    POLICY_ENGINE = "POLICY_ENGINE"
    LLM = "LLM"
    HUMAN = "HUMAN"


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String, nullable=False)
    debtor_id: Mapped[str] = mapped_column(ForeignKey("debtors.id"), nullable=False)
    debtor_name: Mapped[str] = mapped_column(String, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    due_date: Mapped[Date] = mapped_column(Date, nullable=False)
    issued_date: Mapped[Date] = mapped_column(Date, nullable=False)
    days_overdue: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_tier: Mapped[RiskTier | None] = mapped_column(
        Enum(RiskTier, native_enum=False),
        nullable=True,
    )
    state: Mapped[InvoiceState] = mapped_column(
        Enum(InvoiceState, native_enum=False),
        default=InvoiceState.NEW,
        nullable=False,
    )
    contact_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_contacted_at: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
    opted_out: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    debtor: Mapped["Debtor"] = relationship(back_populates="invoices")
    actions: Mapped[list["Action"]] = relationship(back_populates="invoice")
    replies: Mapped[list["Reply"]] = relationship(back_populates="invoice")
    promises: Mapped[list["Promise"]] = relationship(back_populates="invoice")
    payments: Mapped[list["Payment"]] = relationship(back_populates="invoice")
    audit_logs: Mapped[list["AuditLog"]] = relationship(back_populates="invoice")


class Debtor(Base):
    __tablename__ = "debtors"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    historical_promise_kept_rate: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    historical_avg_days_late: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    invoices: Mapped[list[Invoice]] = relationship(back_populates="debtor")


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.id"), nullable=False)
    type: Mapped[ActionType] = mapped_column(Enum(ActionType, native_enum=False), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    generated_by: Mapped[str] = mapped_column(String, default="LLM", nullable=False)
    policy_decision: Mapped[PolicyDecision] = mapped_column(
        Enum(PolicyDecision, native_enum=False),
        nullable=False,
    )
    policy_reason: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    invoice: Mapped[Invoice] = relationship(back_populates="actions")


class Reply(Base):
    __tablename__ = "replies"
    __table_args__ = (
        CheckConstraint(
            "extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)",
            name="ck_replies_extraction_confidence_0_1",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.id"), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_promise_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    extracted_promise_date: Mapped[Date | None] = mapped_column(Date, nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    invoice: Mapped[Invoice] = relationship(back_populates="replies")
    promises: Mapped[list["Promise"]] = relationship(back_populates="reply")


class Promise(Base):
    __tablename__ = "promises"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.id"), nullable=False)
    reply_id: Mapped[int] = mapped_column(ForeignKey("replies.id"), nullable=False)
    promised_amount: Mapped[float] = mapped_column(Float, nullable=False)
    promised_date: Mapped[Date] = mapped_column(Date, nullable=False)
    status: Mapped[PromiseStatus] = mapped_column(Enum(PromiseStatus, native_enum=False), nullable=False)
    verified_at: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)

    invoice: Mapped[Invoice] = relationship(back_populates="promises")
    reply: Mapped[Reply] = relationship(back_populates="promises")


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.id"), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    paid_at: Mapped[Date] = mapped_column(Date, nullable=False)

    invoice: Mapped[Invoice] = relationship(back_populates="payments")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str | None] = mapped_column(ForeignKey("invoices.id"), nullable=True)
    actor: Mapped[AuditActor] = mapped_column(Enum(AuditActor, native_enum=False), nullable=False)
    event: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    invoice: Mapped[Invoice | None] = relationship(back_populates="audit_logs")

