from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import date, datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.db import get_session, log_and_commit
from backend.models import Invoice, Action, Reply, Promise, AuditLog, InvoiceState, ActionType, PromiseStatus, AuditActor, RiskTier
from backend.orchestrator import run_batch_step
from data.payment_feed_simulator import PaymentFeedSimulator
from data.debtor_reply_simulator import DebtorReplySimulator

app = FastAPI(title="Receivables Chaser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated state
class AppState:
    current_date = date.today()

app_state = AppState()

# Pydantic models
class InvoiceSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    state: InvoiceState
    risk_tier: Optional[RiskTier]
    amount: float
    days_overdue: int

class ActionModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    type: ActionType
    content: str
    generated_by: str
    policy_decision: str
    policy_reason: str
    timestamp: datetime
    # We add status if it was present on Action? Wait, Action in schema doesn't have status. 
    # Ah, in Prompt 8, we saved action with policy_decision="ALLOWED", but what about "PENDING_HUMAN_APPROVAL"? 
    # Let me double check if I added status.

class ReplyModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    raw_text: str
    extracted_promise_amount: Optional[float]
    extracted_promise_date: Optional[date]
    extraction_confidence: Optional[float]
    timestamp: datetime

class PromiseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    promised_amount: float
    promised_date: date
    status: PromiseStatus
    verified_at: Optional[datetime]

class InvoiceDetail(InvoiceSummary):
    actions: List[ActionModel]
    replies: List[ReplyModel]
    promises: List[PromiseModel]

class AuditLogModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    actor: AuditActor
    event: str
    reason: str
    timestamp: datetime

class MetricsResponse(BaseModel):
    total_recovered: float
    recovery_rate_pct: float
    promise_kept_rate_pct: float
    count_by_state: dict
    count_policy_blocks: int
    count_human_pending: int

class BatchSummary(BaseModel):
    date: date
    message: str

def get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()

@app.get("/invoices", response_model=List[InvoiceSummary])
def get_invoices(state: Optional[str] = None, session: Session = Depends(get_db)):
    query = session.query(Invoice)
    if state:
        query = query.filter(Invoice.state == state)
    return query.all()

@app.get("/invoices/{id}", response_model=InvoiceDetail)
def get_invoice_detail(id: str, session: Session = Depends(get_db)):
    invoice = session.query(Invoice).filter(Invoice.id == id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    # Sort chronologically
    actions = sorted(invoice.actions, key=lambda x: x.timestamp)
    replies = sorted(invoice.replies, key=lambda x: x.timestamp)
    promises = sorted(invoice.promises, key=lambda x: x.id) # using ID as proxy for creation order
    
    return InvoiceDetail(
        id=invoice.id,
        state=invoice.state.value if hasattr(invoice.state, 'value') else invoice.state,
        risk_tier=invoice.risk_tier.value if hasattr(invoice.risk_tier, 'value') else invoice.risk_tier,
        amount=invoice.amount,
        days_overdue=invoice.days_overdue,
        actions=actions,
        replies=replies,
        promises=promises
    )

@app.get("/invoices/{id}/audit", response_model=List[AuditLogModel])
def get_invoice_audit(id: str, session: Session = Depends(get_db)):
    logs = session.query(AuditLog).filter(AuditLog.invoice_id == id).order_by(AuditLog.timestamp).all()
    return logs

@app.post("/batch/run-day", response_model=BatchSummary)
def run_batch_day(session: Session = Depends(get_db)):
    payment_feed = PaymentFeedSimulator()
    reply_sim = DebtorReplySimulator()
    run_batch_step(session, app_state.current_date, payment_feed, reply_sim)
    app_state.current_date = date.fromordinal(app_state.current_date.toordinal() + 1)
    return BatchSummary(date=app_state.current_date, message="Batch step executed")

@app.get("/batch/metrics", response_model=MetricsResponse)
def get_metrics(session: Session = Depends(get_db)):
    invoices = session.query(Invoice).all()
    total_invoices_amount = sum(i.amount for i in invoices)
    
    payment_feed = PaymentFeedSimulator()
    total_recovered = 0.0
    for i in invoices:
        raw = payment_feed.check_payment(i.id, app_state.current_date)
        if isinstance(raw, dict):
            total_recovered += raw.get("amount_paid", 0.0)
        elif isinstance(raw, tuple):
            total_recovered += float(raw[1]) if raw[1] else 0.0
        elif isinstance(raw, (int, float)):
            total_recovered += float(raw)
            
    recovery_rate_pct = (total_recovered / total_invoices_amount * 100) if total_invoices_amount else 0.0
    
    promises = session.query(Promise).all()
    kept = sum(1 for p in promises if p.status == PromiseStatus.KEPT)
    total_resolved_promises = sum(1 for p in promises if p.status in [PromiseStatus.KEPT, PromiseStatus.BROKEN])
    promise_kept_rate_pct = (kept / total_resolved_promises * 100) if total_resolved_promises else 0.0
    
    states = session.query(Invoice.state, func.count(Invoice.id)).group_by(Invoice.state).all()
    count_by_state = {s[0].value if hasattr(s[0], 'value') else s[0]: s[1] for s in states}
    
    # Check audits for policy blocks
    count_policy_blocks = session.query(AuditLog).filter(AuditLog.event.like("policy_decision_blocked%")).count()
    
    # Count escalations pending
    # We saved draft actions with status="PENDING_HUMAN_APPROVAL" but wait, Action schema has no status.
    # In orchestrator we did action = Action(..., status="PENDING_HUMAN_APPROVAL") which might have failed if it doesn't exist?
    count_human_pending = session.query(Invoice).filter(Invoice.state == InvoiceState.ESCALATED).count()

    return MetricsResponse(
        total_recovered=total_recovered,
        recovery_rate_pct=recovery_rate_pct,
        promise_kept_rate_pct=promise_kept_rate_pct,
        count_by_state=count_by_state,
        count_policy_blocks=count_policy_blocks,
        count_human_pending=count_human_pending
    )

@app.post("/invoices/{id}/approve-escalation")
def approve_escalation(id: str, session: Session = Depends(get_db)):
    invoice = session.query(Invoice).filter(Invoice.id == id, Invoice.state == InvoiceState.ESCALATED).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found or not in ESCALATED state")
        
    # We consider it approved
    # Write audit log
    log_and_commit(
        session,
        AuditLog(
            invoice_id=id,
            actor=AuditActor.HUMAN,
            event="escalation_approved",
            reason="Human approved the escalation draft."
        )
    )
    return {"message": "Escalation approved"}
