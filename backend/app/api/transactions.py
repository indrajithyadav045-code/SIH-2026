"""
VoiceGuard Financial Transaction Authorization Gateway
Enforces live voice risk gating for sensitive fund transfers.
"""

import uuid
import datetime
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.schema_models import Transaction, Call, Alert, VerificationEvent
from backend.app.schemas.api_schemas import (
    TransactionRequest, TransactionResponse, TransactionActionRequest
)

logger = logging.getLogger("VoiceGuard.API.Transactions")
router = APIRouter(prefix="/transactions", tags=["Transactions"])

@router.post("/request", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
def request_transaction(payload: TransactionRequest, db: Session = Depends(get_db)):
    """Submits a financial instruction (e.g. ₹25,00,000 transfer)."""
    call = db.query(Call).filter(Call.id == payload.call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Associated call session not found")

    current_risk = call.final_risk_score or 10.0
    tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"

    if current_risk >= 75.0:
        initial_status = "BLOCKED"
    elif current_risk > 30.0:
        initial_status = "PENDING"
    else:
        initial_status = "AUTHORIZED"

    tx = Transaction(
        id=tx_id,
        call_id=payload.call_id,
        amount=payload.amount,
        currency=payload.currency,
        transaction_type=payload.transaction_type,
        beneficiary=payload.beneficiary,
        is_new_beneficiary=payload.is_new_beneficiary,
        risk_score_at_request=current_risk,
        status=initial_status
    )
    db.add(tx)

    if initial_status == "BLOCKED":
        alert = Alert(
            id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            call_id=payload.call_id,
            recipient_role="SUPERVISOR",
            severity="CRITICAL",
            message=f"TRANSACTION BLOCKED: {payload.currency} {payload.amount:,.2f} to {payload.beneficiary} blocked. Voice impersonation risk {current_risk}%.",
            status="ACTIVE"
        )
        db.add(alert)

    db.commit()
    db.refresh(tx)
    return tx

@router.get("", response_model=List[TransactionResponse])
def list_transactions(limit: int = 25, db: Session = Depends(get_db)):
    return db.query(Transaction).order_by(Transaction.created_at.desc()).limit(limit).all()

@router.post("/{tx_id}/action", response_model=TransactionResponse)
def handle_transaction_action(
    tx_id: str,
    payload: TransactionActionRequest,
    db: Session = Depends(get_db)
):
    tx = db.query(Transaction).filter(Transaction.id == tx_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    action = payload.action.upper()
    verif_id = f"VERIF-{uuid.uuid4().hex[:8].upper()}"

    if action == "VERIFY_MFA":
        tx.status = "PENDING_MFA"
        verif = VerificationEvent(
            id=verif_id,
            call_id=tx.call_id,
            method="OUT_OF_BAND_MFA",
            result="PENDING",
            notes=payload.notes or "Push MFA dispatched to registered mobile."
        )
        db.add(verif)
    elif action == "REGISTERED_CALLBACK":
        tx.status = "PENDING_CALLBACK"
        verif = VerificationEvent(
            id=verif_id,
            call_id=tx.call_id,
            method="REGISTERED_CALLBACK",
            result="PENDING",
            notes=payload.notes or "Automated callback placed to registered number."
        )
        db.add(verif)
    elif action == "ESCALATE":
        tx.status = "ESCALATED"
        alert = Alert(
            id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            call_id=tx.call_id,
            recipient_role="FRAUD_OPS",
            severity="CRITICAL",
            message=f"Manual Escalation for Transaction {tx_id} ({tx.currency} {tx.amount:,.2f}) by Security Operator.",
            status="ACTIVE"
        )
        db.add(alert)
    elif action == "REJECT":
        tx.status = "BLOCKED"
    elif action == "OVERRIDE_APPROVE":
        tx.status = "AUTHORIZED"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    db.commit()
    db.refresh(tx)
    return tx