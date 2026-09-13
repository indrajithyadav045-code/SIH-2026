"""
VoiceGuard Financial Transaction Gatekeeper Endpoints
"""

import uuid
import datetime
import logging
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.schema_models import Transaction, Call, VerificationEvent
from backend.app.schemas.api_schemas import (
    TransactionRequest, TransactionActionRequest, TransactionResponse
)

logger = logging.getLogger("VoiceGuard.API.Transactions")
router = APIRouter(prefix="/transactions", tags=["Transactions"])


def get_or_create_transaction(db: Session, tx_id: str) -> Transaction:
    """Ensures transaction object exists in database."""
    tx = db.query(Transaction).filter(Transaction.id == tx_id).first()
    if not tx:
        tx = Transaction(
            id=tx_id,
            call_id="CALL-LIVE-OPERATOR",
            amount=2500000.0,
            currency="INR",
            transaction_type="WIRE_TRANSFER",
            beneficiary="Apex Global Holdings LLC",
            is_new_beneficiary=True,
            status="BLOCKED",
            risk_score_at_request=87.5
        )
        db.add(tx)
        db.commit()
        db.refresh(tx)
    return tx


@router.post("", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
def initiate_transaction(payload: TransactionRequest, db: Session = Depends(get_db)):
    """Initiates a high-risk financial transaction."""
    tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"
    new_tx = Transaction(
        id=tx_id,
        call_id=payload.call_id,
        amount=payload.amount,
        currency=payload.currency,
        transaction_type=payload.transaction_type,
        beneficiary=payload.beneficiary,
        is_new_beneficiary=payload.is_new_beneficiary,
        status="BLOCKED" if payload.amount > 1000000.0 else "PENDING",
        risk_score_at_request=75.0
    )
    db.add(new_tx)
    db.commit()
    db.refresh(new_tx)
    return new_tx


@router.get("", response_model=List[TransactionResponse])
def list_transactions(db: Session = Depends(get_db)):
    """Lists transactions."""
    return db.query(Transaction).order_by(Transaction.created_at.desc()).limit(20).all()


@router.post("/{tx_id}/action", response_model=TransactionResponse)
def execute_transaction_action(
    tx_id: str,
    payload: TransactionActionRequest,
    db: Session = Depends(get_db)
):
    """
    Executes defensive action on transaction:
    - DISPATCH_MFA: Step-up authentication
    - ESCALATE: Sent to senior fraud supervisor
    - REJECT: Hard block
    - OVERRIDE: Human supervisor clearance
    """
    tx = get_or_create_transaction(db, tx_id)

    action = payload.action.upper()
    if action == "DISPATCH_MFA":
        tx.status = "PENDING_MFA"
    elif action in ("REJECT", "BLOCK", "SUSPEND"):
        tx.status = "BLOCKED"
    elif action == "ESCALATE":
        tx.status = "ESCALATED"
    elif action in ("OVERRIDE", "APPROVE", "CLEAR"):
        tx.status = "AUTHORIZED"
    else:
        tx.status = action

    # Record verification event
    v_event = VerificationEvent(
        call_id=tx.call_id or "UNKNOWN",
        method=f"OPERATOR_{action}",
        result=tx.status,
        operator_notes=payload.notes,
        timestamp=datetime.datetime.utcnow()
    )
    db.add(v_event)
    db.commit()
    db.refresh(tx)

    logger.info(f"Transaction {tx_id} action {action} applied. Status: {tx.status}")
    return tx
