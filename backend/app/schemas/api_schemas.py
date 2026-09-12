"""
VoiceGuard API Pydantic Schemas
Harmonized with SQLAlchemy schema_models.py
"""

import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

class ORMBaseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class UserProfileResponse(ORMBaseModel):
    id: str
    name: str
    role: str
    organization: Optional[str] = None
    registered_phone: str
    is_enrolled: bool = False
    created_at: Optional[datetime.datetime] = None

class CallStartRequest(BaseModel):
    caller_id: str = Field(..., description="Phone number or SIP URI of caller")
    claimed_identity: Optional[str] = Field(None, description="User ID or Name claimed")
    channel: str = Field(default="TELEPHONY_PSTN", description="Call channel (PSTN, VOIP_SIP, BROWSER_MIC)")

class CallResponse(ORMBaseModel):
    id: str
    caller_id: str
    claimed_identity: Optional[str] = None
    channel: str
    status: str
    final_risk_score: float
    final_risk_level: str
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None

class RiskEventResponse(ORMBaseModel):
    id: Optional[int] = None
    call_id: str
    chunk_index: int = 0
    timestamp: datetime.datetime
    synthetic_score: float
    speaker_similarity: float
    speaker_mismatch: float
    behavior_score: float
    context_score: float
    replay_score: float
    risk_score: float
    risk_level: str
    reasons: Optional[List[str]] = None

class CallSummaryResponse(BaseModel):
    id: str
    caller_id: str
    claimed_identity: Optional[str] = None
    channel: str
    status: str
    final_risk_score: float
    final_risk_level: str
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    risk_events_count: int
    recent_events: List[RiskEventResponse] = []

class TransactionRequest(BaseModel):
    call_id: str
    amount: float
    currency: str = "INR"
    beneficiary: str
    is_new_beneficiary: bool = True
    transaction_type: str = "WIRE_TRANSFER"

class TransactionResponse(ORMBaseModel):
    id: str
    call_id: str
    amount: float
    currency: str
    transaction_type: str
    beneficiary: str
    is_new_beneficiary: bool
    status: str
    risk_score_at_request: float
    created_at: Optional[datetime.datetime] = None

class TransactionActionRequest(BaseModel):
    action: str  # VERIFY_MFA, REGISTERED_CALLBACK, ESCALATE, REJECT, OVERRIDE_APPROVE
    notes: Optional[str] = None

class AlertResponse(ORMBaseModel):
    id: str
    call_id: str
    recipient_role: str
    severity: str
    message: str
    status: str
    timestamp: Optional[datetime.datetime] = None