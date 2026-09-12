"""
SQLAlchemy Database Models for SIH26104 Voice Integrity & Fraud Prevention Gateway
Tables: users, voice_profiles, calls, risk_events, transactions, verification_events, alerts
"""

import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from backend.app.database import Base

class User(Base):
    """Registered institutional user (e.g. Executives, Finance Officers, Authorized Personnel)."""
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    role = Column(String(64), nullable=False)  # e.g., "CFO", "VP Finance", "Director"
    organization = Column(String(128), default="Institutional Enterprise")
    registered_phone = Column(String(32), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    voice_profile = relationship("VoiceProfile", back_populates="user", uselist=False)


class VoiceProfile(Base):
    """Enrolled biometric voice profile with reference acoustic embeddings."""
    __tablename__ = "voice_profiles"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, unique=True)
    embedding = Column(JSON, nullable=False)  # List[float] representing 256-dim embedding vector
    language = Column(String(16), default="en")  # "en", "hi", "ta"
    sample_count = Column(Integer, default=1)
    enrolled_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="voice_profile")


class Call(Base):
    """Live or historical incoming voice interaction session."""
    __tablename__ = "calls"

    id = Column(String(64), primary_key=True, index=True)
    caller_id = Column(String(32), index=True, nullable=False)  # Inbound caller phone number
    claimed_identity = Column(String(128), nullable=True)       # e.g., "CFO" or user_id
    channel = Column(String(32), default="VOIP_SIP")             # "TELEPHONY_PSTN", "VOIP_SIP", "BROWSER_MIC"
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String(32), default="IN_PROGRESS")          # "IN_PROGRESS", "TERMINATED", "BLOCKED"
    final_risk_score = Column(Float, default=0.0)
    final_risk_level = Column(String(16), default="LOW")

    risk_events = relationship("RiskEvent", back_populates="call", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="call", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="call", cascade="all, delete-orphan")


class RiskEvent(Base):
    """Granular risk assessment for each 1-3 second audio chunk."""
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(64), ForeignKey("calls.id"), nullable=False, index=True)
    chunk_index = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Sub-component scores in [0.0, 1.0]
    synthetic_score = Column(Float, nullable=False)
    speaker_similarity = Column(Float, nullable=False)
    speaker_mismatch = Column(Float, nullable=False)
    behavior_score = Column(Float, nullable=False)
    context_score = Column(Float, nullable=False)
    replay_score = Column(Float, default=0.0)
    
    # Aggregate dynamic risk score in [0.0, 100.0]
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)  # "LOW", "MEDIUM", "HIGH"
    
    # Explainable tags (e.g. ["HIGH_SYNTHETIC_PROBABILITY", "LOW_SPEAKER_SIMILARITY", "UNREGISTERED_CALLER"])
    reasons = Column(JSON, default=list)

    call = relationship("Call", back_populates="risk_events")


class Transaction(Base):
    """Financial or sensitive institutional action requested during the call."""
    __tablename__ = "transactions"

    id = Column(String(64), primary_key=True, index=True)
    call_id = Column(String(64), ForeignKey("calls.id"), nullable=False, index=True)
    amount = Column(Float, nullable=False)                       # e.g., 2500000.0 (₹25,00,000)
    currency = Column(String(8), default="INR")
    transaction_type = Column(String(64), default="WIRE_TRANSFER")
    beneficiary = Column(String(128), nullable=False)            # Beneficiary account/name
    is_new_beneficiary = Column(Boolean, default=True)
    status = Column(String(32), default="PENDING")               # "PENDING", "AUTHORIZED", "BLOCKED", "ESCALATED"
    risk_score_at_request = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    call = relationship("Call", back_populates="transactions")


class VerificationEvent(Base):
    """Secondary authentication challenge event (e.g. Registered Callback, Push MFA)."""
    __tablename__ = "verification_events"

    id = Column(String(64), primary_key=True, index=True)
    call_id = Column(String(64), ForeignKey("calls.id"), nullable=False, index=True)
    method = Column(String(64), nullable=False)                  # "REGISTERED_CALLBACK", "OUT_OF_BAND_MFA", "SUPERVISOR_OVERRIDE"
    result = Column(String(32), default="PENDING")               # "PASSED", "FAILED", "PENDING"
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Alert(Base):
    """Role-based security alert triggered by impersonation threat."""
    __tablename__ = "alerts"

    id = Column(String(64), primary_key=True, index=True)
    call_id = Column(String(64), ForeignKey("calls.id"), nullable=False, index=True)
    recipient_role = Column(String(32), nullable=False)          # "AGENT", "SUPERVISOR", "FRAUD_OPS"
    severity = Column(String(16), nullable=False)                # "INFO", "WARNING", "CRITICAL"
    message = Column(Text, nullable=False)
    status = Column(String(32), default="ACTIVE")                # "ACTIVE", "ACKNOWLEDGED", "DISMISSED"
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    call = relationship("Call", back_populates="alerts")