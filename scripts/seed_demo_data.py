import sys
import os
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
"""
VoiceGuard Demo Data Seeder (SIH26104)
Populates institutional executives, voiceprints, calls, and CFO INR 25L scenario.
"""

import sys
import os
import uuid
import datetime
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.database import engine, SessionLocal, Base
from backend.app.models.schema_models import (
    User, VoiceProfile, Call, RiskEvent, Transaction, Alert, VerificationEvent
)

def seed():
    print("=" * 70)
    print("VoiceGuard Gateway: Seeding Institutional Demonstration Data")
    print("=" * 70)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Clear existing seed data
        db.query(Alert).delete()
        db.query(VerificationEvent).delete()
        db.query(Transaction).delete()
        db.query(RiskEvent).delete()
        db.query(Call).delete()
        db.query(VoiceProfile).delete()
        db.query(User).delete()
        db.commit()

        # 1. Seed Users
        print("\n[1] Registering Authorized Executives...")
        cfo = User(
            id="USR-CFO-001",
            name="Rajesh Verma",
            role="Chief Financial Officer (CFO)",
            organization="Institutional Treasury Corp",
            registered_phone="+91-98765-43210"
        )
        vp_ops = User(
            id="USR-OPS-002",
            name="Priya Sharma",
            role="VP Global Operations",
            organization="Institutional Treasury Corp",
            registered_phone="+91-98123-45678"
        )
        treasury_dir = User(
            id="USR-TREAS-003",
            name="Vikram Patel",
            role="Director of Payments",
            organization="Institutional Treasury Corp",
            registered_phone="+91-98999-11223"
        )
        db.add_all([cfo, vp_ops, treasury_dir])
        db.commit()
        print(f"    - Enrolled: {cfo.name} ({cfo.role})")
        print(f"    - Enrolled: {vp_ops.name} ({vp_ops.role})")
        print(f"    - Enrolled: {treasury_dir.name} ({treasury_dir.role})")

        # 2. Seed Reference Voice Profiles
        print("\n[2] Enrolling Biometric Voice Profiles (256-dim unit vectors)...")
        np.random.seed(42)
        cfo_emb = np.random.normal(0.0, 1.0, 256)
        cfo_emb = (cfo_emb / np.linalg.norm(cfo_emb)).tolist()

        vp_emb = np.random.normal(0.5, 1.0, 256)
        vp_emb = (vp_emb / np.linalg.norm(vp_emb)).tolist()

        cfo_profile = VoiceProfile(
            id="VP-CFO-001",
            user_id=cfo.id,
            embedding=cfo_emb,
            language="en",
            sample_count=3
        )
        vp_profile = VoiceProfile(
            id="VP-OPS-002",
            user_id=vp_ops.id,
            embedding=vp_emb,
            language="en",
            sample_count=2
        )
        db.add_all([cfo_profile, vp_profile])
        db.commit()
        print("    - Enrolled reference voiceprint for Rajesh Verma (CFO)")
        print("    - Enrolled reference voiceprint for Priya Sharma (VP Ops)")

        # 3. Seed Demonstrative Calls
        print("\n[3] Generating Demonstrative Calls & Scenarios...")
        # Scenario A: Legitimate Call
        call_legit = Call(
            id="CALL-LEGIT-001",
            caller_id="+91-98765-43210",
            claimed_identity=cfo.name,
            channel="TELEPHONY_PSTN",
            status="TERMINATED",
            final_risk_score=14.2,
            final_risk_level="LOW",
            start_time=datetime.datetime.utcnow() - datetime.timedelta(hours=2),
            end_time=datetime.datetime.utcnow() - datetime.timedelta(hours=1, minutes=55)
        )
        db.add(call_legit)
        db.commit()

        # Scenario B: CFO INR 25,00,000 Voice Cloning Impersonation Attack
        call_attack = Call(
            id="CALL-IMPERSONATE-999",
            caller_id="+91-99999-00000",
            claimed_identity=cfo.name,
            channel="VOIP_SIP",
            status="FLAGGED_THREAT",
            final_risk_score=87.5,
            final_risk_level="HIGH",
            start_time=datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
        )
        db.add(call_attack)
        db.commit()

        # Risk Events for Attack Call
        re1 = RiskEvent(
            call_id="CALL-IMPERSONATE-999",
            chunk_index=1,
            timestamp=datetime.datetime.utcnow() - datetime.timedelta(minutes=8),
            synthetic_score=0.89,
            speaker_similarity=0.31,
            speaker_mismatch=0.69,
            behavior_score=0.75,
            context_score=0.85,
            replay_score=0.40,
            risk_score=87.5,
            risk_level="HIGH",
            reasons=[
                "AI_VOICE_CLONING_DETECTED",
                "BIOMETRIC_VOICEPRINT_MISMATCH",
                "UNREGISTERED_CALLER_ID_SPOOF_RISK",
                "HIGH_VALUE_TRANSACTION: INR 25,00,000.00",
                "FIRST_TIME_UNVERIFIED_BENEFICIARY",
                "EXCEEDED_KILL_SWITCH_THRESHOLD"
            ]
        )
        db.add(re1)

        # High-Stakes Financial Transaction
        tx_cfo_attack = Transaction(
            id="TX-CFO-25L-001",
            call_id="CALL-IMPERSONATE-999",
            amount=2500000.0,
            currency="INR",
            transaction_type="URGENT_WIRE",
            beneficiary="Apex Global Holdings LLC (Unverified Offshore, IFSC: HDFC0009999)",
            is_new_beneficiary=True,
            risk_score_at_request=87.5,
            status="BLOCKED"
        )
        db.add(tx_cfo_attack)

        # Alert
        alert1 = Alert(
            id="ALT-CFO-25L",
            call_id="CALL-IMPERSONATE-999",
            recipient_role="FRAUD_OPS",
            severity="CRITICAL",
            message="CRITICAL TRANSACTION BLOCKED: INR 25,00,000.00 to Apex Global Holdings LLC held. High-confidence AI voice cloning detected (Risk 87.5%).",
            status="ACTIVE"
        )
        db.add(alert1)

        db.commit()
        print(f"    - Attack scenario configured: Call {call_attack.id}")
        print(f"    - Blocked Transaction: {tx_cfo_attack.id} (Amount: INR {tx_cfo_attack.amount:,.2f})")
        print(f"    - Critical Alert: {alert1.id}")

        print("\n" + "=" * 70)
        print("SEEDING COMPLETE: All institutional profiles and demo scenarios ready!")
        print("=" * 70)
    finally:
        db.close()

if __name__ == "__main__":
    seed()