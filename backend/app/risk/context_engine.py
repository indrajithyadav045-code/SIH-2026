"""
VoiceGuard Transaction & Telephony Context Engine
Evaluates caller ID legitimacy, high-value transaction exposure (> ₹10,00,000),
new beneficiary age, and out-of-band policy adherence.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger("VoiceGuard.ContextEngine")

class ContextEngine:
    """
    Evaluates situational and financial risk context to safeguard high-stake transactions.
    """

    HIGH_VALUE_THRESHOLD = 1000000.0  # ₹10,00,000

    def evaluate(
        self,
        caller_phone: str,
        registered_phone: Optional[str] = None,
        transaction_amount: Optional[float] = None,
        is_new_beneficiary: bool = False,
        is_international_call: bool = False,
        time_of_day_anomaly: bool = False
    ) -> Dict[str, Any]:
        """
        Computes context risk score [0.0, 1.0] and contextual risk tags.
        """
        tags = []
        risk_score = 0.10

        # 1. Caller ID Check
        if registered_phone and caller_phone:
            clean_call = caller_phone.replace("+", "").replace(" ", "").replace("-", "")
            clean_reg = registered_phone.replace("+", "").replace(" ", "").replace("-", "")
            if clean_call != clean_reg:
                risk_score += 0.40
                tags.append("UNREGISTERED_CALLER_ID_SPOOF_RISK")
        elif not caller_phone or caller_phone.lower() in ["anonymous", "unknown", "private"]:
            risk_score += 0.30
            tags.append("CALLER_ID_ANONYMIZED")

        # 2. Transaction Amount
        if transaction_amount is not None:
            if transaction_amount >= self.HIGH_VALUE_THRESHOLD:
                risk_score += 0.30
                tags.append(f"HIGH_VALUE_TRANSACTION: ₹{transaction_amount:,.2f}")
            elif transaction_amount > 200000.0:
                risk_score += 0.15
                tags.append(f"ELEVATED_TRANSACTION_VALUE: ₹{transaction_amount:,.2f}")

        # 3. New Beneficiary
        if is_new_beneficiary:
            risk_score += 0.25
            tags.append("FIRST_TIME_UNVERIFIED_BENEFICIARY")

        # 4. Telephony routing
        if is_international_call:
            risk_score += 0.15
            tags.append("FOREIGN_TELECOM_ROUTING")

        if time_of_day_anomaly:
            risk_score += 0.15
            tags.append("OFF_HOURS_FINANCIAL_INSTRUCTION")

        final_risk = max(0.05, min(0.98, risk_score))
        return {
            "context_risk_score": round(final_risk, 4),
            "context_tags": tags
        }