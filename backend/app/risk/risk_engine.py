"""
VoiceGuard Dynamic Multi-Factor Risk Scoring Engine
Implements the explainable weighted mathematical formulation:
Risk = 100 * (0.35 * Synthetic + 0.25 * SpeakerMismatch + 0.15 * Behavior + 0.15 * Context + 0.10 * Replay)
Categorizes risk into LOW (0-30), MEDIUM (31-70), HIGH (71-100),
triggers the emergency kill-switch at >= 75.0, and recommends defensive actions.
"""

import logging
from typing import Dict, List, Optional, Any
from backend.app.config import settings

logger = logging.getLogger("VoiceGuard.RiskEngine")

class RiskEngine:
    """
    Probabilistic Security Decisioning Engine with Explainable Reason Tags.
    """

    def __init__(self):
        self.w_synth = settings.WEIGHT_SYNTHETIC
        self.w_speaker = settings.WEIGHT_SPEAKER_MISMATCH
        self.w_behavior = settings.WEIGHT_BEHAVIOR
        self.w_context = settings.WEIGHT_CONTEXT
        self.w_replay = settings.WEIGHT_REPLAY

        self.threshold_low = settings.RISK_THRESHOLD_LOW
        self.threshold_high = settings.RISK_THRESHOLD_HIGH
        self.kill_switch_threshold = settings.KILL_SWITCH_THRESHOLD

    def calculate_risk(
        self,
        synthetic_prob: float,
        speaker_mismatch: float,
        behavior_score: float,
        context_score: float,
        replay_prob: float,
        collected_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Computes dynamic risk score, determines categorization, triggers defensive protocol.
        """
        # Weighted sum: 0.0 to 1.0 -> scaled to 0.0 to 100.0
        raw_score = 100.0 * (
            self.w_synth * synthetic_prob +
            self.w_speaker * speaker_mismatch +
            self.w_behavior * behavior_score +
            self.w_context * context_score +
            self.w_replay * replay_prob
        )

        final_risk = round(max(0.0, min(100.0, raw_score)), 1)

        # Categorization
        if final_risk <= self.threshold_low:
            level = "LOW"
            recommended_action = "ALLOW"
            action_description = "Interaction authorized. Baseline risk within institutional tolerances."
        elif final_risk <= self.threshold_high:
            level = "MEDIUM"
            recommended_action = "VERIFY_MFA"
            action_description = "Step-up verification required. Issue Push MFA challenge or registered callback."
        else:
            level = "HIGH"
            recommended_action = "BLOCK_AND_ESCALATE"
            action_description = "High-confidence voice impersonation threat. Block sensitive actions and escalate to Fraud Operations."

        # Emergency Kill-Switch Evaluation
        kill_switch = final_risk >= self.kill_switch_threshold
        if kill_switch:
            recommended_action = "BLOCK_AND_DISCONNECT"
            action_description = "EMERGENCY THREAT KILL-SWITCH TRIGGERED: Immediate transaction freeze and line disconnect."

        # Compile explainability breakdown
        breakdown = {
            "synthetic_component": round(self.w_synth * synthetic_prob * 100.0, 1),
            "speaker_component": round(self.w_speaker * speaker_mismatch * 100.0, 1),
            "behavior_component": round(self.w_behavior * behavior_score * 100.0, 1),
            "context_component": round(self.w_context * context_score * 100.0, 1),
            "replay_component": round(self.w_replay * replay_prob * 100.0, 1)
        }

        # Consolidate human-readable reason tags
        reason_tags = list(set(collected_tags or []))
        if synthetic_prob > 0.60:
            reason_tags.append("AI_VOICE_CLONING_DETECTED")
        if speaker_mismatch > 0.50:
            reason_tags.append("BIOMETRIC_VOICEPRINT_MISMATCH")
        if kill_switch:
            reason_tags.append("EXCEEDED_KILL_SWITCH_THRESHOLD")

        return {
            "risk_score": final_risk,
            "risk_level": level,
            "recommended_action": recommended_action,
            "action_description": action_description,
            "kill_switch_triggered": kill_switch,
            "score_breakdown": breakdown,
            "reason_tags": reason_tags
        }