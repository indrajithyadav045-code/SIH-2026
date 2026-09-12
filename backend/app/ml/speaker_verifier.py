"""
VoiceGuard Speaker Biometric Verifier
Computes cosine similarity between incoming audio chunk embeddings and registered reference embeddings.
Determines identity match confidence and detects biometric voice impersonation.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger("VoiceGuard.SpeakerVerifier")

class SpeakerVerifier:
    """
    Biometric Speaker Comparator using Cosine Distance on unit-normalized 256-d forensic vectors.
    """

    MATCH_THRESHOLD_HIGH = 0.78
    MATCH_THRESHOLD_MEDIUM = 0.60

    def verify_similarity(self, claimed_embedding: List[float], chunk_embedding: List[float]) -> Dict[str, Any]:
        """
        Calculates cosine similarity:
            similarity = (u . v) / (||u|| * ||v||)
        Returns:
            - match_score: float (0.0 to 1.0)
            - mismatch_score: float (0.0 to 1.0)
            - is_match: bool
            - confidence_level: str ("HIGH", "MEDIUM", "LOW_MISMATCH")
        """
        if not claimed_embedding or not chunk_embedding:
            return {
                "match_score": 0.50,
                "mismatch_score": 0.50,
                "is_match": False,
                "confidence_level": "NO_ENROLLMENT_AVAILABLE"
            }

        u = np.array(claimed_embedding, dtype=np.float32)
        v = np.array(chunk_embedding, dtype=np.float32)

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u < 1e-6 or norm_v < 1e-6:
            return {
                "match_score": 0.50,
                "mismatch_score": 0.50,
                "is_match": False,
                "confidence_level": "INVALID_EMBEDDING_ZERO_NORM"
            }

        # Raw cosine similarity [-1.0, 1.0]
        cos_sim = float(np.dot(u, v) / (norm_u * norm_v))

        # Rescale cosine similarity [-0.2, 1.0] to [0.0, 1.0] probability
        # In deep speaker embeddings, orthogonal vectors have cos ~ 0.0, true mismatch is < 0.4
        match_score = max(0.0, min(1.0, (cos_sim + 0.2) / 1.2))
        mismatch_score = 1.0 - match_score

        if cos_sim >= self.MATCH_THRESHOLD_HIGH:
            confidence = "HIGH_CONFIDENCE_MATCH"
            is_match = True
        elif cos_sim >= self.MATCH_THRESHOLD_MEDIUM:
            confidence = "AMBIGUOUS_PARTIAL_MATCH"
            is_match = True
        else:
            confidence = "CRITICAL_SPEAKER_MISMATCH"
            is_match = False

        return {
            "cosine_similarity": round(cos_sim, 4),
            "match_score": round(match_score, 4),
            "mismatch_score": round(mismatch_score, 4),
            "is_match": is_match,
            "confidence_level": confidence
        }