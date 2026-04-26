"""
Federated Learning Aggregator (Sprint 4).

Implements FedAvg-style aggregation with optional Differential Privacy noise,
allowing multiple Brazilian banks to train a shared fraud model collaboratively
WITHOUT exchanging raw transaction data (LGPD-compliant).

Each participating institution:
1. Trains a local model on its own data
2. Sends ONLY model weights (or weight deltas) to the aggregator
3. Receives back the aggregated global model

This is a coordinator-side simulator: in production, replace the in-memory
participant registry with secure mTLS RPC (Flower / TensorFlow Federated).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ParticipantUpdate:
    """One participant's local model update for federated aggregation."""

    participant_id: str
    weights: np.ndarray  # flattened model weights / deltas
    n_samples: int  # used for FedAvg weighting
    metadata: Dict[str, float] = field(default_factory=dict)


class FederatedAggregator:
    """FedAvg aggregator with optional differential privacy.

    Algorithm:
        global_weights = sum_i (n_i / N_total) * local_weights_i

    With DP enabled:
        Adds Gaussian noise calibrated by epsilon (privacy budget).
    """

    def __init__(
        self,
        differential_privacy: bool = False,
        dp_epsilon: float = 1.0,
        dp_clip_norm: float = 1.0,
        min_participants: int = 3,
    ):
        """
        Args:
            differential_privacy: Enable DP noise on aggregated weights
            dp_epsilon: Privacy budget (smaller = more noise = more private)
            dp_clip_norm: L2 norm clipping for individual updates (DP requirement)
            min_participants: Minimum participants required for aggregation
                              (prevents inference attacks from too few participants)
        """
        if dp_epsilon <= 0:
            raise ValueError("dp_epsilon must be positive")
        if dp_clip_norm <= 0:
            raise ValueError("dp_clip_norm must be positive")
        if min_participants < 2:
            raise ValueError("min_participants must be >= 2 for federated learning")

        self.differential_privacy = differential_privacy
        self.dp_epsilon = dp_epsilon
        self.dp_clip_norm = dp_clip_norm
        self.min_participants = min_participants
        self.round_number = 0
        self.history: List[Dict] = []

    # ------------------------------------------------------------ DP utils

    def _clip_update(self, weights: np.ndarray) -> np.ndarray:
        """Clip update L2 norm to dp_clip_norm (required for DP guarantees)."""
        norm = float(np.linalg.norm(weights))
        if norm > self.dp_clip_norm:
            return weights * (self.dp_clip_norm / (norm + 1e-9))
        return weights

    def _add_dp_noise(self, weights: np.ndarray) -> np.ndarray:
        """Add Gaussian noise calibrated to epsilon-DP."""
        # Simplified Gaussian mechanism: sigma = clip / epsilon
        sigma = self.dp_clip_norm / self.dp_epsilon
        noise = np.random.normal(loc=0.0, scale=sigma, size=weights.shape)
        return weights + noise

    # --------------------------------------------------------- aggregation

    def aggregate(
        self, updates: List[ParticipantUpdate]
    ) -> Dict[str, object]:
        """Aggregate participant updates via weighted FedAvg.

        Args:
            updates: List of ParticipantUpdate from N institutions

        Returns:
            Dict with aggregated_weights, round_number, n_participants, total_samples
        """
        if len(updates) < self.min_participants:
            raise ValueError(
                f"Need at least {self.min_participants} participants for aggregation; "
                f"got {len(updates)}"
            )

        # Validate consistent shapes
        ref_shape = updates[0].weights.shape
        for u in updates:
            if u.weights.shape != ref_shape:
                raise ValueError(
                    f"All participant updates must have same shape; "
                    f"got {u.weights.shape} vs {ref_shape}"
                )

        # Optional clipping for DP guarantees
        if self.differential_privacy:
            clipped = [self._clip_update(u.weights) for u in updates]
        else:
            clipped = [u.weights for u in updates]

        # FedAvg: weighted by sample count
        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            raise ValueError("Total sample count across participants is zero")

        aggregated = np.zeros(ref_shape)
        for u, w in zip(updates, clipped):
            aggregated += (u.n_samples / total_samples) * w

        # Optional DP noise on aggregated weights
        if self.differential_privacy:
            aggregated = self._add_dp_noise(aggregated)

        self.round_number += 1
        result = {
            "aggregated_weights": aggregated,
            "round_number": self.round_number,
            "n_participants": len(updates),
            "total_samples": total_samples,
            "differential_privacy": self.differential_privacy,
            "dp_epsilon": self.dp_epsilon if self.differential_privacy else None,
            "participant_ids": [u.participant_id for u in updates],
        }

        # Track history (without weights for privacy)
        self.history.append(
            {
                "round": self.round_number,
                "n_participants": len(updates),
                "total_samples": total_samples,
                "dp_enabled": self.differential_privacy,
            }
        )
        return result

    def get_history(self) -> List[Dict]:
        """Return aggregation history (no weights stored)."""
        return list(self.history)
