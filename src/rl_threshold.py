"""
Reinforcement Learning Adaptive Threshold (Sprint 4).

Uses an epsilon-greedy multi-armed bandit to dynamically select the optimal
fraud threshold based on real-time analyst feedback.

Each "arm" is a candidate threshold. Reward is the F1-score impact of decisions
made at that threshold (computed from recent labeled feedback).

Production usage: replaces static thresholds with self-adjusting ones that
adapt to concept drift WITHOUT requiring full retraining.
"""

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

import numpy as np


class EpsilonGreedyThresholdSelector:
    """Multi-armed bandit for adaptive fraud-threshold selection.

    Arms = candidate thresholds (e.g., 0.1, 0.2, ..., 0.9).
    Reward = F1-impact of decisions at that arm (from analyst feedback).
    """

    def __init__(
        self,
        candidate_thresholds: Optional[List[float]] = None,
        epsilon: float = 0.1,
        reward_window: int = 200,
        random_state: int = 42,
    ):
        """
        Args:
            candidate_thresholds: Discrete thresholds to try (default: 0.1..0.9)
            epsilon: Exploration probability (0..1). 0.1 = 10% exploration
            reward_window: Number of recent rewards to average per arm
            random_state: Random seed for reproducibility
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if reward_window < 1:
            raise ValueError("reward_window must be >= 1")

        self.candidate_thresholds = candidate_thresholds or [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]
        self.epsilon = epsilon
        self.reward_window = reward_window
        self.rng = np.random.default_rng(random_state)

        # Track rewards per arm (rolling window)
        self.rewards_by_arm: Dict[float, Deque[float]] = {
            t: deque(maxlen=reward_window) for t in self.candidate_thresholds
        }
        # Track number of times each arm was selected
        self.pull_counts: Dict[float, int] = defaultdict(int)
        self.total_pulls = 0

    # ------------------------------------------------------------- select

    def select_threshold(self) -> float:
        """Choose a threshold via epsilon-greedy strategy.

        Returns:
            Selected threshold value
        """
        self.total_pulls += 1

        # Explore: random arm with probability epsilon
        if self.rng.random() < self.epsilon:
            chosen = float(self.rng.choice(self.candidate_thresholds))
        else:
            # Exploit: arm with highest mean reward
            mean_rewards = {
                t: (np.mean(r) if len(r) > 0 else 0.0)
                for t, r in self.rewards_by_arm.items()
            }
            chosen = float(max(mean_rewards, key=mean_rewards.get))

        self.pull_counts[chosen] += 1
        return chosen

    # ------------------------------------------------------------- update

    def update_reward(self, threshold: float, reward: float) -> None:
        """Register a reward for a threshold (e.g., from analyst feedback).

        Reward design suggestion:
        +1 if decision was correct (true positive or true negative)
        -1 if false positive
        -2 if false negative (missed fraud is worse for the bank)

        Args:
            threshold: The threshold that was used for the decision
            reward: Numeric reward signal (any range)
        """
        if threshold not in self.rewards_by_arm:
            # Snap to nearest candidate
            threshold = min(
                self.candidate_thresholds,
                key=lambda t: abs(t - threshold),
            )
        self.rewards_by_arm[threshold].append(float(reward))

    # ------------------------------------------------------------- stats

    def get_arm_stats(self) -> Dict[float, Dict[str, float]]:
        """Return current stats per arm (mean reward, n_samples, pulls)."""
        return {
            t: {
                "mean_reward": float(np.mean(r)) if len(r) > 0 else 0.0,
                "n_samples": len(r),
                "pulls": self.pull_counts[t],
            }
            for t, r in self.rewards_by_arm.items()
        }

    def best_threshold(self) -> float:
        """Return the threshold with highest mean reward (for monitoring)."""
        means = {
            t: (float(np.mean(r)) if len(r) > 0 else 0.0)
            for t, r in self.rewards_by_arm.items()
        }
        return float(max(means, key=means.get))
