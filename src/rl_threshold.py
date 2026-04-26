"""
Reinforcement Learning Adaptive Threshold (Sprint 4).

Uses an epsilon-greedy multi-armed bandit to dynamically select the optimal
fraud threshold based on real-time analyst feedback.

Each "arm" is a candidate threshold. Reward is the F1-score impact of decisions
made at that threshold (computed from recent labeled feedback).

Production usage: replaces static thresholds with self-adjusting ones that
adapt to concept drift WITHOUT requiring full retraining.
"""

import json
from collections import defaultdict, deque
from pathlib import Path
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
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        state_path: Optional[str] = None,
    ):
        """
        Args:
            candidate_thresholds: Discrete thresholds to try (default: 0.1..0.9)
            epsilon: Initial exploration probability (0..1). 0.1 = 10% exploration
            reward_window: Number of recent rewards to average per arm
            random_state: Random seed for reproducibility
            epsilon_decay: Multiplicative decay per pull (1.0 = no decay)
            epsilon_min: Floor for epsilon after decay
            state_path: Optional path to persist/restore RL state across restarts
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if reward_window < 1:
            raise ValueError("reward_window must be >= 1")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if epsilon_min < 0:
            raise ValueError("epsilon_min must be >= 0")

        self.candidate_thresholds = candidate_thresholds or [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reward_window = reward_window
        self.rng = np.random.default_rng(random_state)
        self.state_path = state_path

        # Track rewards per arm (rolling window)
        self.rewards_by_arm: Dict[float, Deque[float]] = {
            t: deque(maxlen=reward_window) for t in self.candidate_thresholds
        }
        # Track number of times each arm was selected
        self.pull_counts: Dict[float, int] = defaultdict(int)
        self.total_pulls = 0

        # Auto-load persisted state if available
        if state_path and Path(state_path).exists():
            self.load_state()

    # ------------------------------------------------------------- select

    def select_threshold(self) -> float:
        """Choose a threshold via epsilon-greedy strategy with optional decay.

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

        # Decay epsilon (with floor)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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

    # --------------------------------------------------------- persistence

    def save_state(self) -> None:
        """Persist RL state to JSON (survives restarts)."""
        if not self.state_path:
            raise ValueError("state_path was not configured at init")
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "candidate_thresholds": self.candidate_thresholds,
            "epsilon": self.epsilon,
            "epsilon_initial": self.epsilon_initial,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "reward_window": self.reward_window,
            "total_pulls": self.total_pulls,
            "pull_counts": {str(k): v for k, v in self.pull_counts.items()},
            "rewards_by_arm": {
                str(k): list(v) for k, v in self.rewards_by_arm.items()
            },
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_state(self) -> None:
        """Restore RL state from JSON."""
        if not self.state_path or not Path(self.state_path).exists():
            return
        with open(self.state_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.candidate_thresholds = payload.get(
            "candidate_thresholds", self.candidate_thresholds
        )
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.epsilon_initial = payload.get("epsilon_initial", self.epsilon)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)
        self.epsilon_min = payload.get("epsilon_min", self.epsilon_min)
        self.reward_window = payload.get("reward_window", self.reward_window)
        self.total_pulls = payload.get("total_pulls", 0)
        self.pull_counts = defaultdict(
            int, {float(k): v for k, v in payload.get("pull_counts", {}).items()}
        )
        self.rewards_by_arm = {
            float(k): deque(v, maxlen=self.reward_window)
            for k, v in payload.get("rewards_by_arm", {}).items()
        }
        # Ensure all candidate thresholds have a deque
        for t in self.candidate_thresholds:
            if t not in self.rewards_by_arm:
                self.rewards_by_arm[t] = deque(maxlen=self.reward_window)
