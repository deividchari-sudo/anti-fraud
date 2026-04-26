"""Tests for Sprint 4: FederatedAggregator + EpsilonGreedyThresholdSelector."""

import numpy as np
import pytest

from src.federated_aggregator import FederatedAggregator, ParticipantUpdate
from src.rl_threshold import EpsilonGreedyThresholdSelector


class TestFederatedAggregator:
    def test_initialization_validates_args(self):
        with pytest.raises(ValueError, match="dp_epsilon"):
            FederatedAggregator(dp_epsilon=-0.1)
        with pytest.raises(ValueError, match="dp_clip_norm"):
            FederatedAggregator(dp_clip_norm=0)
        with pytest.raises(ValueError, match="min_participants"):
            FederatedAggregator(min_participants=1)

    def test_aggregate_requires_min_participants(self):
        agg = FederatedAggregator(min_participants=3)
        updates = [
            ParticipantUpdate("bank-A", np.ones(10), n_samples=100),
            ParticipantUpdate("bank-B", np.ones(10), n_samples=100),
        ]
        with pytest.raises(ValueError, match="at least 3"):
            agg.aggregate(updates)

    def test_aggregate_simple_fedavg(self):
        agg = FederatedAggregator(min_participants=2)
        # Equal weights, equal samples -> simple average
        updates = [
            ParticipantUpdate("A", np.array([1.0, 2.0, 3.0]), n_samples=100),
            ParticipantUpdate("B", np.array([3.0, 4.0, 5.0]), n_samples=100),
        ]
        result = agg.aggregate(updates)
        np.testing.assert_array_almost_equal(
            result["aggregated_weights"], np.array([2.0, 3.0, 4.0])
        )
        assert result["n_participants"] == 2
        assert result["round_number"] == 1

    def test_aggregate_weighted_by_sample_count(self):
        agg = FederatedAggregator(min_participants=2)
        updates = [
            ParticipantUpdate("A", np.array([10.0]), n_samples=900),  # 90% weight
            ParticipantUpdate("B", np.array([0.0]), n_samples=100),  # 10% weight
        ]
        result = agg.aggregate(updates)
        np.testing.assert_almost_equal(result["aggregated_weights"][0], 9.0)

    def test_aggregate_validates_consistent_shapes(self):
        agg = FederatedAggregator(min_participants=2)
        updates = [
            ParticipantUpdate("A", np.ones(10), n_samples=100),
            ParticipantUpdate("B", np.ones(5), n_samples=100),
        ]
        with pytest.raises(ValueError, match="same shape"):
            agg.aggregate(updates)

    def test_aggregate_with_dp_adds_noise(self):
        agg = FederatedAggregator(
            differential_privacy=True,
            dp_epsilon=1.0,
            dp_clip_norm=1.0,
            min_participants=2,
        )
        np.random.seed(0)
        updates = [
            ParticipantUpdate("A", np.array([0.5, 0.5]), n_samples=100),
            ParticipantUpdate("B", np.array([0.5, 0.5]), n_samples=100),
        ]
        result = agg.aggregate(updates)
        # With DP, result is noised - should NOT exactly equal [0.5, 0.5]
        diff = np.abs(result["aggregated_weights"] - np.array([0.5, 0.5])).sum()
        assert diff > 0.0
        assert result["differential_privacy"] is True
        assert result["dp_epsilon"] == 1.0

    def test_history_tracks_rounds(self):
        agg = FederatedAggregator(min_participants=2)
        updates = [
            ParticipantUpdate("A", np.ones(3), n_samples=50),
            ParticipantUpdate("B", np.ones(3), n_samples=50),
        ]
        agg.aggregate(updates)
        agg.aggregate(updates)
        history = agg.get_history()
        assert len(history) == 2
        assert history[0]["round"] == 1
        assert history[1]["round"] == 2
        # History should NOT contain raw weights (privacy)
        assert all("weights" not in h for h in history)


class TestEpsilonGreedyThresholdSelector:
    def test_initialization_validates_args(self):
        with pytest.raises(ValueError, match="epsilon"):
            EpsilonGreedyThresholdSelector(epsilon=-0.1)
        with pytest.raises(ValueError, match="reward_window"):
            EpsilonGreedyThresholdSelector(reward_window=0)

    def test_select_threshold_returns_candidate(self):
        sel = EpsilonGreedyThresholdSelector()
        thr = sel.select_threshold()
        assert thr in sel.candidate_thresholds

    def test_update_reward_persists(self):
        sel = EpsilonGreedyThresholdSelector()
        sel.update_reward(0.5, 1.0)
        assert sel.rewards_by_arm[0.5][-1] == 1.0

    def test_exploitation_picks_best_arm(self):
        sel = EpsilonGreedyThresholdSelector(epsilon=0.0)  # pure exploit
        # Reward arm 0.3 highly, others negatively
        for _ in range(50):
            sel.update_reward(0.3, 1.0)
        sel.update_reward(0.5, -1.0)
        sel.update_reward(0.7, -1.0)

        # With epsilon=0, must pick arm with highest mean reward
        chosen = [sel.select_threshold() for _ in range(20)]
        assert all(c == 0.3 for c in chosen)

    def test_exploration_with_epsilon_one_random(self):
        sel = EpsilonGreedyThresholdSelector(epsilon=1.0, random_state=42)
        # Pure exploration - chosen values should vary
        choices = {sel.select_threshold() for _ in range(100)}
        assert len(choices) > 1

    def test_unknown_threshold_snaps_to_nearest(self):
        sel = EpsilonGreedyThresholdSelector()
        sel.update_reward(0.55, 1.0)  # not a candidate; should snap to 0.5 or 0.6
        # One of those arms should now have the reward
        snapped_to_05 = len(sel.rewards_by_arm[0.5]) == 1
        snapped_to_06 = len(sel.rewards_by_arm[0.6]) == 1
        assert snapped_to_05 or snapped_to_06

    def test_get_arm_stats_returns_per_arm(self):
        sel = EpsilonGreedyThresholdSelector()
        sel.update_reward(0.3, 1.0)
        sel.update_reward(0.3, 0.5)
        stats = sel.get_arm_stats()
        assert stats[0.3]["n_samples"] == 2
        assert stats[0.3]["mean_reward"] == 0.75
        assert stats[0.5]["n_samples"] == 0

    def test_best_threshold_returns_highest_mean_reward(self):
        sel = EpsilonGreedyThresholdSelector()
        sel.update_reward(0.2, 0.1)
        for _ in range(10):
            sel.update_reward(0.7, 1.0)
        sel.update_reward(0.5, -0.5)
        assert sel.best_threshold() == 0.7
