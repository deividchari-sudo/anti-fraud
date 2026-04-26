"""
Open Finance Features Extractor (Sprint 2 - simulated).

In production, these features would come from Open Finance APIs aggregating
the customer's data across multiple financial institutions (with consent).

Features extracted:
- Number of accounts in other institutions
- Total balance across all institutions
- Average transaction value across all institutions
- Number of recent payment failures (other banks)
- Credit score (aggregated)
- Income stability indicator
- Days since first relationship
- Spending pattern divergence between institutions

This module provides a deterministic simulator based on CPF hash,
allowing reproducible behavior without real Open Finance API access.
"""

import hashlib
from typing import Dict, Optional


class OpenFinanceFeatureExtractor:
    """Extracts Open Finance features for a given CPF (simulated)."""

    def __init__(self, seed_offset: int = 0):
        """
        Args:
            seed_offset: Optional offset for deterministic but variable simulation.
        """
        self.seed_offset = seed_offset

    @staticmethod
    def _cpf_to_seed(cpf: str) -> int:
        """Deterministic seed from CPF (simulates persistent customer profile)."""
        if not cpf:
            return 0
        h = hashlib.sha256(cpf.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def _pseudo_uniform(self, cpf: str, salt: str, low: float, high: float) -> float:
        """Deterministic pseudo-random uniform value derived from CPF + salt."""
        seed = self._cpf_to_seed(cpf + salt) + self.seed_offset
        # Map seed to [0, 1)
        normalized = (seed % 1_000_000) / 1_000_000
        return float(low + normalized * (high - low))

    def extract_features(
        self, cpf: Optional[str], transaction: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract Open Finance features for a CPF.

        Args:
            cpf: Customer CPF (used as deterministic seed)
            transaction: Optional transaction context

        Returns:
            Dictionary of Open Finance features (all numeric, ML-ready)
        """
        if not cpf:
            return self._empty_features()

        # Number of accounts in other institutions (1-5, weighted toward 1-2)
        n_accounts_other = int(self._pseudo_uniform(cpf, "n_accounts", 1, 5))

        # Total balance across all institutions (R$)
        total_balance = self._pseudo_uniform(cpf, "balance", 100, 50000)

        # Average ticket across institutions
        avg_ticket = self._pseudo_uniform(cpf, "avg_ticket", 50, 5000)

        # Recent payment failures in other banks (last 90 days)
        recent_failures = int(self._pseudo_uniform(cpf, "failures", 0, 5))

        # Aggregated credit score (300-1000, like Serasa/Boa Vista)
        credit_score = self._pseudo_uniform(cpf, "credit", 300, 1000)

        # Income stability (0=unstable, 1=stable)
        income_stability = self._pseudo_uniform(cpf, "income", 0, 1)

        # Days since first banking relationship
        days_first_relationship = int(
            self._pseudo_uniform(cpf, "first_rel", 30, 5475)  # up to 15 years
        )

        # Spending pattern divergence (z-score-like, 0=similar, 1=divergent)
        spending_divergence = self._pseudo_uniform(cpf, "divergence", 0, 1)

        # Risk flag: customer has reported fraud in another institution
        # (boolean as 0/1, low probability)
        reported_fraud_other = (
            1 if self._pseudo_uniform(cpf, "reported_fraud", 0, 1) > 0.95 else 0
        )

        return {
            "of_n_accounts_other": float(n_accounts_other),
            "of_total_balance": float(total_balance),
            "of_avg_ticket": float(avg_ticket),
            "of_recent_failures": float(recent_failures),
            "of_credit_score": float(credit_score),
            "of_income_stability": float(income_stability),
            "of_days_first_relationship": float(days_first_relationship),
            "of_spending_divergence": float(spending_divergence),
            "of_reported_fraud_other": float(reported_fraud_other),
        }

    @staticmethod
    def _empty_features() -> Dict[str, float]:
        """Default features when CPF is unavailable (cold start)."""
        return {
            "of_n_accounts_other": 0.0,
            "of_total_balance": 0.0,
            "of_avg_ticket": 0.0,
            "of_recent_failures": 0.0,
            "of_credit_score": 500.0,  # neutral midpoint
            "of_income_stability": 0.5,
            "of_days_first_relationship": 0.0,
            "of_spending_divergence": 0.0,
            "of_reported_fraud_other": 0.0,
        }

    def feature_names(self) -> list:
        """Return list of feature names (useful for ML pipelines)."""
        return list(self._empty_features().keys())
