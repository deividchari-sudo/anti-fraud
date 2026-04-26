"""Tests for OpenFinanceFeatureExtractor (Sprint 2)."""

import pytest

from src.open_finance_features import OpenFinanceFeatureExtractor


class TestOpenFinanceFeatureExtractor:
    def test_extract_features_returns_all_keys(self):
        extractor = OpenFinanceFeatureExtractor()
        features = extractor.extract_features("12345678901")

        expected = [
            "of_n_accounts_other",
            "of_total_balance",
            "of_avg_ticket",
            "of_recent_failures",
            "of_credit_score",
            "of_income_stability",
            "of_days_first_relationship",
            "of_spending_divergence",
            "of_reported_fraud_other",
        ]
        for key in expected:
            assert key in features

    def test_features_are_deterministic_for_same_cpf(self):
        extractor = OpenFinanceFeatureExtractor()
        f1 = extractor.extract_features("12345678901")
        f2 = extractor.extract_features("12345678901")
        assert f1 == f2

    def test_features_differ_for_different_cpfs(self):
        extractor = OpenFinanceFeatureExtractor()
        f1 = extractor.extract_features("12345678901")
        f2 = extractor.extract_features("98765432109")
        # At least some features should differ
        assert any(f1[k] != f2[k] for k in f1)

    def test_empty_cpf_returns_default_features(self):
        extractor = OpenFinanceFeatureExtractor()
        features = extractor.extract_features(None)
        assert features["of_credit_score"] == 500.0  # neutral midpoint
        assert features["of_n_accounts_other"] == 0.0

    def test_credit_score_in_valid_range(self):
        extractor = OpenFinanceFeatureExtractor()
        for cpf in ["11122233344", "55566677788", "99988877766"]:
            features = extractor.extract_features(cpf)
            assert 300 <= features["of_credit_score"] <= 1000

    def test_income_stability_in_valid_range(self):
        extractor = OpenFinanceFeatureExtractor()
        for cpf in ["11122233344", "55566677788", "99988877766"]:
            features = extractor.extract_features(cpf)
            assert 0 <= features["of_income_stability"] <= 1

    def test_feature_names_match_extracted_keys(self):
        extractor = OpenFinanceFeatureExtractor()
        features = extractor.extract_features("12345678901")
        names = extractor.feature_names()
        assert set(names) == set(features.keys())

    def test_seed_offset_changes_features(self):
        e1 = OpenFinanceFeatureExtractor(seed_offset=0)
        e2 = OpenFinanceFeatureExtractor(seed_offset=999)
        f1 = e1.extract_features("12345678901")
        f2 = e2.extract_features("12345678901")
        # Some features should differ with different offsets
        assert any(f1[k] != f2[k] for k in f1)
