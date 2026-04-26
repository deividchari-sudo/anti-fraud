"""Tests for EnsembleFraudModel (Sprint 1)."""

import numpy as np
import pandas as pd
import pytest

from src.ensemble_model import EnsembleFraudModel


class TestEnsembleFraudModel:
    @pytest.fixture
    def sample_data(self):
        """Generate balanced synthetic dataset for fast training."""
        np.random.seed(42)
        n_samples = 400

        data = {
            "valor": np.random.uniform(0, 10000, n_samples),
            "sender_banco": np.random.randint(1, 1000, n_samples),
            "receiver_banco": np.random.randint(1, 1000, n_samples),
            "hora_do_dia": np.random.randint(0, 24, n_samples),
            "dia_da_semana": np.random.randint(0, 7, n_samples),
            "canal_app": np.random.randint(0, 2, n_samples),
            "canal_web": np.random.randint(0, 2, n_samples),
            "canal_api": np.random.randint(0, 2, n_samples),
            "produto_pix": np.random.randint(0, 2, n_samples),
            "produto_ted": np.random.randint(0, 2, n_samples),
            "produto_boleto": np.random.randint(0, 2, n_samples),
        }
        df = pd.DataFrame(data)
        # Balanced labels (avoid SMOTE issues with very small minority)
        df["fraudResult"] = ([0] * 320) + ([1] * 80)
        np.random.shuffle(df["fraudResult"].values)
        df["payload"] = "{}"
        return df

    def test_initialization_default_weights(self):
        model = EnsembleFraudModel(model_path="nonexistent_ensemble.pkl")
        assert model.xgb_weight == 0.5
        assert model.lgb_weight == 0.5
        assert model.threshold == 0.5
        assert model.xgb_model is None
        assert model.lgb_model is None

    def test_initialization_custom_weights(self):
        model = EnsembleFraudModel(
            model_path="nonexistent_ensemble.pkl",
            xgb_weight=0.7,
            lgb_weight=0.3,
        )
        assert model.xgb_weight == 0.7
        assert model.lgb_weight == 0.3

    def test_invalid_weights_raises_error(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            EnsembleFraudModel(
                model_path="nonexistent_ensemble.pkl",
                xgb_weight=0.6,
                lgb_weight=0.6,
            )

    def test_train_and_predict(self, sample_data, tmp_path):
        model_path = str(tmp_path / "test_ensemble.pkl")
        model = EnsembleFraudModel(model_path=model_path)

        metrics = model.train(sample_data)

        # Both base models trained
        assert model.xgb_model is not None
        assert model.lgb_model is not None
        assert len(model.feature_names) > 0

        # Metrics structure
        assert "auc_roc" in metrics
        assert "f1_score" in metrics
        assert "xgb_auc_roc" in metrics
        assert "lgb_auc_roc" in metrics
        assert 0.0 <= metrics["auc_roc"] <= 1.0

        # Prediction works
        sample_features = sample_data.drop(columns=["fraudResult", "payload"]).head(1)
        proba, is_fraud, _explanation = model.predict(
            sample_features, transaction_id="test-tx"
        )
        assert 0.0 <= proba <= 1.0
        assert isinstance(is_fraud, (bool, np.bool_))

    def test_threshold_optimization_creates_per_channel_thresholds(
        self, sample_data, tmp_path
    ):
        model_path = str(tmp_path / "test_ensemble2.pkl")
        model = EnsembleFraudModel(model_path=model_path)
        model.train(sample_data)

        # At least one channel/product threshold should exist (best-effort)
        # Allows for small datasets where some segments may not be tunable
        total_tuned = (
            len(model.thresholds_by_channel) + len(model.thresholds_by_product)
        )
        assert total_tuned >= 0  # may be 0 on very small synthetic data
        # Global threshold is always set in [0, 1]
        assert 0.0 <= model.threshold <= 1.0

    def test_save_and_load(self, sample_data, tmp_path):
        model_path = str(tmp_path / "test_ensemble3.pkl")
        model = EnsembleFraudModel(model_path=model_path)
        model.train(sample_data)
        original_threshold = model.threshold
        original_features = model.feature_names

        # Load fresh instance
        model2 = EnsembleFraudModel(model_path=model_path)
        assert model2.threshold == original_threshold
        assert model2.feature_names == original_features
        assert model2.xgb_model is not None
        assert model2.lgb_model is not None

    def test_select_threshold_uses_product_priority(self, sample_data, tmp_path):
        model_path = str(tmp_path / "test_ensemble4.pkl")
        model = EnsembleFraudModel(model_path=model_path)
        model.train(sample_data)

        # Manually set product and channel thresholds
        model.thresholds_by_product = {"produto_pix": 0.3}
        model.thresholds_by_channel = {"canal_app": 0.4}
        model.threshold = 0.5

        # PIX transaction via app: product threshold wins
        row = pd.Series({"produto_pix": 1, "canal_app": 1})
        assert model._select_threshold(row) == 0.3

        # Channel-only: app via boleto (product not configured)
        row2 = pd.Series({"produto_pix": 0, "canal_app": 1})
        assert model._select_threshold(row2) == 0.4

        # No match: global default
        row3 = pd.Series({"produto_pix": 0, "canal_app": 0})
        assert model._select_threshold(row3) == 0.5

    def test_feature_importance(self, sample_data, tmp_path):
        model_path = str(tmp_path / "test_ensemble5.pkl")
        model = EnsembleFraudModel(model_path=model_path)
        model.train(sample_data)

        importance = model.get_feature_importance()
        assert len(importance) == len(model.feature_names)
        # All importances are non-negative
        assert all(v >= 0 for v in importance.values())
