"""Tests for StackingFraudModel (Sprint 2)."""

import numpy as np
import pandas as pd
import pytest

from src.repositories import EnsembleModelRepository
from src.stacking_model import StackingFraudModel


class InMemoryRepository(EnsembleModelRepository):
    """In-memory repository for fast testing (DI mock)."""

    def __init__(self):
        self._bundle = None

    def load_bundle(self) -> dict:
        if self._bundle is None:
            raise FileNotFoundError("No bundle stored")
        return self._bundle

    def save_bundle(self, bundle: dict) -> None:
        self._bundle = bundle

    def exists(self) -> bool:
        return self._bundle is not None


class TestStackingFraudModel:
    @pytest.fixture
    def sample_data(self):
        """Generate synthetic dataset for fast training."""
        np.random.seed(42)
        n_samples = 500

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
        # Balanced labels for fast SMOTE
        labels = ([0] * 400) + ([1] * 100)
        np.random.shuffle(labels)
        df["fraudResult"] = labels
        df["payload"] = "{}"
        return df

    def test_initialization(self):
        model = StackingFraudModel(model_path="nonexistent_stacking.pkl")
        assert model.stacking_model is None
        assert model.threshold == 0.5
        assert model.n_folds == 3

    def test_train_and_predict(self, sample_data, tmp_path):
        model_path = str(tmp_path / "test_stack.pkl")
        model = StackingFraudModel(model_path=model_path, n_folds=2)
        metrics = model.train(sample_data)

        assert model.stacking_model is not None
        assert len(model.feature_names) > 0
        assert "auc_roc" in metrics
        assert 0.0 <= metrics["auc_roc"] <= 1.0

        # Predict
        sample_features = sample_data.drop(columns=["fraudResult", "payload"]).head(1)
        proba, is_fraud, _explanation = model.predict(
            sample_features, transaction_id="test-tx"
        )
        assert 0.0 <= proba <= 1.0
        assert isinstance(is_fraud, (bool, np.bool_))

    def test_save_and_load_via_repository(self, sample_data):
        repo = InMemoryRepository()
        model = StackingFraudModel(repository=repo, n_folds=2)
        model.train(sample_data)
        original_threshold = model.threshold
        original_features = model.feature_names

        # Reload via fresh model with same repo
        model2 = StackingFraudModel(repository=repo)
        assert model2.threshold == original_threshold
        assert model2.feature_names == original_features
        assert model2.stacking_model is not None

    def test_select_threshold_priority(self, sample_data, tmp_path):
        model_path = str(tmp_path / "test_stack2.pkl")
        model = StackingFraudModel(model_path=model_path, n_folds=2)
        model.train(sample_data)

        model.thresholds_by_product = {"produto_pix": 0.3}
        model.thresholds_by_channel = {"canal_app": 0.4}
        model.threshold = 0.5

        # Product wins over channel
        row = pd.Series({"produto_pix": 1, "canal_app": 1})
        assert model._select_threshold(row) == 0.3

        # Channel when no product match
        row2 = pd.Series({"produto_pix": 0, "canal_app": 1})
        assert model._select_threshold(row2) == 0.4

        # Global default fallback
        row3 = pd.Series({"produto_pix": 0, "canal_app": 0})
        assert model._select_threshold(row3) == 0.5

    def test_feature_importance(self, sample_data, tmp_path):
        model_path = str(tmp_path / "test_stack3.pkl")
        model = StackingFraudModel(model_path=model_path, n_folds=2)
        model.train(sample_data)

        importance = model.get_feature_importance()
        assert len(importance) == len(model.feature_names)
        assert all(v >= 0 for v in importance.values())
