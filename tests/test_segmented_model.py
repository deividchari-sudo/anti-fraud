"""Tests for SegmentedModelRepository and SegmentedPredictionService."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.segmented_model import (
    SegmentedModelRepository,
    SegmentedPredictionService,
)


@pytest.fixture
def mock_global_model():
    """Return a mock FraudDetectionModel with predictable predict()."""
    m = MagicMock()
    m.threshold = 0.5
    m.predict.return_value = (0.85, True, {"top_contributing_features": {"valor": 0.3}})
    return m


@pytest.fixture
def mock_feature_engineer():
    """Return a mock FeatureEngineer."""
    m = MagicMock()
    m.extract_features.return_value = {"feature_a": 1.0}
    m.prepare_dataframe.return_value = pd.DataFrame({"feature_a": [1.0]})
    return m


class TestSegmentedModelRepository:
    """Unit tests for the segmented model router / cache."""

    def test_segment_key_normalization(self, mock_global_model):
        repo = SegmentedModelRepository(global_model=mock_global_model)
        assert repo._segment_key("PIX", "Mobile") == "pix_mobile"
        assert repo._segment_key("  TED  ", " WEB ") == "ted_web"

    def test_has_segment_model_when_missing(self, mock_global_model, tmp_path):
        repo = SegmentedModelRepository(
            global_model=mock_global_model, models_dir=str(tmp_path)
        )
        assert repo.has_segment_model("pix", "mobile") is False

    def test_has_segment_model_when_present(self, mock_global_model, tmp_path):
        repo = SegmentedModelRepository(
            global_model=mock_global_model, models_dir=str(tmp_path)
        )
        # Create dummy pkl file following naming convention
        (tmp_path / "fraud_model_pix_mobile.pkl").write_bytes(b"dummy")
        assert repo.has_segment_model("pix", "mobile") is True

    def test_available_segments(self, mock_global_model, tmp_path):
        repo = SegmentedModelRepository(
            global_model=mock_global_model, models_dir=str(tmp_path)
        )
        (tmp_path / "fraud_model_pix_mobile.pkl").write_bytes(b"dummy")
        (tmp_path / "fraud_model_cartao_web.pkl").write_bytes(b"dummy")
        segs = repo.available_segments()
        assert set(segs) == {"pix_mobile", "cartao_web"}

    def test_get_model_strategy_global(self, mock_global_model):
        repo = SegmentedModelRepository(global_model=mock_global_model)
        model, source = repo.get_model("pix", "mobile", strategy="global")
        assert model is mock_global_model
        assert source == "global"
        mock_global_model.predict.assert_not_called()

    def test_get_model_strategy_auto_fallback(self, mock_global_model, tmp_path):
        repo = SegmentedModelRepository(
            global_model=mock_global_model, models_dir=str(tmp_path)
        )
        model, source = repo.get_model("pix", "mobile", strategy="auto")
        assert model is mock_global_model
        assert source == "global"

    def test_get_model_strategy_specialized_raises_when_missing(
        self, mock_global_model, tmp_path
    ):
        repo = SegmentedModelRepository(
            global_model=mock_global_model, models_dir=str(tmp_path)
        )
        with pytest.raises(FileNotFoundError) as exc:
            repo.get_model("pix", "mobile", strategy="specialized")
        assert "pix_mobile" in str(exc.value)

    def test_cache_info(self, mock_global_model, tmp_path):
        repo = SegmentedModelRepository(
            global_model=mock_global_model, models_dir=str(tmp_path)
        )
        (tmp_path / "fraud_model_pix_mobile.pkl").write_bytes(b"dummy")
        info = repo.cache_info()
        assert info["cached_segments"] == []
        assert info["available_segments"] == ["pix_mobile"]


class TestSegmentedPredictionService:
    """Unit tests for the high-level prediction service."""

    def test_predict_routing_and_response_shape(
        self, mock_global_model, mock_feature_engineer
    ):
        router = SegmentedModelRepository(global_model=mock_global_model)
        svc = SegmentedPredictionService(
            router=router, feature_engineer=mock_feature_engineer
        )
        tx = {
            "id": "tx-001",
            "produto": "pix",
            "canal": "mobile",
            "valor": 1000.0,
        }
        result = svc.predict(tx, transaction_id="tx-001", strategy="global")

        assert result["transaction_id"] == "tx-001"
        assert result["fraud_probability"] == 0.85
        assert result["is_fraud"] is True
        assert result["confidence"] == "high"
        assert result["segment"] == "pix_mobile"
        assert result["model_used"] == "global"
        assert result["threshold_used"] == 0.5
        assert "processing_time_ms" in result
        assert "timestamp" in result

    def test_predict_batch(self, mock_global_model, mock_feature_engineer):
        router = SegmentedModelRepository(global_model=mock_global_model)
        svc = SegmentedPredictionService(
            router=router, feature_engineer=mock_feature_engineer
        )
        txs = [
            {"id": "tx-001", "produto": "pix", "canal": "mobile"},
            {"id": "tx-002", "produto": "ted", "canal": "web"},
        ]
        results = svc.predict_batch(txs, strategy="global")
        assert len(results) == 2
        assert results[0]["transaction_id"] == "tx-001"
        assert results[1]["transaction_id"] == "tx-002"

    def test_predict_with_segmented_model(self, mock_global_model, mock_feature_engineer, tmp_path):
        """Test that auto strategy prefers specialized model when present."""
        # Create a real pkl file so has_segment_model returns True.
        (tmp_path / "fraud_model_pix_mobile.pkl").write_bytes(b"dummy")

        # Patch FraudDetectionModel.__init__ to avoid loading the dummy bytes as joblib.
        mock_specialized = MagicMock()
        mock_specialized.threshold = 0.7
        mock_specialized.predict.return_value = (0.92, True, {"shap": 0.5})

        with patch(
            "src.segmented_model.FraudDetectionModel",
            return_value=mock_specialized,
        ):
            router = SegmentedModelRepository(
                global_model=mock_global_model, models_dir=str(tmp_path)
            )
            svc = SegmentedPredictionService(
                router=router, feature_engineer=mock_feature_engineer
            )
            tx = {"id": "tx-001", "produto": "pix", "canal": "mobile"}
            result = svc.predict(tx, transaction_id="tx-001", strategy="auto")

        assert result["model_used"] == "specialized"
        assert result["threshold_used"] == 0.7
        assert result["fraud_probability"] == 0.92

    def test_predict_unknown_product_channel(self, mock_global_model, mock_feature_engineer):
        router = SegmentedModelRepository(global_model=mock_global_model)
        svc = SegmentedPredictionService(
            router=router, feature_engineer=mock_feature_engineer
        )
        tx = {
            "id": "tx-003",
            "produto": "cheque",
            "canal": "whatsapp",
        }
        result = svc.predict(tx, transaction_id="tx-003")
        assert result["segment"] == "cheque_whatsapp"
        assert result["model_used"] == "global"

    def test_confidence_levels(self):
        assert SegmentedPredictionService._confidence_level(0.9) == "high"
        assert SegmentedPredictionService._confidence_level(0.7) == "medium"
        assert SegmentedPredictionService._confidence_level(0.3) == "low"
