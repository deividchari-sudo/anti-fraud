"""
Segmented Fraud Detection Model Router.

Lazy-loads specialized models per (product, channel) segment and routes
predictions accordingly. Falls back to a global model when a segment model
is missing or when the strategy demands it.

This is the implementation for the Abordagem B discussed in the squad
ML Experiment workflow: model-level segmentation with fallback.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.model import FraudDetectionModel
from src.repositories import JoblibModelRepository

logger = logging.getLogger("fraud_audit_segmented")


class SegmentedModelRepository:
    """Lazy-loads and caches specialized models per (product, channel) segment.

    File naming convention on disk:
        models/fraud_model_{product}_{channel}.pkl
        models/feature_names_{product}_{channel}.json

    A global model is always required as fallback.
    """

    def __init__(
        self,
        global_model: FraudDetectionModel,
        models_dir: str = "models",
        segment_separator: str = "_",
    ):
        self.global_model = global_model
        self.models_dir = Path(models_dir)
        self.segment_separator = segment_separator
        # in-memory cache: key -> FraudDetectionModel
        self._cache: Dict[str, FraudDetectionModel] = {}

    def _segment_key(self, product: str, channel: str) -> str:
        """Normalize segment key (lowercase, strip)."""
        return f"{product.lower().strip()}{self.segment_separator}{channel.lower().strip()}"

    def _model_path(self, key: str) -> str:
        return str(self.models_dir / f"fraud_model_{key}.pkl")

    def _feature_names_path(self, key: str) -> str:
        return str(self.models_dir / f"feature_names_{key}.json")

    def has_segment_model(self, product: str, channel: str) -> bool:
        """Check whether a persisted model exists for this segment."""
        key = self._segment_key(product, channel)
        pkl = Path(self._model_path(key))
        return pkl.exists()

    def get_model(
        self, product: str, channel: str, strategy: str = "auto"
    ) -> Tuple[FraudDetectionModel, str]:
        """Return (model, model_source) for the given segment.

        Args:
            product: e.g. 'pix', 'ted', 'cartao'
            channel: e.g. 'mobile', 'web', 'atm'
            strategy: 'auto' | 'global' | 'specialized'
                - auto: use specialized if available, else global
                - global: always use global model
                - specialized: use specialized if available, raise if missing

        Returns:
            Tuple of (model, source) where source is 'specialized' or 'global'.
        """
        if strategy == "global":
            return self.global_model, "global"

        key = self._segment_key(product, channel)

        # cache hit
        if key in self._cache:
            return self._cache[key], "specialized"

        # attempt lazy load
        pkl_path = self._model_path(key)
        if Path(pkl_path).exists():
            repo = JoblibModelRepository(
                model_path=pkl_path,
                feature_names_path=self._feature_names_path(key),
            )
            segment_model = FraudDetectionModel(
                model_path=pkl_path, model_repository=repo
            )
            self._cache[key] = segment_model
            logger.info(
                f"Segmented model loaded for {key} from {pkl_path}"
            )
            return segment_model, "specialized"

        if strategy == "specialized":
            raise FileNotFoundError(
                f"Specialized model not found for segment {key} "
                f"(expected {pkl_path})."
            )

        # auto fallback
        logger.debug(
            f"No specialized model for {key}; falling back to global."
        )
        return self.global_model, "global"

    def warm_cache(self, segments: list[Tuple[str, str]]) -> None:
        """Pre-load models for known segments to avoid cold-start latency."""
        for product, channel in segments:
            try:
                self.get_model(product, channel, strategy="auto")
            except FileNotFoundError:
                pass

    def available_segments(self) -> list[str]:
        """List all segment keys that have a persisted model on disk."""
        segments = []
        if not self.models_dir.exists():
            return segments
        for pkl in self.models_dir.glob("fraud_model_*_*.pkl"):
            # e.g. fraud_model_pix_mobile.pkl -> pix_mobile
            stem = pkl.stem.replace("fraud_model_", "")
            segments.append(stem)
        return segments

    def cache_info(self) -> Dict[str, Any]:
        return {
            "cached_segments": list(self._cache.keys()),
            "available_segments": self.available_segments(),
        }


class SegmentedPredictionService:
    """High-level service that extracts features and routes to the right model.

    Mirrors the prediction flow in main.py:
      1. Extract features via FeatureEngineer
      2. Route to segment-specific or global model
      3. Return prediction enriched with segment metadata
    """

    def __init__(
        self,
        router: SegmentedModelRepository,
        feature_engineer: Any,  # FeatureEngineer instance
    ):
        self.router = router
        self.feature_engineer = feature_engineer

    def predict(
        self,
        transaction_dict: Dict[str, Any],
        transaction_id: Optional[str] = None,
        strategy: str = "auto",
    ) -> Dict[str, Any]:
        """Predict fraud for a single transaction with segmentation.

        Returns a dict compatible with FraudPrediction fields plus
        segmentation metadata.
        """
        start_time = time.time()

        product = transaction_dict.get("produto", "unknown")
        channel = transaction_dict.get("canal", "unknown")

        model, model_source = self.router.get_model(product, channel, strategy)

        # Feature extraction (same as global path)
        features = self.feature_engineer.extract_features(transaction_dict)
        df = self.feature_engineer.prepare_dataframe(features)

        # Prediction
        fraud_probability, is_fraud, explanation = model.predict(
            df, transaction_id=transaction_id
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Build response metadata
        segment_key = self.router._segment_key(product, channel)

        return {
            "transaction_id": transaction_id or "unknown",
            "fraud_probability": float(fraud_probability),
            "is_fraud": bool(is_fraud),
            "confidence": self._confidence_level(fraud_probability),
            "processing_time_ms": float(processing_time_ms),
            "timestamp": pd.Timestamp.now().isoformat(),
            "explanation": explanation,
            "segment": segment_key,
            "model_used": model_source,
            "threshold_used": float(model.threshold),
        }

    @staticmethod
    def _confidence_level(probability: float) -> str:
        if probability >= 0.8:
            return "high"
        if probability >= 0.5:
            return "medium"
        return "low"

    def predict_batch(
        self,
        transactions: list[Dict[str, Any]],
        strategy: str = "auto",
    ) -> list[Dict[str, Any]]:
        """Batch prediction with per-transaction routing."""
        return [
            self.predict(tx, transaction_id=tx.get("id"), strategy=strategy)
            for tx in transactions
        ]
