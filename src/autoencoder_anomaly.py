"""
AutoEncoder for Zero-Day Fraud Detection (Sprint 3).

Uses MLPRegressor (sklearn) as a proxy for AutoEncoder:
- Trains to reconstruct only LEGITIMATE transactions
- High reconstruction error on a new transaction = potential zero-day fraud
- Complements supervised models by detecting NEW fraud patterns not seen in training

This approach is used by Itaú and Bradesco in production for emerging fraud schemas.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.repositories import (
    AnomalyModelRepository,
    JoblibAnomalyModelRepository,
)


class AutoEncoderAnomalyDetector:
    """Reconstruction-based anomaly detector for zero-day fraud."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (32, 16, 8, 16, 32),
        max_iter: int = 100,
        random_state: int = 42,
        anomaly_percentile: float = 95.0,
        model_path: Optional[str] = None,
        repository: Optional[AnomalyModelRepository] = None,
    ):
        """
        Initialize AutoEncoder anomaly detector.

        Args:
            hidden_dims: Encoder-bottleneck-decoder layer dimensions
            max_iter: Max training iterations
            random_state: Random seed
            anomaly_percentile: Percentile of reconstruction error used as anomaly threshold
            model_path: Path to save/load model bundle (used if repository not given)
            repository: Optional injected AnomalyModelRepository (DI)
        """
        self.hidden_dims = hidden_dims
        self.max_iter = max_iter
        self.random_state = random_state
        self.anomaly_percentile = anomaly_percentile
        self.model_path = model_path or "models/autoencoder_anomaly.pkl"

        self.scaler = StandardScaler()
        self.autoencoder: Optional[MLPRegressor] = None
        self.threshold: float = 0.0
        self.is_fitted = False

        # Dependency Injection for persistence
        self.repository = repository or JoblibAnomalyModelRepository(self.model_path)

        # Audit logger (BACEN compliance)
        self.logger = logging.getLogger("fraud_audit_autoencoder")
        self.logger.setLevel(logging.INFO)
        Path("logs").mkdir(exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "").endswith("audit.log")
            for h in self.logger.handlers
        ):
            handler = logging.FileHandler("logs/audit.log")
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(handler)

    def fit(self, X_legitimate: pd.DataFrame) -> dict:
        """Fit the autoencoder on LEGITIMATE transactions only.

        Args:
            X_legitimate: Feature DataFrame containing only legitimate transactions

        Returns:
            Training metrics dict
        """
        if len(X_legitimate) < 10:
            raise ValueError("Need at least 10 legitimate transactions to train")

        X_filled = X_legitimate.fillna(0)
        X_scaled = self.scaler.fit_transform(X_filled)

        # AutoEncoder = MLP with output equal to input (reconstruction)
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=self.hidden_dims,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=False,  # don't waste data on validation
            verbose=False,
        )
        self.autoencoder.fit(X_scaled, X_scaled)

        # Compute reconstruction errors on training set
        recon = self.autoencoder.predict(X_scaled)
        errors = np.mean((X_scaled - recon) ** 2, axis=1)

        # Set anomaly threshold at chosen percentile of legit errors
        self.threshold = float(np.percentile(errors, self.anomaly_percentile))
        self.is_fitted = True

        return {
            "n_train": len(X_legitimate),
            "mean_error": float(errors.mean()),
            "std_error": float(errors.std()),
            "anomaly_threshold": self.threshold,
            "anomaly_percentile": self.anomaly_percentile,
        }

    def predict(self, X: pd.DataFrame) -> Dict[str, float]:
        """Predict anomaly score for a single transaction or batch.

        Args:
            X: Feature DataFrame (single row or batch)

        Returns:
            Dict with anomaly_score, is_zero_day_anomaly, threshold
        """
        if not self.is_fitted or self.autoencoder is None:
            return {
                "anomaly_score": 0.0,
                "is_zero_day_anomaly": False,
                "threshold": self.threshold,
                "model_fitted": False,
            }

        X_filled = X.fillna(0)
        X_scaled = self.scaler.transform(X_filled)
        recon = self.autoencoder.predict(X_scaled)

        errors = np.mean((X_scaled - recon) ** 2, axis=1)
        score = float(errors[0]) if len(errors) == 1 else float(errors.mean())
        is_anomaly = score > self.threshold

        # Normalize score to [0, 1] using ratio against threshold
        normalized = min(1.0, score / (self.threshold * 2 + 1e-9))

        result = {
            "anomaly_score": float(normalized),
            "raw_reconstruction_error": float(score),
            "is_zero_day_anomaly": bool(is_anomaly),
            "threshold": self.threshold,
            "model_fitted": True,
        }

        # Audit log if zero-day fraud suspected (BACEN)
        if is_anomaly:
            self.logger.info(
                f"ZERO-DAY ANOMALY DETECTED: score={normalized:.4f}, "
                f"raw_error={score:.6f}, threshold={self.threshold:.6f}"
            )

        return result

    def save(self) -> None:
        """Persist autoencoder bundle via injected repository."""
        bundle = {
            "autoencoder": self.autoencoder,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "anomaly_percentile": self.anomaly_percentile,
            "hidden_dims": self.hidden_dims,
            "is_fitted": self.is_fitted,
        }
        self.repository.save_bundle(bundle)

    def load(self) -> None:
        """Load autoencoder bundle via injected repository."""
        if not self.repository.exists():
            raise FileNotFoundError(self.model_path)
        bundle = self.repository.load_bundle()
        self.autoencoder = bundle["autoencoder"]
        self.scaler = bundle["scaler"]
        self.threshold = bundle["threshold"]
        self.anomaly_percentile = bundle.get("anomaly_percentile", 95.0)
        self.hidden_dims = bundle.get("hidden_dims", (32, 16, 8, 16, 32))
        self.is_fitted = bundle.get("is_fitted", False)
