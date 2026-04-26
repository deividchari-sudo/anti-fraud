"""
Base class for fraud detection models.
Centralizes common logic shared by FraudDetectionModel and EnsembleFraudModel:
- Audit logging setup
- Train/test split + SMOTE balancing
- Feature alignment for prediction
- Decision logging
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

DEFAULT_THRESHOLD = 0.5
LOG_DIR = "logs"
AUDIT_LOG_FILE = "logs/audit.log"


class BaseFraudModel(ABC):
    """Abstract base class for fraud detection models.

    Subclasses must implement: _fit_estimators, _predict_proba_array, save_model, load_model.
    """

    def __init__(self, logger_name: str = "fraud_audit"):
        self.feature_names: list = []
        self.threshold: float = DEFAULT_THRESHOLD
        self.explainer = None  # SHAP (optional)
        self.logger = self._setup_audit_logger(logger_name)

    # ------------------------------------------------------------ logging

    @staticmethod
    def _setup_audit_logger(name: str) -> logging.Logger:
        """Set up audit logger writing to logs/audit.log (BACEN)."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        Path(LOG_DIR).mkdir(exist_ok=True)
        already_attached = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "").endswith("audit.log")
            for h in logger.handlers
        )
        if not already_attached:
            file_handler = logging.FileHandler(AUDIT_LOG_FILE)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(file_handler)
        return logger

    # --------------------------------------------------------- data prep

    @staticmethod
    def prepare_train_test(
        df: pd.DataFrame,
        target_col: str = "fraudResult",
        test_size: float = 0.2,
        random_state: int = 42,
        apply_smote: bool = True,
        smote_k_neighbors: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Prepare train/test split with optional SMOTE balancing.

        Returns:
            X_train_resampled, X_test_filled, y_train_resampled, y_test, X_train_orig, X_test_orig
        """
        X = df.drop(columns=[target_col, "payload"])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train_filled = X_train.fillna(0)
        X_test_filled = X_test.fillna(0)

        if apply_smote:
            smote = SMOTE(random_state=random_state, k_neighbors=smote_k_neighbors)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_filled, y_train
            )
        else:
            X_train_resampled, y_train_resampled = X_train_filled, y_train

        return (
            X_train_resampled,
            X_test_filled,
            y_train_resampled,
            y_test,
            X_train_filled,
            y_train,
        )

    @staticmethod
    def calculate_scale_pos_weight(y: pd.Series, min_weight: float = 50.0) -> float:
        """Calculate scale_pos_weight for imbalanced datasets."""
        n_pos = sum(y)
        if n_pos == 0:
            return min_weight
        weight = (len(y) - n_pos) / n_pos
        return float(max(weight, min_weight))

    # -------------------------------------------------- prediction helpers

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ensure prediction features match training features (fill missing with 0)."""
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0
        return features[self.feature_names].fillna(0)

    def _log_decision(
        self,
        transaction_id: Optional[str],
        fraud_probability: float,
        is_fraud: bool,
        threshold_used: Optional[float] = None,
        explanation: Optional[Dict[str, Any]] = None,
        model_name: str = "fraud_model",
    ) -> None:
        """Log prediction decision for BACEN audit trail."""
        log_entry = {
            "transaction_id": transaction_id or "unknown",
            "timestamp": datetime.now().isoformat(),
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
            "threshold": threshold_used if threshold_used is not None else self.threshold,
            "model": model_name,
            "explanation": explanation,
        }
        if is_fraud:
            self.logger.info(f"FRAUD DETECTED ({model_name}): {json.dumps(log_entry)}")
        else:
            self.logger.debug(f"LEGITIMATE ({model_name}): {json.dumps(log_entry)}")

    # ------------------------------------------------------ abstract API

    @abstractmethod
    def train(self, df: pd.DataFrame, target_col: str = "fraudResult") -> dict:
        """Train the model and return evaluation metrics."""

    @abstractmethod
    def predict(
        self, features: pd.DataFrame, transaction_id: Optional[str] = None
    ) -> Tuple[float, bool, Optional[Dict[str, Any]]]:
        """Predict fraud probability and decision."""

    @abstractmethod
    def save_model(self) -> None:
        """Persist model to storage."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model from storage."""

    @abstractmethod
    def get_feature_importance(self) -> dict:
        """Return feature importance scores."""
