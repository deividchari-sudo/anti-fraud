"""
Reusable mixins for ML fraud models (Sprint 2 audit fix).
Extracted from EnsembleFraudModel and StackingFraudModel to eliminate duplication.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# Channel feature columns (one-hot encoded in feature engineering)
CHANNEL_FEATURES = ["canal_app", "canal_web", "canal_api"]
# Product feature columns (one-hot encoded in feature engineering)
PRODUCT_FEATURES = [
    "produto_pix",
    "produto_ted",
    "produto_boleto",
    "produto_autenticacao",
    "produto_financeiro_generico",
]


class ThresholdTuningMixin:
    """Provides per-channel and per-product F1 threshold optimization.

    Requires that the host class defines:
    - self.threshold (float)
    - self.thresholds_by_channel (Dict[str, float])
    - self.thresholds_by_product (Dict[str, float])
    - a method _proba(X) returning probabilities for the positive class
    """

    @staticmethod
    def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
        """Find threshold maximizing F1-Score."""
        if len(np.unique(y_true)) < 2:
            return None
        precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = int(np.argmax(f1s))
        if best_idx >= len(thresholds):
            return None
        return float(thresholds[best_idx])

    def optimize_thresholds(
        self,
        proba: np.ndarray,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        min_segment_size: int = 50,
    ) -> None:
        """Optimize global, per-channel, and per-product F1 thresholds.

        Args:
            proba: Probability predictions for positive class (shape: n_samples,)
            X_test: Test features (must have channel/product one-hot columns)
            y_test: Test labels
            min_segment_size: Minimum segment size to compute a tuned threshold
        """
        # Global
        global_thr = self.best_f1_threshold(y_test.values, proba)
        if global_thr is not None:
            self.threshold = float(global_thr)

        # Per-channel
        for col in CHANNEL_FEATURES:
            if col in X_test.columns:
                mask = X_test[col] == 1
                if mask.sum() > min_segment_size and y_test[mask].sum() > 0:
                    thr = self.best_f1_threshold(y_test[mask].values, proba[mask])
                    if thr is not None:
                        self.thresholds_by_channel[col] = float(thr)

        # Per-product
        for col in PRODUCT_FEATURES:
            if col in X_test.columns:
                mask = X_test[col] == 1
                if mask.sum() > min_segment_size and y_test[mask].sum() > 0:
                    thr = self.best_f1_threshold(y_test[mask].values, proba[mask])
                    if thr is not None:
                        self.thresholds_by_product[col] = float(thr)

    def select_threshold(self, features_row: pd.Series) -> float:
        """Pick best threshold for a transaction (product > channel > global)."""
        for col, thr in self.thresholds_by_product.items():
            if col in features_row.index and features_row[col] == 1:
                return thr
        for col, thr in self.thresholds_by_channel.items():
            if col in features_row.index and features_row[col] == 1:
                return thr
        return self.threshold


class SHAPExplainerMixin:
    """Provides SHAP-based explanation generation.

    Requires that the host class defines:
    - self.explainer (SHAP TreeExplainer or None)
    - self.feature_names (list)
    - self.logger (logging.Logger)
    """

    @staticmethod
    def _summarize_explanation(top_features: Dict[str, float]) -> str:
        """Convert SHAP top features into human-readable summary."""
        if not top_features:
            return "No significant features."
        positives = [f for f, v in top_features.items() if v > 0][:3]
        negatives = [f for f, v in top_features.items() if v < 0][:3]
        parts = []
        if positives:
            parts.append(f"Indicadores de fraude: {', '.join(positives)}")
        if negatives:
            parts.append(f"Indicadores legítimos: {', '.join(negatives)}")
        return ". ".join(parts)

    def get_shap_explanation(
        self,
        X: pd.DataFrame,
        fraud_probability: float,
        model_name: str,
        top_n: int = 10,
        contribution_floor: float = 0.01,
    ) -> Optional[Dict[str, Any]]:
        """Build SHAP explanation dict for a single prediction.

        Args:
            X: Single-row DataFrame (1, n_features)
            fraud_probability: Probability score for the prediction
            model_name: Name of the model (for the explanation payload)
            top_n: Number of top features to include
            contribution_floor: Minimum |contribution| to be included

        Returns:
            Explanation dict or None on failure
        """
        if self.explainer is None:
            return None

        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            expected_value = self.explainer.expected_value
            base_value = (
                float(expected_value[0])
                if isinstance(expected_value, (list, np.ndarray))
                and len(np.atleast_1d(expected_value)) > 0
                else float(expected_value)
            )

            contributions: Dict[str, float] = {}
            for i, feature in enumerate(self.feature_names):
                if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
                    contribution = float(shap_values[0][i])
                else:
                    contribution = float(shap_values[i])
                if abs(contribution) > contribution_floor:
                    contributions[feature] = contribution

            sorted_contributions = dict(
                sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            top_features = dict(list(sorted_contributions.items())[:top_n])

            return {
                "base_value": base_value,
                "fraud_probability": fraud_probability,
                "top_contributing_features": top_features,
                "explanation_summary": self._summarize_explanation(top_features),
                "model": model_name,
            }
        except Exception as e:
            self.logger.error(f"SHAP explanation error: {e}")
            return None
