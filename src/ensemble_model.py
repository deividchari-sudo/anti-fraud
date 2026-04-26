"""
Ensemble Fraud Detection Model (refactored).
Uses BaseFraudModel for shared logic and JoblibEnsembleModelRepository for persistence.

Sprint 1 implementation (PM/Data Specialist roadmap):
- LightGBM ensemble with XGBoost
- Isotonic probability calibration (CalibratedClassifierCV)
- Per-channel and per-product threshold tuning
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.base_model import DEFAULT_THRESHOLD, BaseFraudModel
from src.ml_mixins import SHAPExplainerMixin, ThresholdTuningMixin
from src.repositories import (
    EnsembleModelRepository,
    JoblibEnsembleModelRepository,
)


class EnsembleFraudModel(BaseFraudModel, ThresholdTuningMixin, SHAPExplainerMixin):
    """Ensemble fraud detection model: XGBoost + LightGBM with calibration and per-channel thresholds."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        xgb_weight: float = 0.5,
        lgb_weight: float = 0.5,
        repository: Optional[EnsembleModelRepository] = None,
    ):
        """
        Initialize ensemble fraud model.

        Args:
            model_path: Path to save/load ensemble bundle (used if repository not given)
            xgb_weight: Weight for XGBoost predictions (0..1)
            lgb_weight: Weight for LightGBM predictions (0..1)
            repository: Optional injected EnsembleModelRepository (DI for testing/storage)
        """
        super().__init__(logger_name="fraud_audit_ensemble")

        if abs(xgb_weight + lgb_weight - 1.0) > 1e-6:
            raise ValueError("xgb_weight + lgb_weight must equal 1.0")

        self.xgb_model = None  # CalibratedClassifierCV wrapping XGBoost
        self.lgb_model = None  # CalibratedClassifierCV wrapping LightGBM
        self.threshold = DEFAULT_THRESHOLD
        self.thresholds_by_channel: Dict[str, float] = {}
        self.thresholds_by_product: Dict[str, float] = {}
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.model_path = model_path or "models/ensemble_fraud_model.pkl"

        # Dependency Injection for persistence
        self.repository = repository or JoblibEnsembleModelRepository(self.model_path)

        if self.repository.exists():
            self.load_model()

    # ---------------------------------------------------------------- training

    def train(self, df: pd.DataFrame, target_col: str = "fraudResult") -> dict:
        """Train ensemble model.

        Args:
            df: Training dataframe (with target and 'payload' columns)
            target_col: Name of target column

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Training ensemble (XGBoost + LightGBM) with {len(df)} samples...")

        # Use shared data prep from BaseFraudModel
        (
            X_train_resampled,
            X_test_filled,
            y_train_resampled,
            y_test,
            X_train_orig,
            _,
        ) = self.prepare_train_test(df, target_col=target_col, apply_smote=True)

        print(f"Training set: {X_train_orig.shape}, Test set: {X_test_filled.shape}")
        print(
            f"After SMOTE: {X_train_resampled.shape}, "
            f"fraud ratio: {y_train_resampled.mean():.4f}"
        )

        scale_pos_weight = self.calculate_scale_pos_weight(y_train_resampled)

        # ---- Train XGBoost (base estimator) ----
        print("\nTraining XGBoost base estimator...")
        xgb_base = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

        # Wrap in calibrator (isotonic - more robust for tree models)
        self.xgb_model = CalibratedClassifierCV(xgb_base, method="isotonic", cv=3)
        self.xgb_model.fit(X_train_resampled, y_train_resampled)

        # ---- Train LightGBM (base estimator) ----
        print("Training LightGBM base estimator...")
        lgb_base = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            scale_pos_weight=scale_pos_weight,
            max_depth=5,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        self.lgb_model = CalibratedClassifierCV(lgb_base, method="isotonic", cv=3)
        self.lgb_model.fit(X_train_resampled, y_train_resampled)

        self.feature_names = list(X_train_orig.columns)

        # ---- Evaluate ensemble at default threshold ----
        print("\n--- Metrics @ default threshold (0.5) ---")
        metrics_default = self.evaluate(X_test_filled, y_test)

        # ---- Optimize thresholds (global + per-channel + per-product) ----
        self._optimize_thresholds(X_test_filled, y_test)

        # ---- Re-evaluate at optimized threshold ----
        print("\n--- Metrics @ optimized global threshold ---")
        metrics = self.evaluate(X_test_filled, y_test)
        metrics["metrics_at_default_threshold"] = metrics_default

        # ---- Initialize SHAP for XGBoost (explainability) ----
        try:
            xgb_estimator = self._get_xgb_estimator()
            if xgb_estimator is not None:
                self.explainer = shap.TreeExplainer(xgb_estimator)
                print("SHAP explainer initialized (XGBoost calibrated)")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP: {e}")
            self.explainer = None

        self.save_model()
        return metrics

    def _get_xgb_estimator(self):
        """Extract underlying XGBoost from CalibratedClassifierCV (first fold)."""
        if self.xgb_model is None:
            return None
        try:
            return self.xgb_model.calibrated_classifiers_[0].estimator
        except (AttributeError, IndexError):
            return None

    # -------------------------------------------------------------- evaluation

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate ensemble performance."""
        y_pred_proba = self._ensemble_proba(X_test)
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # Per-base-model metrics for comparison
        xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X_test)[:, 1]
        metrics["xgb_auc_roc"] = roc_auc_score(y_test, xgb_proba)
        metrics["lgb_auc_roc"] = roc_auc_score(y_test, lgb_proba)

        print("\n=== Ensemble Evaluation ===")
        print(f"AUC-ROC (Ensemble): {metrics['auc_roc']:.4f}")
        print(f"AUC-ROC (XGBoost only): {metrics['xgb_auc_roc']:.4f}")
        print(f"AUC-ROC (LightGBM only): {metrics['lgb_auc_roc']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"])
        )
        return metrics

    def _ensemble_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute weighted average probability from XGBoost + LightGBM."""
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        return self.xgb_weight * xgb_proba + self.lgb_weight * lgb_proba

    # ------------------------------------------------------- threshold tuning

    def _optimize_thresholds(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Find optimal F1 threshold globally and per channel/product (delegates to mixin)."""
        proba = self._ensemble_proba(X_test)
        self.optimize_thresholds(proba, X_test, y_test)
        print(f"\n=== Threshold Optimization ===")
        print(f"Global optimal threshold (F1): {self.threshold:.4f}")
        print(f"Thresholds by channel: {self.thresholds_by_channel}")
        print(f"Thresholds by product: {self.thresholds_by_product}")

    # ------------------------------------------------------------- prediction

    def predict(
        self, features: pd.DataFrame, transaction_id: Optional[str] = None
    ) -> Tuple[float, bool, Optional[Dict[str, Any]]]:
        """Predict fraud probability with ensemble + per-channel threshold."""
        if self.xgb_model is None or self.lgb_model is None:
            raise ValueError("Ensemble model not loaded. Train or load first.")

        # Use shared feature alignment from BaseFraudModel
        X = self._align_features(features)

        fraud_probability = float(self._ensemble_proba(X)[0])
        threshold_used = self.select_threshold(X.iloc[0])
        is_fraud = fraud_probability >= threshold_used

        explanation = None
        if is_fraud and self.explainer is not None:
            explanation = self.get_shap_explanation(
                X, fraud_probability, model_name="ensemble_xgb_lgb_calibrated"
            )

        # Use shared decision logging from BaseFraudModel
        self._log_decision(
            transaction_id,
            fraud_probability,
            is_fraud,
            threshold_used=threshold_used,
            explanation=explanation,
            model_name="ensemble_xgb_lgb",
        )
        return fraud_probability, is_fraud, explanation

    # -------------------------------------------------------- persistence I/O

    def save_model(self):
        """Save full ensemble bundle via injected repository."""
        bundle = {
            "xgb_model": self.xgb_model,
            "lgb_model": self.lgb_model,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "thresholds_by_channel": self.thresholds_by_channel,
            "thresholds_by_product": self.thresholds_by_product,
            "xgb_weight": self.xgb_weight,
            "lgb_weight": self.lgb_weight,
        }
        self.repository.save_bundle(bundle)
        print(f"Ensemble model saved to {self.model_path}")

    def load_model(self):
        """Load ensemble bundle via injected repository."""
        bundle = self.repository.load_bundle()
        self.xgb_model = bundle["xgb_model"]
        self.lgb_model = bundle["lgb_model"]
        self.feature_names = bundle["feature_names"]
        self.threshold = bundle.get("threshold", DEFAULT_THRESHOLD)
        self.thresholds_by_channel = bundle.get("thresholds_by_channel", {})
        self.thresholds_by_product = bundle.get("thresholds_by_product", {})
        self.xgb_weight = bundle.get("xgb_weight", 0.5)
        self.lgb_weight = bundle.get("lgb_weight", 0.5)

        try:
            xgb_estimator = self._get_xgb_estimator()
            if xgb_estimator is not None:
                self.explainer = shap.TreeExplainer(xgb_estimator)
                print("SHAP explainer initialized")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP: {e}")
            self.explainer = None
        print(f"Ensemble model loaded from {self.model_path}")

    def get_feature_importance(self) -> dict:
        """Average feature importance from both base estimators."""
        if self.xgb_model is None or self.lgb_model is None:
            raise ValueError("Model not loaded.")

        try:
            xgb_est = self.xgb_model.calibrated_classifiers_[0].estimator
            lgb_est = self.lgb_model.calibrated_classifiers_[0].estimator
            xgb_imp = np.array(xgb_est.feature_importances_, dtype=float)
            lgb_imp = np.array(lgb_est.feature_importances_, dtype=float)

            xgb_imp = xgb_imp / (xgb_imp.sum() + 1e-9)
            lgb_imp = lgb_imp / (lgb_imp.sum() + 1e-9)
            avg = self.xgb_weight * xgb_imp + self.lgb_weight * lgb_imp

            importance = dict(zip(self.feature_names, avg))
            return dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )
        except (AttributeError, IndexError) as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return {}
