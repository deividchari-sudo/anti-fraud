"""
Ensemble Fraud Detection Model.
Combines XGBoost + LightGBM with probability calibration and channel/product-specific thresholds.

Sprint 1 implementation (PM/Data Specialist roadmap):
- LightGBM ensemble with XGBoost
- Isotonic probability calibration (CalibratedClassifierCV)
- Per-channel and per-product threshold tuning
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

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

DEFAULT_THRESHOLD = 0.5


class EnsembleFraudModel:
    """Ensemble fraud detection model: XGBoost + LightGBM with calibration and per-channel thresholds."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        xgb_weight: float = 0.5,
        lgb_weight: float = 0.5,
    ):
        """
        Initialize ensemble fraud model.

        Args:
            model_path: Path to save/load ensemble bundle
            xgb_weight: Weight for XGBoost predictions (0..1)
            lgb_weight: Weight for LightGBM predictions (0..1)
        """
        if abs(xgb_weight + lgb_weight - 1.0) > 1e-6:
            raise ValueError("xgb_weight + lgb_weight must equal 1.0")

        self.xgb_model = None  # CalibratedClassifierCV wrapping XGBoost
        self.lgb_model = None  # CalibratedClassifierCV wrapping LightGBM
        self.feature_names: list = []
        self.threshold = DEFAULT_THRESHOLD
        self.thresholds_by_channel: Dict[str, float] = {}
        self.thresholds_by_product: Dict[str, float] = {}
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.explainer = None  # SHAP explainer for XGBoost (used for explainability)
        self.model_path = model_path or "models/ensemble_fraud_model.pkl"

        # Audit logger
        self.logger = logging.getLogger("fraud_audit_ensemble")
        self.logger.setLevel(logging.INFO)
        Path("logs").mkdir(exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "").endswith("audit.log")
            for h in self.logger.handlers
        ):
            file_handler = logging.FileHandler("logs/audit.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)

        if Path(self.model_path).exists():
            self.load_model()

    # ---------------------------------------------------------------- training

    def train(self, df: pd.DataFrame, target_col: str = "fraudResult") -> dict:
        """
        Train ensemble model.

        Args:
            df: Training dataframe (with target and 'payload' columns)
            target_col: Name of target column

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Training ensemble (XGBoost + LightGBM) with {len(df)} samples...")

        X = df.drop(columns=[target_col, "payload"])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        X_train_filled = X_train.fillna(0)
        X_test_filled = X_test.fillna(0)

        # SMOTE for class balancing
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_filled, y_train
        )
        print(
            f"After SMOTE: {X_train_resampled.shape}, "
            f"fraud ratio: {y_train_resampled.mean():.4f}"
        )

        scale_pos_weight = max(
            (len(y_train_resampled) - sum(y_train_resampled)) / sum(y_train_resampled),
            50,
        )

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

        self.feature_names = list(X_train.columns)

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
        """Find optimal F1 threshold globally and per channel/product."""
        proba = self._ensemble_proba(X_test)

        # Global optimal threshold
        global_thr = self._best_f1_threshold(y_test.values, proba)
        print(f"\n=== Threshold Optimization ===")
        print(f"Global optimal threshold (F1): {global_thr:.4f}")

        # Use a balanced default; keep optimal for reference
        self.threshold = float(global_thr) if global_thr is not None else DEFAULT_THRESHOLD

        # Per-channel thresholds
        for col in CHANNEL_FEATURES:
            if col in X_test.columns:
                mask = X_test[col] == 1
                if mask.sum() > 50 and y_test[mask].sum() > 0:
                    thr = self._best_f1_threshold(y_test[mask].values, proba[mask])
                    if thr is not None:
                        self.thresholds_by_channel[col] = float(thr)

        # Per-product thresholds
        for col in PRODUCT_FEATURES:
            if col in X_test.columns:
                mask = X_test[col] == 1
                if mask.sum() > 50 and y_test[mask].sum() > 0:
                    thr = self._best_f1_threshold(y_test[mask].values, proba[mask])
                    if thr is not None:
                        self.thresholds_by_product[col] = float(thr)

        print(f"Thresholds by channel: {self.thresholds_by_channel}")
        print(f"Thresholds by product: {self.thresholds_by_product}")

    @staticmethod
    def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
        """Find threshold maximizing F1-Score."""
        if len(np.unique(y_true)) < 2:
            return None
        precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = int(np.argmax(f1s))
        if best_idx >= len(thresholds):
            return None
        return float(thresholds[best_idx])

    def _select_threshold(self, features_row: pd.Series) -> float:
        """Pick best threshold for a transaction based on its channel/product."""
        # Product-specific threshold takes priority (more granular)
        for col, thr in self.thresholds_by_product.items():
            if col in features_row.index and features_row[col] == 1:
                return thr
        for col, thr in self.thresholds_by_channel.items():
            if col in features_row.index and features_row[col] == 1:
                return thr
        return self.threshold

    # ------------------------------------------------------------- prediction

    def predict(
        self, features: pd.DataFrame, transaction_id: Optional[str] = None
    ) -> Tuple[float, bool, Optional[Dict[str, Any]]]:
        """Predict fraud probability with ensemble + per-channel threshold."""
        if self.xgb_model is None or self.lgb_model is None:
            raise ValueError("Ensemble model not loaded. Train or load first.")

        # Ensure all expected features exist
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0

        X = features[self.feature_names].fillna(0)

        fraud_probability = float(self._ensemble_proba(X)[0])

        # Pick threshold for this transaction
        threshold_used = self._select_threshold(X.iloc[0])
        is_fraud = fraud_probability >= threshold_used

        explanation = None
        if is_fraud and self.explainer is not None:
            explanation = self._get_explanation(X, fraud_probability)

        self._log_decision(
            transaction_id, fraud_probability, is_fraud, threshold_used, explanation
        )
        return fraud_probability, is_fraud, explanation

    def _get_explanation(
        self, X: pd.DataFrame, fraud_probability: float
    ) -> Optional[Dict[str, Any]]:
        """SHAP explanation using underlying XGBoost (calibrated wrapper)."""
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

            contributions = {}
            for i, feature in enumerate(self.feature_names):
                if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
                    contribution = float(shap_values[0][i])
                else:
                    contribution = float(shap_values[i])
                if abs(contribution) > 0.01:
                    contributions[feature] = contribution

            sorted_contributions = dict(
                sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            top_features = dict(list(sorted_contributions.items())[:10])

            return {
                "base_value": base_value,
                "fraud_probability": fraud_probability,
                "top_contributing_features": top_features,
                "explanation_summary": self._summarize(top_features),
                "model": "ensemble_xgb_lgb_calibrated",
            }
        except Exception as e:
            self.logger.error(f"SHAP explanation error: {e}")
            return None

    @staticmethod
    def _summarize(top_features: Dict[str, float]) -> str:
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

    def _log_decision(
        self,
        transaction_id: Optional[str],
        proba: float,
        is_fraud: bool,
        threshold_used: float,
        explanation: Optional[Dict[str, Any]],
    ):
        log_entry = {
            "transaction_id": transaction_id or "unknown",
            "timestamp": datetime.now().isoformat(),
            "fraud_probability": proba,
            "is_fraud": is_fraud,
            "threshold_used": threshold_used,
            "model": "ensemble_xgb_lgb",
            "explanation": explanation,
        }
        if is_fraud:
            self.logger.info(f"FRAUD DETECTED (ENSEMBLE): {json.dumps(log_entry)}")
        else:
            self.logger.debug(f"LEGITIMATE (ENSEMBLE): {json.dumps(log_entry)}")

    # -------------------------------------------------------- persistence I/O

    def save_model(self):
        """Save full ensemble bundle to disk."""
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
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
        joblib.dump(bundle, self.model_path)
        print(f"Ensemble model saved to {self.model_path}")

    def load_model(self):
        """Load ensemble bundle from disk."""
        bundle = joblib.load(self.model_path)
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
