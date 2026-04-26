"""
Stacking Fraud Detection Model (Sprint 2).

Architecture:
    Level 0 (base estimators): XGBoost + LightGBM + CatBoost
    Level 1 (meta-learner): Logistic Regression
    Calibration: Isotonic on the meta-learner output
    Threshold tuning: Global + per-channel + per-product

Trains base estimators on K-fold splits to generate out-of-fold predictions
that feed the meta-learner (avoids data leakage).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.base_model import DEFAULT_THRESHOLD, BaseFraudModel
from src.ml_mixins import SHAPExplainerMixin, ThresholdTuningMixin
from src.repositories import (
    EnsembleModelRepository,
    JoblibEnsembleModelRepository,
)


class StackingFraudModel(BaseFraudModel, ThresholdTuningMixin, SHAPExplainerMixin):
    """Stacking ensemble: XGBoost + LightGBM + CatBoost -> Logistic Regression meta-learner.

    Uses sklearn StackingClassifier with K-fold cross-validation to train base estimators
    and generate meta-features. Final calibration applied via CalibratedClassifierCV.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_folds: int = 3,
        repository: Optional[EnsembleModelRepository] = None,
    ):
        """
        Initialize stacking fraud model.

        Args:
            model_path: Path to save/load model bundle
            n_folds: Number of CV folds for stacking (3-5 typical)
            repository: Optional injected EnsembleModelRepository (DI)
        """
        super().__init__(logger_name="fraud_audit_stacking")

        self.stacking_model = None  # CalibratedClassifierCV(StackingClassifier)
        self.threshold = DEFAULT_THRESHOLD
        self.thresholds_by_channel: Dict[str, float] = {}
        self.thresholds_by_product: Dict[str, float] = {}
        self.n_folds = n_folds
        self.model_path = model_path or "models/stacking_fraud_model.pkl"

        # SHAP explainer attached to first base estimator (XGBoost) for explainability
        self.explainer = None

        # Dependency Injection for persistence
        self.repository = repository or JoblibEnsembleModelRepository(self.model_path)

        if self.repository.exists():
            self.load_model()

    # ---------------------------------------------------------------- training

    def train(self, df: pd.DataFrame, target_col: str = "fraudResult") -> dict:
        """Train stacking model on the dataframe.

        Args:
            df: Training dataframe (with target and 'payload' columns)
            target_col: Target column name

        Returns:
            Evaluation metrics
        """
        print(
            f"Training stacking model (XGBoost + LightGBM + CatBoost -> LR) "
            f"with {len(df)} samples..."
        )

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

        # ---- Build base estimators ----
        xgb_base = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

        lgb_base = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            scale_pos_weight=scale_pos_weight,
            max_depth=5,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        cat_base = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3.0,
            scale_pos_weight=scale_pos_weight,
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
        )

        # ---- Meta-learner ----
        meta = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        # ---- Stacking ----
        print("\nFitting StackingClassifier with 3 base estimators + LR meta...")
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        stacking = StackingClassifier(
            estimators=[
                ("xgb", xgb_base),
                ("lgb", lgb_base),
                ("cat", cat_base),
            ],
            final_estimator=meta,
            cv=cv,
            stack_method="predict_proba",
            n_jobs=1,  # avoid pickling issues with CatBoost on Windows
            passthrough=False,
        )

        # Calibrate the full stacking pipeline
        print("Wrapping with isotonic calibration...")
        self.stacking_model = CalibratedClassifierCV(
            stacking, method="isotonic", cv=3
        )
        self.stacking_model.fit(X_train_resampled, y_train_resampled)

        self.feature_names = list(X_train_orig.columns)

        # ---- Evaluate at default threshold ----
        print("\n--- Metrics @ default threshold (0.5) ---")
        metrics_default = self.evaluate(X_test_filled, y_test)

        # ---- Optimize thresholds ----
        self._optimize_thresholds(X_test_filled, y_test)

        # ---- Re-evaluate at optimized threshold ----
        print("\n--- Metrics @ optimized global threshold ---")
        metrics = self.evaluate(X_test_filled, y_test)
        metrics["metrics_at_default_threshold"] = metrics_default

        # ---- SHAP via underlying XGBoost (first base estimator) ----
        try:
            xgb_estimator = self._get_xgb_estimator()
            if xgb_estimator is not None:
                self.explainer = shap.TreeExplainer(xgb_estimator)
                print("SHAP explainer initialized (XGBoost from stacking)")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP: {e}")
            self.explainer = None

        self.save_model()
        return metrics

    def _get_xgb_estimator(self):
        """Extract XGBoost from the calibrated stacking pipeline."""
        if self.stacking_model is None:
            return None
        try:
            # CalibratedClassifierCV -> calibrated_classifiers_[0].estimator -> StackingClassifier
            stacking = self.stacking_model.calibrated_classifiers_[0].estimator
            # StackingClassifier.named_estimators_ has the FITTED base estimators
            return stacking.named_estimators_["xgb"]
        except (AttributeError, IndexError, KeyError):
            return None

    # -------------------------------------------------------------- evaluation

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate stacking model performance."""
        y_pred_proba = self.stacking_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        print("\n=== Stacking Evaluation ===")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"])
        )
        return metrics

    # ------------------------------------------------------- threshold tuning

    def _optimize_thresholds(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Find optimal F1 threshold globally and per channel/product (delegates to mixin)."""
        proba = self.stacking_model.predict_proba(X_test)[:, 1]
        self.optimize_thresholds(proba, X_test, y_test)
        print("\n=== Threshold Optimization ===")
        print(f"Global optimal threshold (F1): {self.threshold:.4f}")
        print(f"Thresholds by channel: {self.thresholds_by_channel}")
        print(f"Thresholds by product: {self.thresholds_by_product}")

    # ------------------------------------------------------------- prediction

    def predict(
        self, features: pd.DataFrame, transaction_id: Optional[str] = None
    ) -> Tuple[float, bool, Optional[Dict[str, Any]]]:
        """Predict fraud probability with stacking + per-segment threshold."""
        if self.stacking_model is None:
            raise ValueError("Stacking model not loaded. Train or load first.")

        X = self._align_features(features)

        fraud_probability = float(self.stacking_model.predict_proba(X)[0, 1])
        threshold_used = self.select_threshold(X.iloc[0])
        is_fraud = fraud_probability >= threshold_used

        explanation = None
        if is_fraud and self.explainer is not None:
            explanation = self.get_shap_explanation(
                X, fraud_probability, model_name="stacking_xgb_lgb_cat_lr_calibrated"
            )

        self._log_decision(
            transaction_id,
            fraud_probability,
            is_fraud,
            threshold_used=threshold_used,
            explanation=explanation,
            model_name="stacking_xgb_lgb_cat_lr",
        )
        return fraud_probability, is_fraud, explanation

    # -------------------------------------------------------- persistence I/O

    def save_model(self):
        """Save full stacking bundle via injected repository."""
        bundle = {
            "stacking_model": self.stacking_model,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "thresholds_by_channel": self.thresholds_by_channel,
            "thresholds_by_product": self.thresholds_by_product,
            "n_folds": self.n_folds,
            "model_type": "stacking_xgb_lgb_cat_lr",
        }
        self.repository.save_bundle(bundle)
        print(f"Stacking model saved to {self.model_path}")

    def load_model(self):
        """Load stacking bundle via injected repository."""
        bundle = self.repository.load_bundle()
        self.stacking_model = bundle["stacking_model"]
        self.feature_names = bundle["feature_names"]
        self.threshold = bundle.get("threshold", DEFAULT_THRESHOLD)
        self.thresholds_by_channel = bundle.get("thresholds_by_channel", {})
        self.thresholds_by_product = bundle.get("thresholds_by_product", {})
        self.n_folds = bundle.get("n_folds", 3)

        try:
            xgb_estimator = self._get_xgb_estimator()
            if xgb_estimator is not None:
                self.explainer = shap.TreeExplainer(xgb_estimator)
                print("SHAP explainer initialized")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP: {e}")
            self.explainer = None
        print(f"Stacking model loaded from {self.model_path}")

    def get_feature_importance(self) -> dict:
        """Average feature importance across the 3 base estimators."""
        if self.stacking_model is None:
            raise ValueError("Model not loaded.")

        try:
            stacking = self.stacking_model.calibrated_classifiers_[0].estimator
            xgb_imp = np.array(
                stacking.named_estimators_["xgb"].feature_importances_, dtype=float
            )
            lgb_imp = np.array(
                stacking.named_estimators_["lgb"].feature_importances_, dtype=float
            )
            cat_imp = np.array(
                stacking.named_estimators_["cat"].feature_importances_, dtype=float
            )

            xgb_imp = xgb_imp / (xgb_imp.sum() + 1e-9)
            lgb_imp = lgb_imp / (lgb_imp.sum() + 1e-9)
            cat_imp = cat_imp / (cat_imp.sum() + 1e-9)
            avg = (xgb_imp + lgb_imp + cat_imp) / 3.0

            importance = dict(zip(self.feature_names, avg))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except (AttributeError, IndexError, KeyError) as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return {}
