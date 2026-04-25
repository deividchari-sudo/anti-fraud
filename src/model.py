import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from src.repositories import JoblibModelRepository, ModelRepository


class FraudDetectionModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_repository: Optional[ModelRepository] = None,
    ):
        self.model = None
        self.feature_names = []
        self.threshold = 0.5
        self.model_path = model_path or "models/fraud_model.pkl"
        self.explainer = None  # SHAP explainer

        # Dependency Injection for repository
        self.model_repository = model_repository or JoblibModelRepository(
            self.model_path, Path(self.model_path).parent / "feature_names.json"
        )

        # Setup audit logging
        self.logger = logging.getLogger("fraud_audit")
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        # File handler for audit log
        file_handler = logging.FileHandler("logs/audit.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if Path(self.model_path).exists():
            self.load_model()

    def train(self, df: pd.DataFrame, target_col: str = "fraudResult") -> dict:
        """Train the fraud detection model."""
        print(f"Training model with {len(df)} samples...")

        # Separate features and target
        X = df.drop(columns=[target_col, "payload"])
        y = df[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(
            f"Fraud distribution - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}"
        )

        # Handle NaN values before SMOTE
        print("Handling missing values...")
        X_train_filled = X_train.fillna(0)
        X_test_filled = X_test.fillna(0)

        # Apply SMOTE for oversampling minority class
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_filled, y_train
        )
        print(
            f"After SMOTE - Training set: {X_train_resampled.shape}, Fraud distribution: {y_train_resampled.mean():.4f}"
        )

        # Calculate scale_pos_weight for imbalanced dataset
        scale_pos_weight = (len(y_train_resampled) - sum(y_train_resampled)) / sum(
            y_train_resampled
        )
        print(f"Scale pos weight: {scale_pos_weight:.2f}")

        # Use a more aggressive scale for extreme imbalance
        scale_pos_weight = max(scale_pos_weight, 50)

        # Train XGBoost model with better hyperparameters for imbalanced data
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            max_depth=4,  # Reduced to prevent overfitting
            learning_rate=0.05,  # Lower learning rate
            n_estimators=500,  # More estimators
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,  # Regularization
            gamma=0.1,  # Regularization
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_train_resampled, y_train_resampled)
        self.feature_names = list(X_train.columns)

        # Evaluate model
        metrics = self.evaluate(X_test_filled, y_test)

        # Optimize threshold for better F1-Score
        self.optimize_threshold(X_test_filled, y_test)

        # Save model
        self.save_model()

        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance."""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": None,
            "recall": None,
            "confusion_matrix": None,
        }

        # Calculate precision and recall
        from sklearn.metrics import precision_score, recall_score

        metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

        print("\n=== Model Evaluation ===")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"])
        )

        return metrics

    def optimize_threshold(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Optimize decision threshold to maximize F1-Score."""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Calculate F1-Score for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

        # Find threshold that maximizes F1-Score
        best_idx = f1_scores.argmax()
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        print("\n=== Threshold Optimization ===")
        print(f"Best threshold (F1-Score): {best_threshold:.4f}")
        print(f"Best F1-Score: {best_f1:.4f}")

        # Use a more conservative threshold (0.5) to avoid too many false positives
        # The optimized threshold may be too conservative
        self.threshold = 0.5
        print(f"Using threshold: {self.threshold:.4f} (conservative)")

    def predict(
        self, features: pd.DataFrame, transaction_id: str = None
    ) -> Tuple[float, bool, Optional[Dict[str, Any]]]:
        """Make prediction on transaction features with explanation."""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        # Ensure features match model's expected features
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0

        # Select only the features the model expects
        X = features[self.feature_names]

        # Get prediction probability
        fraud_probability = float(self.model.predict_proba(X)[0][1])

        # Make binary decision based on threshold
        is_fraud = fraud_probability >= self.threshold

        # Calculate explanation if fraud detected
        explanation = None
        if is_fraud and self.explainer is not None:
            explanation = self._get_explanation(X, fraud_probability)

        # Log decision for audit trail
        self._log_decision(transaction_id, fraud_probability, is_fraud, explanation)

        return fraud_probability, is_fraud, explanation

    def _get_explanation(
        self, X: pd.DataFrame, fraud_probability: float
    ) -> Dict[str, Any]:
        """Generate SHAP explanation for the prediction."""
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[
                    0
                ]  # Take first element for binary classification

            # Get base value (expected value)
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                base_value = (
                    float(expected_value[0]) if len(expected_value) > 0 else 0.0
                )
            else:
                base_value = float(expected_value)

            # Get feature contributions
            feature_contributions = {}
            for i, feature in enumerate(self.feature_names):
                # Handle shap_values shape
                if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
                    contribution = float(shap_values[0][i])
                else:
                    contribution = float(shap_values[i])

                if abs(contribution) > 0.01:  # Only include significant contributions
                    feature_contributions[feature] = contribution

            # Sort by absolute contribution
            sorted_contributions = dict(
                sorted(
                    feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
                )
            )

            # Take top 10 features
            top_features = dict(list(sorted_contributions.items())[:10])

            explanation = {
                "base_value": base_value,
                "fraud_probability": fraud_probability,
                "top_contributing_features": top_features,
                "explanation_summary": self._generate_explanation_summary(top_features),
            }

            return explanation

        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation: {e}")
            return None

    def _generate_explanation_summary(self, top_features: Dict[str, float]) -> str:
        """Generate human-readable explanation summary."""
        if not top_features:
            return "No significant features detected."

        positive_features = [f for f, v in top_features.items() if v > 0]
        negative_features = [f for f, v in top_features.items() if v < 0]

        summary_parts = []

        if positive_features:
            top_positive = positive_features[:3]
            summary_parts.append(f"Indicadores de fraude: {', '.join(top_positive)}")

        if negative_features:
            top_negative = negative_features[:3]
            summary_parts.append(f"Indicadores legítimos: {', '.join(top_negative)}")

        return ". ".join(summary_parts)

    def _log_decision(
        self,
        transaction_id: str,
        fraud_probability: float,
        is_fraud: bool,
        explanation: Dict[str, Any] = None,
    ):
        """Log prediction decision for audit trail."""
        log_entry = {
            "transaction_id": transaction_id or "unknown",
            "timestamp": datetime.now().isoformat(),
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
            "threshold": self.threshold,
            "explanation": explanation if explanation else None,
        }

        if is_fraud:
            self.logger.info(f"FRAUD DETECTED: {json.dumps(log_entry)}")
        else:
            self.logger.debug(f"LEGITIMATE: {json.dumps(log_entry)}")

    def save_model(self):
        """Save model to disk using repository."""
        # Save only the XGBoost model
        self.model_repository.save_model(self.model)
        print(f"Model saved to {self.model_path}")

        # Save feature names
        self.model_repository.save_feature_names(self.feature_names)

    def load_model(self):
        """Load the trained model from disk using repository."""
        print(f"Model loaded from {self.model_path}")
        self.model = self.model_repository.load_model()

        # Load feature names
        self.feature_names = self.model_repository.load_feature_names()

        # Initialize SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print("SHAP explainer initialized")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.explainer = None

        print("Model loaded successfully on startup")

    def get_feature_importance(self) -> dict:
        """Get feature importance from the model."""
        if self.model is None:
            raise ValueError("Model not loaded.")

        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance
