"""
Training script for Ensemble Fraud Model (Sprint 1).
Reuses data prep from train_model.py and trains XGBoost + LightGBM ensemble
with isotonic calibration and per-channel/product threshold tuning.
"""

from src.ensemble_model import EnsembleFraudModel
from train_model import load_and_prepare_data


def main():
    print("=" * 70)
    print("  ENSEMBLE FRAUD MODEL TRAINING (Sprint 1)")
    print("  Stack: XGBoost + LightGBM + Isotonic Calibration + Per-Channel Thresholds")
    print("=" * 70)

    df = load_and_prepare_data()

    # Initialize model with equal weights (can be tuned later)
    # XGBoost weighted higher (slightly better AUC on this dataset)
    model = EnsembleFraudModel(
        model_path="models/ensemble_fraud_model.pkl",
        xgb_weight=0.6,
        lgb_weight=0.4,
    )

    metrics = model.train(df)

    print("\n=== Top 10 Feature Importance (Ensemble Avg) ===")
    importance = model.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
        print(f"{i}. {feature}: {score:.4f}")

    print("\n=== Training Complete ===")
    print(f"Model saved to: {model.model_path}")
    print(f"Global threshold: {model.threshold:.4f}")
    print(f"Channel thresholds: {model.thresholds_by_channel}")
    print(f"Product thresholds: {model.thresholds_by_product}")
    print(f"\nFinal Metrics:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")


if __name__ == "__main__":
    main()
