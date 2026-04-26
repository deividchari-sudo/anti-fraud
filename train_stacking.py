"""
Training script for StackingFraudModel (Sprint 2).
Reuses data prep from train_model.py and trains the 3-base + LR stacking ensemble.
"""

from src.stacking_model import StackingFraudModel
from train_model import load_and_prepare_data


def main():
    print("=" * 72)
    print("  STACKING FRAUD MODEL TRAINING (Sprint 2)")
    print("  Stack: XGBoost + LightGBM + CatBoost -> Logistic Regression (calibrated)")
    print("=" * 72)

    df = load_and_prepare_data()

    model = StackingFraudModel(
        model_path="models/stacking_fraud_model.pkl",
        n_folds=3,
    )

    metrics = model.train(df)

    print("\n=== Top 10 Feature Importance (Stacking Avg) ===")
    importance = model.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
        print(f"{i}. {feature}: {score:.4f}")

    print("\n=== Training Complete ===")
    print(f"Model saved to: {model.model_path}")
    print(f"Global threshold: {model.threshold:.4f}")
    print(f"Channel thresholds: {model.thresholds_by_channel}")
    print(f"Product thresholds: {model.thresholds_by_product}")
    print("\nFinal Metrics:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")


if __name__ == "__main__":
    main()
