"""
Train segmented fraud detection models per (product, channel) segment.

This script implements the Abordagem B (model-level segmentation) discussed
in the squad ML Experiment workflow. For each priority segment, it:
1. Filters transactions by product and channel
2. Extracts 71 features via FeatureEngineer
3. Trains a FraudDetectionModel (XGBoost solo)
4. Saves to models/fraud_model_{produto}_{canal}.pkl

Usage:
    python train_segmented_models.py

Priority segments: pix_mobile, pix_web, cartao_web
Fallback: all other segments use the global model (fraud_model.pkl)
"""

import json
import sys
from pathlib import Path

import pandas as pd

from src.feature_engineering import FeatureEngineer
from src.model import FraudDetectionModel
from src.repositories import CSVTransactionRepository


SEGMENTS = [
    ("pix", "app"),
    ("pix", "web"),
    ("ted", "web"),
]

MIN_SAMPLES = 5000
MIN_FRAUDS = 40


def load_and_prepare_segment(df_raw: pd.DataFrame, product: str, channel: str) -> pd.DataFrame:
    """Extract features for a specific (product, channel) segment."""
    feature_engineer = FeatureEngineer()

    payloads = []
    for payload_str in df_raw["payload"]:
        try:
            payloads.append(json.loads(payload_str))
        except Exception:
            payloads.append({})

    # Filter by product and channel
    mask = []
    for p in payloads:
        p_prod = str(p.get("produto", "")).lower().strip()
        p_chan = str(p.get("canal", "")).lower().strip()
        mask.append(p_prod == product and p_chan == channel)

    if not any(mask):
        return pd.DataFrame()

    segment_df = df_raw[mask].reset_index(drop=True)
    segment_payloads = [p for p, m in zip(payloads, mask) if m]

    print(f"\n=== Segment {product}_{channel} ===")
    print(f"Samples: {len(segment_df)}")
    if len(segment_df) == 0:
        return pd.DataFrame()

    frauds = segment_df["fraudResult"].sum()
    print(f"Frauds: {frauds} ({frauds / len(segment_df) * 100:.2f}%)")

    if len(segment_df) < MIN_SAMPLES:
        print(f"SKIPPED: insufficient samples (< {MIN_SAMPLES})")
        return pd.DataFrame()
    if frauds < MIN_FRAUDS:
        print(f"SKIPPED: insufficient frauds (< {MIN_FRAUDS})")
        return pd.DataFrame()

    # Extract features
    features_list = []
    for payload in segment_payloads:
        features = feature_engineer.extract_features(payload)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # One-hot encode categoricals
    categorical_cols = ["canal", "produto", "jornada", "direcao"]
    for col in categorical_cols:
        if col in features_df.columns:
            dummies = pd.get_dummies(features_df[col], prefix=col)
            features_df = pd.concat([features_df, dummies], axis=1)
            features_df = features_df.drop(col, axis=1)

    # Combine with target
    df_prepared = pd.concat([features_df, segment_df["fraudResult"]], axis=1)
    df_prepared["payload"] = segment_df["payload"].values

    print(f"Feature shape: {df_prepared.shape}")
    return df_prepared


def train_segment_model(df_prepared: pd.DataFrame, product: str, channel: str) -> dict:
    """Train and save a model for a specific segment."""
    model_path = f"models/fraud_model_{product}_{channel}.pkl"
    feature_names_path = f"models/feature_names_{product}_{channel}.json"

    # Ensure models dir exists
    Path("models").mkdir(exist_ok=True)

    # Initialize model with segment-specific path
    model = FraudDetectionModel(model_path=None)  # Train from scratch
    model.model_path = model_path
    model.model_repository.model_path = model_path
    model.model_repository.feature_names_path = feature_names_path

    # Train
    metrics = model.train(df_prepared)

    # Save
    model.save_model()
    model.model_repository.save_feature_names(model.feature_names)

    print(f"\nModel saved to: {model_path}")
    print(f"Feature names saved to: {feature_names_path}")
    print(f"AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
    print(f"F1-Score: {metrics.get('f1_score', 0):.4f}")

    return metrics


def main():
    print("=" * 60)
    print("Segmented Model Training — Squad ML Experiment")
    print("Segments:", [f"{p}_{c}" for p, c in SEGMENTS])
    print("=" * 60)

    # Load raw data
    repo = CSVTransactionRepository("dataset_transacoes_expanded.csv")
    print("\nLoading dataset...")
    df_raw = repo.load_dataset()
    print(f"Total samples: {len(df_raw)}")

    results = {}
    for product, channel in SEGMENTS:
        df_prepared = load_and_prepare_segment(df_raw, product, channel)
        if df_prepared.empty:
            results[f"{product}_{channel}"] = {"status": "skipped", "reason": "insufficient_data"}
            continue

        try:
            metrics = train_segment_model(df_prepared, product, channel)
            results[f"{product}_{channel}"] = {
                "status": "trained",
                "auc_roc": metrics.get("auc_roc"),
                "f1_score": metrics.get("f1_score"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "samples": len(df_prepared),
            }
        except Exception as e:
            print(f"ERROR training {product}_{channel}: {e}")
            results[f"{product}_{channel}"] = {"status": "error", "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for seg, res in results.items():
        status = res["status"]
        if status == "trained":
            print(
                f"✅ {seg}: AUC={res['auc_roc']:.4f} F1={res['f1_score']:.4f} "
                f"n={res['samples']}"
            )
        elif status == "skipped":
            print(f"⚠️  {seg}: skipped ({res['reason']})")
        else:
            print(f"❌ {seg}: error — {res.get('error', 'unknown')}")

    return results


if __name__ == "__main__":
    main()
