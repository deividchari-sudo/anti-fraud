"""
QA validation script for EnsembleFraudModel.
Measures latency, throughput, and validates acceptance criteria.
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble_model import EnsembleFraudModel
from train_model import load_and_prepare_data


def main():
    print("=" * 70)
    print("  [QA] ENSEMBLE MODEL ACCEPTANCE CRITERIA VALIDATION")
    print("=" * 70)

    # Load model
    model = EnsembleFraudModel(model_path="models/ensemble_fraud_model.pkl")
    if model.xgb_model is None:
        print("❌ Model not found. Run `python train_ensemble.py` first.")
        return 1

    print(f"\n✅ Model loaded successfully")
    print(f"   Features: {len(model.feature_names)}")
    print(f"   Global threshold: {model.threshold:.4f}")
    print(f"   Per-channel thresholds: {len(model.thresholds_by_channel)}")
    print(f"   Per-product thresholds: {len(model.thresholds_by_product)}")

    # Load test data
    print("\n📊 Loading test data...")
    df = load_and_prepare_data()
    sample = df.drop(columns=["fraudResult", "payload"]).head(1000)

    # Warm up
    _ = model.predict(sample.head(1))

    # Latency test (single transaction)
    print("\n⏱️  Latency benchmark (1000 single predictions):")
    latencies = []
    for i in range(min(1000, len(sample))):
        t0 = time.perf_counter()
        model.predict(sample.iloc[[i]], transaction_id=f"qa-{i}")
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"   Mean: {latencies.mean():.2f}ms")
    print(f"   Median (P50): {np.percentile(latencies, 50):.2f}ms")
    print(f"   P95: {np.percentile(latencies, 95):.2f}ms")
    print(f"   P99: {np.percentile(latencies, 99):.2f}ms")
    print(f"   Max: {latencies.max():.2f}ms")

    # Acceptance criteria
    print("\n" + "=" * 70)
    print("  ACCEPTANCE CRITERIA")
    print("=" * 70)
    criteria = [
        ("Latency P95 < 100ms (BACEN)", np.percentile(latencies, 95) < 100),
        ("Latency P99 < 200ms", np.percentile(latencies, 99) < 200),
        ("Mean latency < 50ms", latencies.mean() < 50),
        ("Model has 2 base estimators", model.xgb_model is not None and model.lgb_model is not None),
        ("Per-channel thresholds configured", len(model.thresholds_by_channel) > 0),
        ("Per-product thresholds configured", len(model.thresholds_by_product) > 0),
        ("SHAP explainability available", model.explainer is not None),
        ("Feature names persisted", len(model.feature_names) > 0),
    ]

    passed = 0
    for name, ok in criteria:
        symbol = "✅" if ok else "❌"
        print(f"  {symbol} {name}")
        if ok:
            passed += 1

    print(f"\n  Result: {passed}/{len(criteria)} criteria passed")
    return 0 if passed == len(criteria) else 1


if __name__ == "__main__":
    sys.exit(main())
