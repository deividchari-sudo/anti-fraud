"""QA validation for StackingFraudModel acceptance criteria (Sprint 2)."""

import io
import sys
import time
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stacking_model import StackingFraudModel  # noqa: E402
from train_model import load_and_prepare_data  # noqa: E402


def main():
    print("=" * 70)
    print("  [QA] STACKING MODEL ACCEPTANCE CRITERIA (Sprint 2)")
    print("=" * 70)

    model = StackingFraudModel(model_path="models/stacking_fraud_model.pkl")
    if model.stacking_model is None:
        print("ERROR: Model not found. Run python train_stacking.py first.")
        return 1

    print(f"\n[OK] Model loaded")
    print(f"   Features: {len(model.feature_names)}")
    print(f"   Global threshold: {model.threshold:.4f}")
    print(f"   Per-channel thresholds: {len(model.thresholds_by_channel)}")
    print(f"   Per-product thresholds: {len(model.thresholds_by_product)}")

    print("\nLoading test data...")
    df = load_and_prepare_data()
    sample = df.drop(columns=["fraudResult", "payload"]).head(500)

    # Warm up
    _ = model.predict(sample.head(1))

    print("\nLatency benchmark (500 single predictions):")
    latencies = []
    for i in range(min(500, len(sample))):
        t0 = time.perf_counter()
        model.predict(sample.iloc[[i]], transaction_id=f"qa-stacking-{i}")
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"   Mean: {latencies.mean():.2f}ms")
    print(f"   Median (P50): {np.percentile(latencies, 50):.2f}ms")
    print(f"   P95: {np.percentile(latencies, 95):.2f}ms")
    print(f"   P99: {np.percentile(latencies, 99):.2f}ms")

    print("\n" + "=" * 70)
    print("  ACCEPTANCE CRITERIA")
    print("=" * 70)
    criteria = [
        ("Latency P95 < 200ms (acceptable for 3-base stacking)",
         np.percentile(latencies, 95) < 200),
        ("Latency P99 < 300ms",
         np.percentile(latencies, 99) < 300),
        ("Stacking model loaded",
         model.stacking_model is not None),
        ("Per-channel thresholds configured",
         len(model.thresholds_by_channel) > 0),
        ("Per-product thresholds configured",
         len(model.thresholds_by_product) > 0),
        ("SHAP explainability available",
         model.explainer is not None),
        ("Feature names persisted",
         len(model.feature_names) > 0),
    ]

    passed = 0
    for name, ok in criteria:
        symbol = "[OK]" if ok else "[FAIL]"
        print(f"  {symbol} {name}")
        if ok:
            passed += 1

    print(f"\n  Result: {passed}/{len(criteria)} criteria passed")
    return 0 if passed == len(criteria) else 1


if __name__ == "__main__":
    sys.exit(main())
