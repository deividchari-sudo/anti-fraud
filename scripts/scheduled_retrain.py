"""
Scheduled Retraining Script (Sprint 2).

Runs the full retraining pipeline:
1. Loads latest dataset
2. Trains StackingFraudModel
3. Logs metrics with timestamp for drift monitoring
4. Saves new model bundle (overwrites previous)

Designed to be triggered by cron / Windows Task Scheduler / GitHub Actions weekly.

Usage:
    python scripts/scheduled_retrain.py
    python scripts/scheduled_retrain.py --dataset path/to/data.csv
    python scripts/scheduled_retrain.py --dry-run

Concept Drift Monitoring:
    Each run appends to logs/retrain_history.json with:
    - timestamp
    - dataset size
    - AUC-ROC, F1, Precision, Recall
    - threshold values
    Allows comparing weekly metrics to detect concept drift.
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stacking_model import StackingFraudModel  # noqa: E402
from train_model import load_and_prepare_data  # noqa: E402

RETRAIN_HISTORY_PATH = Path("logs/retrain_history.json")


def append_retrain_history(entry: dict) -> None:
    """Append retrain entry to history file (creates if missing)."""
    RETRAIN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if RETRAIN_HISTORY_PATH.exists():
        try:
            with open(RETRAIN_HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(entry)
    with open(RETRAIN_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)


def detect_drift(history: list) -> dict:
    """Compare last run's metrics to previous to detect concept drift."""
    if len(history) < 2:
        return {"drift_detected": False, "reason": "Not enough history"}

    last = history[-1]
    prev = history[-2]
    auc_drop = prev.get("auc_roc", 0) - last.get("auc_roc", 0)
    f1_drop = prev.get("f1_score", 0) - last.get("f1_score", 0)

    drift = auc_drop > 0.05 or f1_drop > 0.10
    return {
        "drift_detected": drift,
        "auc_drop": round(auc_drop, 4),
        "f1_drop": round(f1_drop, 4),
        "previous_auc": prev.get("auc_roc"),
        "current_auc": last.get("auc_roc"),
    }


def main():
    parser = argparse.ArgumentParser(description="Scheduled stacking model retrain")
    parser.add_argument(
        "--dataset",
        default="dataset_transacoes_expanded.csv",
        help="Path to training dataset CSV",
    )
    parser.add_argument(
        "--model-path",
        default="models/stacking_fraud_model.pkl",
        help="Path to save the model bundle",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run training but don't save model or update history",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"  SCHEDULED RETRAIN - {datetime.now().isoformat()}")
    print("=" * 70)

    print(f"\nLoading dataset: {args.dataset}")
    df = load_and_prepare_data()
    print(f"Dataset size: {len(df)} samples")

    if args.dry_run:
        print("\n[DRY-RUN] Would train StackingFraudModel; skipping.")
        return 0

    model = StackingFraudModel(model_path=args.model_path)
    metrics = model.train(df)

    # Persist run summary for drift monitoring
    entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(df),
        "auc_roc": metrics.get("auc_roc"),
        "f1_score": metrics.get("f1_score"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "threshold": model.threshold,
        "thresholds_by_channel": model.thresholds_by_channel,
        "thresholds_by_product": model.thresholds_by_product,
        "model_path": args.model_path,
    }
    append_retrain_history(entry)

    # Concept drift check
    with open(RETRAIN_HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)
    drift_report = detect_drift(history)

    print("\n=== Retrain Summary ===")
    print(f"  AUC-ROC: {entry['auc_roc']:.4f}")
    print(f"  F1-Score: {entry['f1_score']:.4f}")
    print(f"  Precision: {entry['precision']:.4f}")
    print(f"  Recall: {entry['recall']:.4f}")
    print(f"  Threshold: {entry['threshold']:.4f}")
    print(f"  History file: {RETRAIN_HISTORY_PATH}")

    print("\n=== Concept Drift Check ===")
    if drift_report["drift_detected"]:
        print(f"  ⚠️  DRIFT DETECTED: {drift_report}")
    else:
        print(f"  ✅ No drift: {drift_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
