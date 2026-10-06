"""
ComplaintsPulse — Classifier Training & Rigorous Evaluation Script

Trains:
1. Baseline Model: Unigrams (5k) + Standard Logistic Regression
2. Improved Model: Unigram + Bigram (15k), sublinear TF + Class-balanced Logistic Regression
3. Calibrated Model: Improved model + Platt calibration fitted on validation split

Evaluates strictly on holdout test set with no leakage.
Saves model artifacts and performance comparison JSON.
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is in python search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.classification.data_loader import load_and_split_data
from src.classification.evaluator import evaluate_model_performance
from src.classification.model import ComplaintClassifier
from src.config import MODELS_DIR


def run_training_pipeline(sample_size: int = None):
    print("=" * 70)
    print("ComplaintsPulse — Training & Calibration Pipeline")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/5] Loading and splitting data (sample_size={sample_size or 'FULL'})...")
    splits = load_and_split_data(sample_size=sample_size, clean_text_flag=False)
    summary = splits.summary()
    print(f"      Train samples: {summary['train_size']:,}")
    print(f"      Val samples:   {summary['val_size']:,}")
    print(f"      Test samples:  {summary['test_size']:,}")
    print(f"      Categories:    {', '.join(splits.label_names)}")

    results = {}

    # 2. Train & Evaluate Baseline
    print("\n[2/5] Training Baseline Pipeline (TF-IDF 5k unigram + LogisticRegression)...")
    t0 = time.time()
    baseline = ComplaintClassifier(pipeline_type="baseline")
    baseline.fit(splits.X_train, splits.y_train)
    baseline_train_time = round(time.time() - t0, 2)
    print(f"      Baseline trained in {baseline_train_time}s")

    print("      Evaluating Baseline on holdout Test set...")
    baseline_preds = baseline.predict(splits.X_test)
    baseline_proba = baseline.predict_proba(splits.X_test)
    baseline_metrics = evaluate_model_performance(
        y_true=splits.y_test.values,
        y_pred=baseline_preds,
        y_proba=baseline_proba,
        classes=splits.label_names,
    )
    baseline_metrics["training_time_sec"] = baseline_train_time
    results["baseline"] = baseline_metrics
    print(f"      -> Macro F1: {baseline_metrics['macro_f1']:.4f} | Accuracy: {baseline_metrics['accuracy']:.4f} | Brier: {baseline_metrics.get('brier_score', 'N/A')}")

    # 3. Train Improved Model
    print("\n[3/5] Training Improved Pipeline (TF-IDF 15k uni+bi, sublinear + Balanced LogisticRegression)...")
    t0 = time.time()
    improved = ComplaintClassifier(pipeline_type="improved")
    improved.fit(splits.X_train, splits.y_train)
    improved_train_time = round(time.time() - t0, 2)
    print(f"      Improved model trained in {improved_train_time}s")

    print("      Evaluating Improved Model (uncalibrated) on holdout Test set...")
    improved_preds = improved.predict(splits.X_test)
    improved_proba = improved.predict_proba(splits.X_test)
    improved_metrics = evaluate_model_performance(
        y_true=splits.y_test.values,
        y_pred=improved_preds,
        y_proba=improved_proba,
        classes=splits.label_names,
    )
    improved_metrics["training_time_sec"] = improved_train_time
    results["improved_uncalibrated"] = improved_metrics
    print(f"      -> Macro F1: {improved_metrics['macro_f1']:.4f} | Accuracy: {improved_metrics['accuracy']:.4f} | Brier: {improved_metrics.get('brier_score', 'N/A')}")

    # 4. Calibrate Model on Validation Set
    print("\n[4/5] Calibrating Improved Model using Validation set (Platt scaling)...")
    t0 = time.time()
    improved.calibrate(splits.X_val, splits.y_val)
    calib_time = round(time.time() - t0, 2)
    print(f"      Calibrated in {calib_time}s")

    print("      Evaluating Calibrated Model on holdout Test set...")
    calib_preds = improved.predict(splits.X_test)
    calib_proba = improved.predict_proba(splits.X_test)
    calib_metrics = evaluate_model_performance(
        y_true=splits.y_test.values,
        y_pred=calib_preds,
        y_proba=calib_proba,
        classes=splits.label_names,
    )
    calib_metrics["training_time_sec"] = round(improved_train_time + calib_time, 2)
    results["improved_calibrated"] = calib_metrics
    print(f"      -> Macro F1: {calib_metrics['macro_f1']:.4f} | Accuracy: {calib_metrics['accuracy']:.4f} | Brier: {calib_metrics.get('brier_score', 'N/A')}")

    # 5. Summary Comparison Table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY (Holdout Test Set)")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12} {'Brier Score':<12} {'Log Loss':<10}")
    print("-" * 70)
    for model_name, m in results.items():
        print(f"{model_name:<25} {m['accuracy']:<10.4f} {m['macro_f1']:<10.4f} {m['weighted_f1']:<12.4f} {m.get('brier_score', 0.0):<12.4f} {m.get('log_loss', 0.0):<10.4f}")
    print("=" * 70)

    # 6. Save Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = improved.save(MODELS_DIR / "classifier.joblib")
    metrics_path = MODELS_DIR / "classification_metrics.json"
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[5/5] Artifacts saved:")
    print(f"      -> Model:   {model_path}")
    print(f"      -> Metrics: {metrics_path}")
    print("\nTraining & evaluation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ComplaintsPulse classifier")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for training (omit for full dataset)",
    )
    args = parser.parse_args()
    run_training_pipeline(sample_size=args.sample_size)
