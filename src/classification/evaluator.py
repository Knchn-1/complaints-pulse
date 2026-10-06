"""
ComplaintsPulse — Model Evaluation Suite

Computes comprehensive classification metrics including Macro F1, Weighted F1,
per-class precision/recall, multiclass Log Loss, and Brier Score calibration.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
)


def compute_multiclass_brier_score(y_true_indices: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Computes the multiclass Brier score:
    Brier = (1 / N) * sum_i sum_k (p_ik - y_ik)^2
    Lower is better calibrated.
    """
    n_samples, n_classes = y_proba.shape
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y_true_indices] = 1.0
    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
) -> Dict:
    """
    Generate a full evaluation report for predictions.
    
    Args:
        y_true: True class labels (strings or ints)
        y_pred: Predicted class labels (strings or ints)
        y_proba: Predicted probability matrix of shape (n_samples, n_classes)
        classes: Ordered list of class label names
        
    Returns:
        Structured dictionary of performance metrics.
    """
    if classes is None:
        classes = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true, y_pred, labels=classes, target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    result = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }
            for cls in classes if cls in report
        },
        "confusion_matrix": cm,
        "classes": classes,
    }

    if y_proba is not None:
        try:
            # Map string classes to indices for log loss and Brier score
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            y_indices = np.array([class_to_idx[y] for y in y_true])

            ll = float(log_loss(y_indices, y_proba, labels=list(range(len(classes)))))
            brier = compute_multiclass_brier_score(y_indices, y_proba)
            result["log_loss"] = round(ll, 4)
            result["brier_score"] = round(brier, 4)
        except Exception as e:
            result["prob_metric_error"] = str(e)

    return result
