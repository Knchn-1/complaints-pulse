"""Classification module — baselines, improved TF-IDF, training, calibration."""

from src.classification.data_loader import (
    DatasetSplits,
    load_raw_complaints,
    load_and_split_data,
)
from src.classification.evaluator import (
    compute_multiclass_brier_score,
    evaluate_model_performance,
)
from src.classification.model import (
    ComplaintClassifier,
    PredictionResult,
    build_baseline_pipeline,
    build_improved_pipeline,
)

__all__ = [
    "DatasetSplits",
    "load_raw_complaints",
    "load_and_split_data",
    "compute_multiclass_brier_score",
    "evaluate_model_performance",
    "ComplaintClassifier",
    "PredictionResult",
    "build_baseline_pipeline",
    "build_improved_pipeline",
]
