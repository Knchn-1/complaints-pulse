"""
ComplaintsPulse — Complaint Classifier Pipeline

Combines n-gram TF-IDF feature extraction with class-balanced Logistic Regression
and Platt probability calibration (CalibratedClassifierCV). Also includes
uncertainty and out-of-domain (OOD) detection via Shannon entropy and confidence gating.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    BASELINE_CLF,
    BASELINE_TFIDF,
    IMPROVED_CLF,
    IMPROVED_TFIDF,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.preprocessing.cleaner import clean_text


@dataclass
class PredictionResult:
    """Structured prediction output with calibration and uncertainty flags."""
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    entropy_score: float
    is_uncertain: bool
    status: str  # "Confident", "Moderate Confidence", "Uncertain / Low Confidence"


def build_baseline_pipeline() -> Pipeline:
    """Constructs the baseline TF-IDF + LogisticRegression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**BASELINE_TFIDF)),
        ("clf", LogisticRegression(**BASELINE_CLF)),
    ])


def build_improved_pipeline() -> Pipeline:
    """Constructs the tuned n-gram, sublinear TF-IDF + balanced LogisticRegression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**IMPROVED_TFIDF)),
        ("clf", LogisticRegression(**IMPROVED_CLF)),
    ])


class ComplaintClassifier:
    """
    Production-grade classifier with calibration and uncertainty gating.
    """

    def __init__(
        self,
        pipeline_type: str = "improved",
        confidence_threshold: float = 0.40,
        entropy_threshold: float = 0.82,
    ):
        self.pipeline_type = pipeline_type
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        
        if pipeline_type == "baseline":
            self.raw_pipeline = build_baseline_pipeline()
        else:
            self.raw_pipeline = build_improved_pipeline()

        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.is_fitted = False
        self.is_calibrated = False
        self.classes_: Optional[List[str]] = None

    def fit(self, X_train: Union[pd.Series, List[str]], y_train: Union[pd.Series, List[str]]) -> "ComplaintClassifier":
        """Fit the TF-IDF vectorizer and base classifier on training data."""
        self.raw_pipeline.fit(X_train, y_train)
        self.classes_ = list(self.raw_pipeline.named_steps["clf"].classes_)
        self.is_fitted = True
        return self

    def calibrate(self, X_val: Union[pd.Series, List[str]], y_val: Union[pd.Series, List[str]]) -> "ComplaintClassifier":
        """
        Calibrate probabilities using holdout validation data via Platt scaling (sigmoid).
        Pre-fit base estimator ensures zero validation leakage into training weights.
        """
        if not self.is_fitted:
            raise ValueError("Base pipeline must be fitted before calibration.")

        # Transform X_val with the trained vectorizer
        tfidf = self.raw_pipeline.named_steps["tfidf"]
        clf = self.raw_pipeline.named_steps["clf"]
        
        X_val_tfidf = tfidf.transform(X_val)

        # Calibrate classifier on validation embeddings using all validation data
        n_val = X_val_tfidf.shape[0]
        custom_cv = [(np.arange(n_val), np.arange(n_val))]
        try:
            from sklearn.frozen import FrozenEstimator
            calibrator = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid", cv=custom_cv)
        except (ImportError, TypeError):
            calibrator = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")

        calibrator.fit(X_val_tfidf, y_val)
        
        self.calibrated_model = calibrator
        self.is_calibrated = True
        return self

    def predict_proba(self, X: Union[pd.Series, List[str], str]) -> np.ndarray:
        """Return calibrated (or raw if uncalibrated) class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier is not fitted.")

        if isinstance(X, str):
            X = [X]

        tfidf = self.raw_pipeline.named_steps["tfidf"]
        X_trans = tfidf.transform(X)

        if self.is_calibrated and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_trans)
        
        return self.raw_pipeline.named_steps["clf"].predict_proba(X_trans)

    def predict(self, X: Union[pd.Series, List[str], str]) -> np.ndarray:
        """Predict highest-probability class labels."""
        probas = self.predict_proba(X)
        top_indices = np.argmax(probas, axis=1)
        classes_arr = np.array(self.classes_)
        return classes_arr[top_indices]

    def predict_single(self, text: str) -> PredictionResult:
        """
        Predict a single complaint string with uncertainty and OOD assessment.
        """
        cleaned = clean_text(text)
        probas = self.predict_proba([cleaned])[0]
        
        n_classes = len(self.classes_)
        # Normalized entropy in [0, 1]
        raw_ent = float(entropy(probas, base=n_classes)) if n_classes > 1 else 0.0
        
        top_idx = int(np.argmax(probas))
        confidence = float(probas[top_idx])
        pred_class = self.classes_[top_idx]

        is_uncertain = (confidence < self.confidence_threshold) or (raw_ent > self.entropy_threshold)

        if confidence >= 0.70:
            status = "Confident"
        elif confidence >= self.confidence_threshold:
            status = "Moderate Confidence"
        else:
            status = "Uncertain / Low Confidence"

        prob_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(self.classes_, probas)
        }
        # Sort descending by probability
        prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        return PredictionResult(
            predicted_category=pred_class,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            entropy_score=round(raw_ent, 4),
            is_uncertain=is_uncertain,
            status=status,
        )

    def get_feature_names(self) -> np.ndarray:
        """Return the TF-IDF feature vocabulary."""
        return self.raw_pipeline.named_steps["tfidf"].get_feature_names_out()

    def get_classifier_coefficients(self) -> Tuple[np.ndarray, List[str]]:
        """Return raw Logistic Regression weights and class order."""
        clf = self.raw_pipeline.named_steps["clf"]
        return clf.coef_, list(clf.classes_)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persist the complete model artifact."""
        path = Path(filepath) if filepath else MODELS_DIR / "classifier.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "ComplaintClassifier":
        """Load a persisted classifier model."""
        path = Path(filepath) if filepath else MODELS_DIR / "classifier.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Classifier model file not found at: {path}")
        return joblib.load(path)
