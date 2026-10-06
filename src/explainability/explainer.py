"""
ComplaintsPulse — Linear Feature Attribution & Model Explainability

Calculates exact local feature attributions for linear NLP models (TF-IDF + Logistic Regression).
Computes:
1. Local term contributions: contribution = w_{class, term} * tfidf_{term}
2. Evidence for vs. evidence against the predicted category
3. Contrastive explanations: what differentiated the top class from the runner-up class
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.classification.model import ComplaintClassifier
from src.preprocessing.cleaner import clean_text


@dataclass
class FeatureWeight:
    """Represents a single word/n-gram's mathematical attribution."""
    term: str
    tfidf_value: float
    coefficient: float
    contribution: float  # tfidf_value * coefficient

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "tfidf_value": self.tfidf_value,
            "coefficient": self.coefficient,
            "contribution": self.contribution,
        }


@dataclass
class ExplanationResult:
    """Structured explainability output for a single prediction."""
    predicted_category: str
    top_positive_features: List[FeatureWeight]  # Pushed towards predicted class
    top_negative_features: List[FeatureWeight]  # Pushed away from predicted class
    runner_up_category: Optional[str]
    contrastive_features: List[FeatureWeight]    # Differentiating top class from runner-up
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_category": self.predicted_category,
            "top_positive_features": [f.to_dict() for f in self.top_positive_features],
            "top_negative_features": [f.to_dict() for f in self.top_negative_features],
            "runner_up_category": self.runner_up_category,
            "contrastive_features": [f.to_dict() for f in self.contrastive_features],
            "summary_text": self.summary_text,
        }


class ModelExplainer:
    """
    Intrinsically interpretable feature attribution engine.
    """

    def __init__(self, classifier: ComplaintClassifier):
        self.classifier = classifier
        if not self.classifier.is_fitted:
            raise ValueError("Classifier must be fitted before initializing explainer.")
        
        self.tfidf = self.classifier.raw_pipeline.named_steps["tfidf"]
        self.clf = self.classifier.raw_pipeline.named_steps["clf"]
        self.feature_names = self.tfidf.get_feature_names_out()
        self.classes = list(self.clf.classes_)

    def explain(
        self,
        text: str,
        top_k: int = 6,
    ) -> ExplanationResult:
        """
        Explain the model's prediction on the given text.
        """
        cleaned = clean_text(text)
        tfidf_vec = self.tfidf.transform([cleaned])
        probas = self.classifier.predict_proba([cleaned])[0]

        # Identify predicted class and runner-up class
        sorted_indices = np.argsort(probas)[::-1]
        top_idx = int(sorted_indices[0])
        runner_up_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else None

        predicted_class = self.classes[top_idx]
        runner_up_class = self.classes[runner_up_idx] if runner_up_idx is not None else None

        # Extract non-zero features in the input text
        feature_indices = tfidf_vec.indices
        tfidf_values = tfidf_vec.data

        if len(feature_indices) == 0:
            return ExplanationResult(
                predicted_category=predicted_class,
                top_positive_features=[],
                top_negative_features=[],
                runner_up_category=runner_up_class,
                contrastive_features=[],
                summary_text="No known vocabulary tokens detected in the text.",
            )

        # Get weights for predicted class
        coefs = self.clf.coef_[top_idx]
        contributions = tfidf_values * coefs[feature_indices]

        # Build feature weight objects
        feature_items = [
            FeatureWeight(
                term=str(self.feature_names[f_idx]),
                tfidf_value=round(float(val), 4),
                coefficient=round(float(coefs[f_idx]), 4),
                contribution=round(float(contrib), 4),
            )
            for f_idx, val, contrib in zip(feature_indices, tfidf_values, contributions)
        ]

        # Top positive (evidence FOR predicted class)
        pos_sorted = sorted([f for f in feature_items if f.contribution > 0], key=lambda x: x.contribution, reverse=True)
        top_positive = pos_sorted[:top_k]

        # Top negative (evidence AGAINST predicted class)
        neg_sorted = sorted([f for f in feature_items if f.contribution < 0], key=lambda x: x.contribution)
        top_negative = neg_sorted[:top_k]

        # Contrastive explanation against runner-up
        contrastive_items = []
        if runner_up_idx is not None:
            runner_up_coefs = self.clf.coef_[runner_up_idx]
            contrast_contribs = tfidf_values * (coefs[feature_indices] - runner_up_coefs[feature_indices])
            for f_idx, val, c_contrib in zip(feature_indices, tfidf_values, contrast_contribs):
                if c_contrib > 0:
                    contrastive_items.append(
                        FeatureWeight(
                            term=str(self.feature_names[f_idx]),
                            tfidf_value=round(float(val), 4),
                            coefficient=round(float(coefs[f_idx] - runner_up_coefs[f_idx]), 4),
                            contribution=round(float(c_contrib), 4),
                        )
                    )
            contrastive_items = sorted(contrastive_items, key=lambda x: x.contribution, reverse=True)[:top_k]

        # Human-readable summary
        summary = self._build_summary(
            predicted_class=predicted_class,
            top_positive=top_positive,
            runner_up_class=runner_up_class,
            contrastive=contrastive_items,
        )

        return ExplanationResult(
            predicted_category=predicted_class,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            runner_up_category=runner_up_class,
            contrastive_features=contrastive_items,
            summary_text=summary,
        )

    def _build_summary(
        self,
        predicted_class: str,
        top_positive: List[FeatureWeight],
        runner_up_class: Optional[str],
        contrastive: List[FeatureWeight],
    ) -> str:
        pos_terms = [f"'{f.term}' (+{f.contribution:.2f})" for f in top_positive[:4]]
        if not pos_terms:
            return f"Classified as {predicted_class} based on general document distribution."

        pos_str = ", ".join(pos_terms)
        summary = f"Assigned '{predicted_class}' primarily due to strong indicative signals: {pos_str}."
        
        if runner_up_class and contrastive:
            diff_terms = [f"'{f.term}'" for f in contrastive[:3]]
            diff_str = ", ".join(diff_terms)
            summary += f" Differentiated from '{runner_up_class}' by the presence of {diff_str}."

        return summary


# Export alias
ComplaintExplainer = ModelExplainer
