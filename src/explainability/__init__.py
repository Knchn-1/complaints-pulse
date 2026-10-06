"""Explainability module — feature attributions and decision rationale."""

from src.explainability.explainer import (
    ModelExplainer,
    FeatureWeight,
    ExplanationResult,
)

__all__ = ["ModelExplainer", "FeatureWeight", "ExplanationResult"]
