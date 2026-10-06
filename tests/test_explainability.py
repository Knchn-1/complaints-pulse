"""
Unit tests for model explainability and feature attributions.
"""

import pytest
from src.classification.model import ComplaintClassifier
from src.explainability.explainer import ModelExplainer


@pytest.fixture(scope="module")
def trained_classifier():
    return ComplaintClassifier.load()


def test_explainer_positive_attributions(trained_classifier):
    explainer = ModelExplainer(trained_classifier)
    text = "The mortgage escrow calculation on my home loan is completely wrong."
    explanation = explainer.explain(text, top_k=5)

    assert explanation.predicted_category == "Mortgages & Loans"
    assert len(explanation.top_positive_features) > 0
    
    # Check that key mortgage words carry positive contribution
    pos_terms = [f.term for f in explanation.top_positive_features]
    assert any(term in ("mortgage", "loan", "escrow") for term in pos_terms)
    assert all(f.contribution > 0 for f in explanation.top_positive_features)
    assert explanation.summary_text is not None


def test_explainer_contrastive_with_runner_up(trained_classifier):
    explainer = ModelExplainer(trained_classifier)
    text = "Debt collection agency called repeatedly about credit card balance payment."
    explanation = explainer.explain(text, top_k=5)

    assert explanation.runner_up_category is not None
    assert explanation.predicted_category != explanation.runner_up_category
    assert len(explanation.contrastive_features) > 0


def test_explainer_empty_text(trained_classifier):
    explainer = ModelExplainer(trained_classifier)
    explanation = explainer.explain("", top_k=5)

    assert explanation.predicted_category in trained_classifier.classes_
    assert len(explanation.top_positive_features) == 0
    assert "No known vocabulary" in explanation.summary_text
