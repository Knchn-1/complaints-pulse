"""
Unit tests for classifier model, probability calibration, evaluation, and serialization.
"""

import numpy as np
import pytest
from src.classification.evaluator import compute_multiclass_brier_score, evaluate_model_performance
from src.classification.model import ComplaintClassifier, build_baseline_pipeline, build_improved_pipeline


@pytest.fixture
def synthetic_data():
    X_train = [
        "Unauthorized fraud charge on my credit card stolen money",
        "Credit report has incorrect dispute error and wrong score",
        "Debt collector keeps calling harassing me for invalid debt",
        "Mortgage interest rate calculation is wrong on my loan",
        "Checking account overdraft fee charged by bank unlawfully",
    ] * 10
    y_train = [
        "Credit Card",
        "Credit Reporting",
        "Debt Collection",
        "Mortgages & Loans",
        "Retail Banking",
    ] * 10

    X_val = [
        "Stolen card fraud payment charged to my account",
        "Wrong delinquent mark on credit bureau report",
        "Collection agency calling non stop",
        "Loan balance escrow calculation error",
        "Atm deposit missing from checking account",
    ] * 4
    y_val = [
        "Credit Card",
        "Credit Reporting",
        "Debt Collection",
        "Mortgages & Loans",
        "Retail Banking",
    ] * 4

    return X_train, y_train, X_val, y_val


def test_brier_score_perfect():
    # 2 samples, 3 classes, perfect probability matches
    y_indices = np.array([0, 2])
    y_proba = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    brier = compute_multiclass_brier_score(y_indices, y_proba)
    assert brier == 0.0


def test_classifier_fit_calibrate_and_predict(synthetic_data, tmp_path):
    X_train, y_train, X_val, y_val = synthetic_data

    clf = ComplaintClassifier(pipeline_type="improved")
    clf.fit(X_train, y_train)

    assert clf.is_fitted
    assert not clf.is_calibrated

    # Calibration
    clf.calibrate(X_val, y_val)
    assert clf.is_calibrated

    # Predict single
    res = clf.predict_single("Fraudulent charge on my stolen credit card")
    assert res.predicted_category in clf.classes_
    assert 0.0 <= res.confidence <= 1.0
    assert 0.0 <= res.entropy_score <= 1.0
    assert len(res.probabilities) == len(clf.classes_)

    # Batch prediction
    preds = clf.predict(["escrow mortgage payment missing"])
    assert len(preds) == 1

    # Probability shape and sum to 1.0
    probas = clf.predict_proba(["escrow mortgage payment missing"])
    assert probas.shape == (1, len(clf.classes_))
    assert np.isclose(np.sum(probas), 1.0, atol=1e-3)

    # Save and reload test
    save_file = tmp_path / "test_model.joblib"
    clf.save(save_file)
    assert save_file.exists()

    loaded = ComplaintClassifier.load(save_file)
    assert loaded.is_calibrated
    loaded_res = loaded.predict_single("Debt collector call harassment")
    assert loaded_res.predicted_category in loaded.classes_


def test_evaluator_metrics():
    y_true = ["Credit Card", "Debt Collection", "Credit Reporting"]
    y_pred = ["Credit Card", "Debt Collection", "Credit Reporting"]
    y_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])
    classes = ["Credit Card", "Debt Collection", "Credit Reporting"]

    metrics = evaluate_model_performance(y_true, y_pred, y_proba, classes=classes)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert "brier_score" in metrics
    assert metrics["brier_score"] < 0.1
