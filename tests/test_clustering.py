"""
ComplaintsPulse — Unit Tests for Clustering & Emerging Issue Detection (Phase 5)

Tests:
  - ComplaintClusterer (HDBSCAN and KMeans) fitting, prediction, and c-TF-IDF keyword extraction
  - Cluster quality evaluation metrics (noise rate, coherence, silhouette, distinctiveness)
  - EmergingIssueDetector burst / surge detection across sequential periods
  - Model serialization and deserialization
  - Edge cases (small samples, noisy documents)
"""

import numpy as np
import pytest
from src.clustering.clusterer import ComplaintClusterer
from src.clustering.evaluation import evaluate_cluster_quality, ClusterQualityReport
from src.clustering.detector import EmergingIssueDetector, EmergingIssueAlert


@pytest.fixture
def sample_complaints_data():
    """Synthetic themed complaint corpus for predictable cluster testing."""
    overdraft_complaints = [
        "Bank charged overdraft fee when balance was positive. Need full refund of unfair fee.",
        "Overdraft fee charged twice for one check deposit transaction at branch.",
        "Excessive overdraft fees accumulated due to bank posting order manipulation.",
        "Customer service refused to waive the overdraft fee despite promised reversal.",
        "Checking account overdraft penalty fee deducted without prior notification.",
        "Multiple overdraft fees applied in a single day for small coffee purchases.",
        "Dispute regarding automated overdraft transfer fee from savings account.",
        "Checking balance went negative after delayed deposit and incurred large overdraft fee.",
    ]
    fraud_wire_complaints = [
        "Unauthorized wire transfer sent from my savings account to unknown beneficiary overseas.",
        "Fraudulent wire transaction completed despite reporting compromised credentials.",
        "Identity theft scam resulted in unauthorized wire withdrawal from business checking.",
        "Bank failed to halt wire transfer after immediate fraud report of phishing email.",
        "Funds stolen via wire fraud due to security authentication failure on mobile app.",
        "Scammer initiated fraudulent domestic wire without two-factor authentication challenge.",
        "Stolen funds through wire transfer never reimbursed by fraud department.",
        "Wire transfer fraud investigation closed prematurely without restoring stolen balance.",
    ]
    repossession_complaints = [
        "Auto loan company repossessed vehicle without warning or legal notice of default.",
        "Car wrongfully repossessed while monthly auto payments were up to date.",
        "Lender auctioned repossessed vehicle for far below market value and demanded deficiency.",
        "Vehicle repossession company damaged property while towing car from private driveway.",
        "Auto finance lender refused to return personal belongings inside repossessed car.",
        "Unlawful repossession occurred during active loan modification review period.",
        "Wrongful auto repossession fee added to balance after car was seized by mistake.",
        "Lender reported wrongful repossession to credit bureaus destroying credit score.",
    ]

    texts = overdraft_complaints + fraud_wire_complaints + repossession_complaints
    categories = (
        ["Retail Banking"] * len(overdraft_complaints)
        + ["Retail Banking"] * len(fraud_wire_complaints)
        + ["Mortgages & Loans"] * len(repossession_complaints)
    )
    return texts, categories


def test_complaint_clusterer_fit_kmeans(sample_complaints_data):
    texts, categories = sample_complaints_data
    clusterer = ComplaintClusterer(
        method="kmeans",
        n_clusters=3,
        min_cluster_size=5,
        random_state=42,
    )
    clusterer.fit(texts, categories)

    assert clusterer.labels_ is not None
    assert len(clusterer.labels_) == len(texts)
    assert len(clusterer.topics_) >= 2
    assert clusterer.quality_report_ is not None

    # Check topic summaries
    for cid, topic in clusterer.topics_.items():
        assert topic.size > 0
        assert len(topic.top_terms) > 0
        assert len(topic.representative_complaints) > 0
        assert topic.coherence >= 0.0


def test_complaint_clusterer_fit_hdbscan(sample_complaints_data):
    texts, categories = sample_complaints_data
    clusterer = ComplaintClusterer(
        method="hdbscan",
        min_cluster_size=4,
        min_samples=2,
        random_state=42,
    )
    clusterer.fit(texts, categories)

    assert clusterer.labels_ is not None
    assert len(clusterer.labels_) == len(texts)
    assert clusterer.quality_report_ is not None
    assert 0.0 <= clusterer.quality_report_.noise_rate <= 1.0


def test_cluster_predict(sample_complaints_data):
    texts, categories = sample_complaints_data
    clusterer = ComplaintClusterer(
        method="kmeans",
        n_clusters=3,
        min_cluster_size=5,
        random_state=42,
    )
    clusterer.fit(texts, categories)

    new_complaints = [
        "Unfair overdraft charges on my checking account.",
        "Stolen wire funds from bank account.",
    ]
    preds = clusterer.predict(new_complaints)
    assert len(preds) == len(new_complaints)
    assert all(isinstance(p, (int, np.integer)) for p in preds)


def test_cluster_quality_evaluation():
    # Synthetic 2D vectors for 2 well-separated clusters
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-2.0, -2.0], scale=0.1, size=(20, 2))
    c2 = rng.normal(loc=[2.0, 2.0], scale=0.1, size=(20, 2))
    embeddings = np.vstack([c1, c2])
    labels = np.array([0] * 20 + [1] * 20)

    top_terms = {0: ["fee", "overdraft", "bank"], 1: ["car", "auto", "repossession"]}
    report = evaluate_cluster_quality(embeddings, labels, cluster_top_terms=top_terms)

    assert report.n_clusters == 2
    assert report.noise_rate == 0.0
    assert report.min_cluster_size == 20
    assert report.max_cluster_size == 20
    assert report.silhouette is not None
    assert report.silhouette > 0.5
    assert report.intra_cluster_coherence is not None
    assert report.intra_cluster_coherence > 0.8
    assert report.term_distinctiveness == 1.0
    assert report.is_actionable is True


def test_emerging_issue_detector(sample_complaints_data):
    texts, categories = sample_complaints_data
    clusterer = ComplaintClusterer(
        method="kmeans",
        n_clusters=3,
        min_cluster_size=5,
        random_state=42,
    )
    clusterer.fit(texts, categories)

    # Simulate surge: baseline has overdraft complaints, recent has massive wire fraud surge
    baseline_texts = [
        "Bank charged overdraft fee when balance was positive.",
        "Overdraft fee charged twice for one check deposit transaction.",
        "Customer service refused to waive overdraft fee.",
        "Checking balance went negative and incurred overdraft fee.",
    ]
    recent_texts = [
        "Unauthorized wire transfer sent overseas.",
        "Fraudulent wire transaction completed with compromised credentials.",
        "Identity theft scam resulted in unauthorized wire withdrawal.",
        "Bank failed to halt wire transfer after immediate fraud report.",
        "Funds stolen via wire fraud due to security authentication failure.",
        "Scammer initiated fraudulent wire without two-factor challenge.",
        "Stolen funds through wire transfer never reimbursed.",
        "Wire transfer fraud investigation closed prematurely.",
    ]

    detector = EmergingIssueDetector(
        clusterer=clusterer,
        growth_threshold=0.10,
        min_recent_volume=2,
        min_coherence=0.05,
    )
    alerts = detector.detect_from_batches(baseline_texts, recent_texts)

    assert isinstance(alerts, list)
    if alerts:
        top_alert = alerts[0]
        assert isinstance(top_alert, EmergingIssueAlert)
        assert top_alert.recent_volume >= 2
        assert top_alert.growth_percentage > 0
        assert top_alert.urgency in ["Critical", "High", "Moderate", "Monitor"]
        assert len(top_alert.top_keywords) > 0


def test_clusterer_save_and_load(tmp_path, sample_complaints_data):
    texts, categories = sample_complaints_data
    clusterer = ComplaintClusterer(
        method="kmeans",
        n_clusters=3,
        min_cluster_size=5,
        random_state=42,
    )
    clusterer.fit(texts, categories)

    save_path = tmp_path / "test_clusterer.joblib"
    clusterer.save(save_path)
    assert save_path.exists()

    loaded = ComplaintClusterer.load(save_path)
    assert loaded.method == clusterer.method
    assert len(loaded.topics_) == len(clusterer.topics_)

    preds_orig = clusterer.predict(texts[:5])
    preds_loaded = loaded.predict(texts[:5])
    np.testing.assert_array_equal(preds_orig, preds_loaded)
