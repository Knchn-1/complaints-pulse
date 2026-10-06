"""
Unit tests for end-to-end ComplaintsPulsePipeline.
"""

import pytest
from src.pipeline import ComplaintsPulsePipeline


@pytest.fixture(scope="module")
def pipeline():
    return ComplaintsPulsePipeline()


def test_pipeline_out_of_domain(pipeline):
    res = pipeline.analyze("My Amazon package was delivered late and the pizza was cold.")
    assert res.is_valid_domain is False
    assert res.category == "Out of Domain"
    assert "Outside supported" in res.classification_status
    assert res.target_queue == "Unsupported Input Review"
    assert res.latency_ms > 0


def test_pipeline_valid_financial_triage(pipeline):
    complaint = (
        "I noticed an unauthorized wire transfer of $5,000 on my checking account statement. "
        "I immediately alerted customer service but the bank manager refused to refund the stolen funds under Reg E."
    )
    res = pipeline.analyze(complaint)
    assert res.is_valid_domain is True
    assert res.category in ["Retail Banking", "Credit Card"]
    assert res.severity_level in ["Medium", "High", "Critical"]
    assert res.risk_score >= 0.30
    assert "Fraud" in res.target_queue or "Operations" in res.target_queue or "Unit" in res.target_queue
    assert len(res.suggested_actions) > 0
    assert res.triage_rationale != ""
    assert isinstance(res.similar_complaints, list)
    assert res.latency_ms > 0


def test_pipeline_debt_collection_harassment(pipeline):
    complaint = (
        "A third-party debt collector is calling my employer ten times a day threatening wage garnishment "
        "and legal arrest for a disputed debt that I already paid in full years ago."
    )
    res = pipeline.analyze(complaint)
    assert res.is_valid_domain is True
    assert res.category == "Debt Collection"
    assert "Dispute" in res.sub_issue or "Harassment" in res.sub_issue or "Threat" in res.sub_issue
    assert res.priority in ["P1 - Critical", "P2 - High"]
