"""
Unit tests for SubIssueClassifier.
"""

import pytest
from src.classification.sub_issue import SubIssueClassifier


@pytest.fixture
def classifier():
    return SubIssueClassifier()


def test_credit_card_sub_issues(classifier):
    fraud_text = "There are fraudulent unauthorized charges on my credit card that I did not authorize."
    res = classifier.classify(fraud_text, "Credit Card")
    assert "Unauthorized" in res.sub_issue
    assert res.confidence >= 0.60
    assert len(res.matched_signals) > 0

    fee_text = "I was charged an unexpected late fee and annual fee with a high interest rate."
    res2 = classifier.classify(fee_text, "Credit Card")
    assert "Interest, Fees" in res2.sub_issue


def test_credit_reporting_sub_issues(classifier):
    dispute_text = "I sent a dispute letter under FCRA but the bureau failed to remove the inaccurate tradeline."
    res = classifier.classify(dispute_text, "Credit Reporting")
    assert "Dispute" in res.sub_issue or "Tradeline" in res.sub_issue


def test_retail_banking_sub_issues(classifier):
    transfer_text = "Someone made an unauthorized wire transfer and drained my checking account."
    res = classifier.classify(transfer_text, "Retail Banking")
    assert "Unauthorized" in res.sub_issue or "Transfers" in res.sub_issue

    lockout_text = "My account is locked and my funds are frozen so I cannot access money."
    res2 = classifier.classify(lockout_text, "Retail Banking")
    assert "Lockout" in res2.sub_issue or "Fund Availability" in res2.sub_issue


def test_fallback_category(classifier):
    res = classifier.classify("Random text", "UnknownCategory")
    assert res.sub_issue == "General Financial Complaint"
