"""
Unit tests for multi-signal severity scoring engine.
"""

import pytest
from src.severity.scorer import SeverityScorer, score_complaint


def test_severity_benign_complaint():
    text = "I would like to inquire about the terms of my monthly account statement."
    res = score_complaint(text)
    assert res.score == 0.0
    assert res.tier == "Low"
    assert len(res.matched_signals) == 0


def test_severity_financial_harm_fraud():
    text = "Someone committed fraud on my account, wire fraud and stolen identity theft scam."
    res = score_complaint(text)
    assert "financial_harm" in res.matched_signals
    assert res.score >= 0.30
    assert any(b.group_name == "financial_harm" and b.is_triggered for b in res.signal_breakdown)


def test_severity_urgent_lockout():
    text = "My account is frozen and blocked, I cannot access my funds and this is an urgent emergency."
    res = score_complaint(text)
    assert "urgency" in res.matched_signals
    assert res.score >= 0.25


def test_severity_legal_regulatory():
    text = "I have hired an attorney to file a lawsuit in court for serious FCRA and CFPB violations."
    res = score_complaint(text)
    assert "legal_regulatory" in res.matched_signals
    assert res.score >= 0.20


def test_severity_critical_compound_risk():
    # Triggers financial harm (0.30) + urgency (0.25) + legal (0.20) = 0.75 -> Critical
    text = (
        "Unauthorized fraudulent transactions occurred. My account is frozen and locked out. "
        "I am contacting my lawyer to report CFPB violations immediately."
    )
    res = score_complaint(text)
    assert res.score >= 0.65
    assert res.tier == "Critical"
    assert len(res.matched_signals) >= 3


def test_severity_empty_and_null_handling():
    res_empty = score_complaint("")
    assert res_empty.score == 0.0
    assert res_empty.tier == "Low"

    res_none = score_complaint(None)
    assert res_none.score == 0.0
    assert res_none.tier == "Low"
