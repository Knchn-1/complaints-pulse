"""
Unit tests for policy-driven triage and operational SLA routing.
"""

import pytest
from src.severity.scorer import SeverityResult, SignalMatch
from src.triage.router import route_complaint, TriageRouter, QUEUE_MAPPING, ESCALATION_QUEUES


def make_dummy_severity(
    score: float,
    tier: str,
    matched_signals: list[str],
) -> SeverityResult:
    return SeverityResult(
        score=score,
        tier=tier,
        matched_signals=matched_signals,
        signal_breakdown=[],
        summary_explanation="test",
    )


def test_standard_routine_triage():
    sev = make_dummy_severity(score=0.1, tier="Low", matched_signals=[])
    decision = route_complaint(
        category="Credit Reporting",
        severity=sev,
        confidence=0.92,
        is_uncertain=False,
    )
    assert decision.priority_level == "P4 - Routine"
    assert decision.target_queue == QUEUE_MAPPING["Credit Reporting"]
    assert not decision.escalation_required
    assert decision.sla_hours == 72
    assert any("FCRA" in action for action in decision.suggested_actions)


def test_fraud_escalation():
    sev = make_dummy_severity(score=0.45, tier="High", matched_signals=["financial_harm"])
    decision = route_complaint(
        category="Credit Card",
        severity=sev,
        confidence=0.88,
        is_uncertain=False,
    )
    assert decision.escalation_required
    assert "Fraud" in decision.target_queue
    assert decision.priority_level == "P2 - High"
    assert decision.sla_hours <= 24
    assert any("affidavit" in action.lower() for action in decision.suggested_actions)


def test_legal_litigation_escalation():
    sev = make_dummy_severity(score=0.75, tier="Critical", matched_signals=["legal_regulatory", "financial_harm"])
    decision = route_complaint(
        category="Debt Collection",
        severity=sev,
        confidence=0.85,
        is_uncertain=False,
    )
    assert decision.escalation_required
    assert decision.priority_level == "P1 - Critical"
    assert "Legal" in decision.target_queue
    assert any("litigation hold" in action.lower() for action in decision.suggested_actions)


def test_lockout_urgency_escalation():
    sev = make_dummy_severity(score=0.50, tier="High", matched_signals=["urgency"])
    decision = route_complaint(
        category="Retail Banking",
        severity=sev,
        confidence=0.80,
        is_uncertain=False,
    )
    assert decision.escalation_required
    assert "Lockout" in decision.escalation_reason.lower() or "funds" in decision.escalation_reason.lower()
    assert decision.sla_hours <= 12


def test_uncertain_model_flagging():
    sev = make_dummy_severity(score=0.05, tier="Low", matched_signals=[])
    decision = route_complaint(
        category="Retail Banking",
        severity=sev,
        confidence=0.35,
        is_uncertain=True,
    )
    assert any("verify category" in action.lower() for action in decision.suggested_actions)
    assert decision.target_queue == ESCALATION_QUEUES["manual_review"]
