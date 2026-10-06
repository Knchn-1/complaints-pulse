"""Severity scoring module — multi-signal rule-based severity engine."""

from src.severity.scorer import (
    SeverityScorer,
    SeverityResult,
    SignalMatch,
    score_complaint,
)

__all__ = ["SeverityScorer", "SeverityResult", "SignalMatch", "score_complaint"]
