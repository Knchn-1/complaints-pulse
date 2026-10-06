"""Triage & routing module — SLA routing, policy-driven action assignment."""

from src.triage.router import (
    TriageRouter,
    TriageDecision,
    route_complaint,
    QUEUE_MAPPING,
    ESCALATION_QUEUES,
)

__all__ = [
    "TriageRouter",
    "TriageDecision",
    "route_complaint",
    "QUEUE_MAPPING",
    "ESCALATION_QUEUES",
]
