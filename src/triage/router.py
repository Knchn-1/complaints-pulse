"""
ComplaintsPulse — Policy-Driven Triage & SLA Routing Engine

Converts model predictions, uncertainty signals, and severity evaluations into
concrete operational routing decisions with target queue assignments, SLA deadlines,
and escalation triggers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.config import TRIAGE_ACTIONS
from src.severity.scorer import SeverityResult


@dataclass
class TriageDecision:
    """Actionable operational routing recommendation."""
    priority_level: str          # "P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Routine"
    target_queue: str            # e.g., "Fraud Operations", "Fair Debt Compliance", etc.
    handling_tier: str           # "Tier 3 — Senior / Legal", "Tier 2 — Senior Agent", etc.
    sla_hours: int               # Guaranteed resolution SLA
    escalation_required: bool    # Needs immediate supervisor / legal escalation
    escalation_reason: Optional[str]
    suggested_actions: List[str] # Playbook action checklist for the case agent
    rationale: str               # Human-readable policy rationale for the routing decision


# Team / Queue mapping matrix based on category and risk domain
QUEUE_MAPPING = {
    "Credit Card": "Credit Card Disputes & Servicing",
    "Credit Reporting": "Credit Bureau Dispute Operations",
    "Debt Collection": "Collections Compliance & FDCPA Unit",
    "Mortgages & Loans": "Mortgage Servicing & Escrow Ops",
    "Retail Banking": "Deposit Accounts & Retail Banking Unit",
}

ESCALATION_QUEUES = {
    "fraud": "Fraud Investigation & Asset Recovery Unit",
    "legal": "Legal Affairs & Regulatory Response Team",
    "lockout": "High-Priority Account Access & Recovery Team",
    "manual_review": "Tier-2 General Triage / Model Quality Review",
}


class TriageRouter:
    """
    Evaluates classification outputs and severity evaluations against institutional policies.
    """

    def __init__(self, triage_config: Optional[Dict] = None):
        self.triage_config = triage_config or TRIAGE_ACTIONS

    def route(
        self,
        category: str,
        severity: SeverityResult,
        confidence: float,
        is_uncertain: bool = False,
    ) -> TriageDecision:
        """
        Evaluate full complaint context and determine operational handling.
        """
        # Determine base SLA and handling tier from severity
        tier_info = self.triage_config.get(
            severity.tier,
            dict(
                action="Standard handling",
                sla_hours=48,
                tier="Tier 1 — Standard Agent",
            ),
        )
        base_sla = tier_info["sla_hours"]
        handling_tier = tier_info["tier"]

        # 1. Check for Critical Risk Escalations
        has_fraud = "financial_harm" in severity.matched_signals
        has_legal = "legal_regulatory" in severity.matched_signals
        has_lockout = "urgency" in severity.matched_signals

        escalation_required = False
        escalation_reason = None
        target_queue = QUEUE_MAPPING.get(category, "General Customer Service")
        suggested_actions: List[str] = []

        if has_legal:
            escalation_required = True
            escalation_reason = "Regulatory agency (CFPB/FTC) or litigation threatened."
            target_queue = ESCALATION_QUEUES["legal"]
            handling_tier = "Tier 3 — Legal & Compliance Affairs"
            base_sla = min(base_sla, 24)
            suggested_actions.extend([
                "Preserve all transaction records and customer communications under litigation hold.",
                "Review notice against CFPB/FCRA/FDCPA compliance statutory timelines.",
                "Assign senior regulatory specialist to draft formalized written response.",
            ])

        elif has_fraud:
            escalation_required = True
            escalation_reason = "Suspected fraudulent activity or unauthorized identity exploitation."
            target_queue = ESCALATION_QUEUES["fraud"]
            handling_tier = "Tier 2 — Fraud Operations"
            base_sla = min(base_sla, 24)
            suggested_actions.extend([
                "Immediately verify recent transactions and initiate temporary security restriction if active.",
                "Issue affidavit of unauthorized transaction to customer.",
                "Cross-check suspicious beneficiary details against known internal watchlists.",
            ])

        elif has_lockout and severity.tier in ("Critical", "High"):
            escalation_required = True
            escalation_reason = "Consumer actively locked out of necessary funds or critical assets."
            target_queue = ESCALATION_QUEUES["lockout"]
            base_sla = min(base_sla, 12)
            suggested_actions.extend([
                "Perform expedited customer identity verification.",
                "Review reason for account suspension / security lock.",
                "Contact customer via verified phone line within 4 business hours.",
            ])

        # 2. Check for Ambiguous / Low-Confidence Model Predictions
        if is_uncertain or confidence < 0.45:
            suggested_actions.append(
                f"Model classification confidence is low ({confidence:.1%}). "
                "Agent should verify category tagging before proceeding."
            )
            if not escalation_required and severity.tier == "Low":
                target_queue = ESCALATION_QUEUES["manual_review"]

        # 3. Add Category-Specific Operational Playbooks if not already escalated
        if not suggested_actions:
            if category == "Credit Reporting":
                suggested_actions.extend([
                    "Retrieve automated dispute verification (AUD) history from bureaus.",
                    "Verify consumer documentation against current credit bureau tradeline.",
                    "Submit correction or confirmation within 30-day FCRA window.",
                ])
            elif category == "Debt Collection":
                suggested_actions.extend([
                    "Verify debt ownership and chain of title.",
                    "Confirm cease-and-desist or dispute status under FDCPA § 809.",
                    "Ensure validation notice was properly dispatched within statutory window.",
                ])
            elif category == "Mortgages & Loans":
                suggested_actions.extend([
                    "Inspect monthly escrow analysis and amortization schedule.",
                    "Verify payment posting dates and check for misapplied principal/interest.",
                ])
            elif category == "Credit Card":
                suggested_actions.extend([
                    "Review billing cycle dispute timeline under Fair Credit Billing Act (FCBA).",
                    "Evaluate provisional credit eligibility for disputed amount.",
                ])
            else:
                suggested_actions.extend([
                    "Review account fee schedule and recent deposit transactions.",
                    "Check customer relationship tenure for potential courtesy fee reversal.",
                ])

        # Priority Level Formatting
        priority_map = {
            "Critical": "P1 - Critical",
            "High": "P2 - High",
            "Medium": "P3 - Medium",
            "Low": "P4 - Routine",
        }
        priority_level = priority_map.get(severity.tier, "P3 - Medium")
        if escalation_required and priority_level in ("P3 - Medium", "P4 - Routine"):
            priority_level = "P2 - High"

        # Rationale string
        rationale_parts = [
            f"Classified as '{category}' with {confidence:.1%} confidence",
            f"Evaluated severity: '{severity.tier}' (score: {severity.score:.2f})",
        ]
        if escalation_reason:
            rationale_parts.append(f"Escalation trigger: {escalation_reason}")
        rationale_parts.append(f"Routed to '{target_queue}' with a {base_sla}h SLA")
        rationale = ". ".join(rationale_parts) + "."

        return TriageDecision(
            priority_level=priority_level,
            target_queue=target_queue,
            handling_tier=handling_tier,
            sla_hours=base_sla,
            escalation_required=escalation_required,
            escalation_reason=escalation_reason,
            suggested_actions=suggested_actions,
            rationale=rationale,
        )


_default_router = TriageRouter()


def route_complaint(
    category: str,
    severity: SeverityResult,
    confidence: float,
    is_uncertain: bool = False,
) -> TriageDecision:
    """Convenience helper for triage routing."""
    return _default_router.route(
        category=category,
        severity=severity,
        confidence=confidence,
        is_uncertain=is_uncertain,
    )
