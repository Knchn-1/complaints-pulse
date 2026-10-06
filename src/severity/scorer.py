"""
ComplaintsPulse — Multi-Signal Severity Scoring Engine

Computes a transparent, domain-grounded severity score for customer complaints.
Because CFPB data lacks ground-truth severity annotations, this engine uses a
weighted multi-signal rule matrix based on regulatory, financial, and consumer risk.

The scoring is fully explainable: it surfaces every matched signal, the matched
keywords, and the mathematical contribution to the final severity tier.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional
from src.config import SEVERITY_SIGNALS, SEVERITY_TIERS
from src.preprocessing.cleaner import clean_text


@dataclass
class SignalMatch:
    """Represents the match details for a single severity signal category."""
    group_name: str
    weight: float
    matched_keywords: List[str]
    is_triggered: bool
    contribution: float  # weight if triggered else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "weight": self.weight,
            "matched_keywords": self.matched_keywords,
            "is_triggered": self.is_triggered,
            "contribution": round(self.contribution, 4),
        }


@dataclass
class SeverityResult:
    """Complete explainable severity evaluation result."""
    score: float
    tier: str
    matched_signals: List[str]
    signal_breakdown: List[SignalMatch]
    summary_explanation: str

    @property
    def risk_score(self) -> float:
        return self.score

    @property
    def severity_level(self) -> str:
        return self.tier

    @property
    def is_critical(self) -> bool:
        return self.tier == "Critical"

    @property
    def detected_keywords(self) -> List[str]:
        kws = []
        for s in self.signal_breakdown:
            kws.extend(s.matched_keywords)
        return list(dict.fromkeys(kws))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "risk_score": round(self.score, 4),
            "tier": self.tier,
            "severity_level": self.tier,
            "is_critical": self.is_critical,
            "matched_signals": self.matched_signals,
            "signal_breakdown": [s.to_dict() for s in self.signal_breakdown],
            "detected_keywords": self.detected_keywords,
            "summary_explanation": self.summary_explanation,
        }



class SeverityScorer:
    """
    Transparent, rule-grounded severity scorer.
    
    Evaluates:
    - Financial Harm (weight 0.30)
    - Urgency & Account Lockout (weight 0.25)
    - Legal & Regulatory Action (weight 0.20)
    - Unresolved & Persistent Friction (weight 0.15)
    - Financial Magnitude (weight 0.10)
    """

    def __init__(
        self,
        signals: Optional[Dict] = None,
        tiers: Optional[List] = None,
    ):
        self.signals = signals or SEVERITY_SIGNALS
        self.tiers = tiers or SEVERITY_TIERS
        
        # Precompile keyword patterns for efficiency
        self._compiled_patterns = {}
        for group, conf in self.signals.items():
            patterns = [
                (kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
                for kw in conf["keywords"]
            ]
            self._compiled_patterns[group] = patterns

    def score(self, text: Optional[str]) -> SeverityResult:
        """
        Evaluate complaint text and produce an explainable severity score.
        """
        if not text or not isinstance(text, str):
            return SeverityResult(
                score=0.0,
                tier="Low",
                matched_signals=[],
                signal_breakdown=[],
                summary_explanation="No text provided. Defaulted to Low severity.",
            )

        cleaned = clean_text(text, lowercase=True)
        raw_score = 0.0
        breakdown: List[SignalMatch] = []
        matched_groups: List[str] = []

        for group, conf in self.signals.items():
            weight = conf["weight"]
            patterns = self._compiled_patterns[group]
            
            matched_kws = []
            for kw, regex in patterns:
                if regex.search(cleaned):
                    matched_kws.append(kw)

            is_triggered = len(matched_kws) > 0
            contribution = weight if is_triggered else 0.0
            raw_score += contribution

            if is_triggered:
                matched_groups.append(group)

            breakdown.append(
                SignalMatch(
                    group_name=group,
                    weight=weight,
                    matched_keywords=matched_kws,
                    is_triggered=is_triggered,
                    contribution=contribution,
                )
            )

        final_score = min(round(raw_score, 4), 1.0)
        tier = self._assign_tier(final_score)
        explanation = self._build_explanation(tier, final_score, breakdown)

        return SeverityResult(
            score=final_score,
            tier=tier,
            matched_signals=matched_groups,
            signal_breakdown=breakdown,
            summary_explanation=explanation,
        )

    def _assign_tier(self, score: float) -> str:
        for tier_name, lower, upper in self.tiers:
            if lower <= score < upper or (score >= 1.0 and upper >= 1.0):
                return tier_name
        return "Low"

    def _build_explanation(
        self, tier: str, score: float, breakdown: List[SignalMatch]
    ) -> str:
        triggered = [b for b in breakdown if b.is_triggered]
        if not triggered:
            return "No high-risk keywords detected. Routine handling."
        
        details = [
            f"{b.group_name.replace('_', ' ').title()} (weight {b.weight*100:.0f}%, matched: {', '.join(b.matched_keywords)})"
            for b in triggered
        ]
        return f"Assigned '{tier}' (score {score:.2f}) due to: {'; '.join(details)}."


_default_scorer = SeverityScorer()


def score_complaint(text: Optional[str]) -> SeverityResult:
    """Convenience functional interface for severity scoring."""
    return _default_scorer.score(text)
