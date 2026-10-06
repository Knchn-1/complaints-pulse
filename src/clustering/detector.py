"""
ComplaintsPulse — Emerging Issue Detection & Early Warning Engine

Detects rapidly surging complaint clusters, temporal bursts, and latent failure modes:
  - Temporal volume comparison (Current Window vs Baseline Window)
  - Relative growth rate and velocity metrics
  - Multi-criteria actionability filtering (size, growth, coherence, noise exclusion)
  - Severity integration (pinpoints high-risk emerging spikes)
  - Human-interpretable alert generation with c-TF-IDF root-cause keywords and exemplars
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    EMERGING_GROWTH_THRESHOLD,
    EMERGING_MIN_VOLUME,
    EMERGING_MIN_COHERENCE,
)
from src.clustering.clusterer import ComplaintClusterer, TopicSummary
from src.severity.scorer import SeverityScorer


@dataclass
class EmergingIssueAlert:
    """An actionable alert for a surging complaint theme."""
    cluster_id: int
    theme_name: str
    dominant_category: str
    recent_volume: int
    baseline_volume: int
    growth_rate: float
    growth_percentage: float
    coherence: float
    avg_severity_score: float
    severity_level: str
    top_keywords: List[str]
    representative_complaints: List[str]
    urgency: str  # 'Critical', 'High', 'Moderate', 'Monitor'
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "theme_name": self.theme_name,
            "dominant_category": self.dominant_category,
            "recent_volume": self.recent_volume,
            "baseline_volume": self.baseline_volume,
            "growth_rate": round(self.growth_rate, 4),
            "growth_percentage": round(self.growth_percentage, 1),
            "coherence": round(self.coherence, 4),
            "avg_severity_score": round(self.avg_severity_score, 4),
            "severity_level": self.severity_level,
            "top_keywords": self.top_keywords,
            "representative_complaints": self.representative_complaints,
            "urgency": self.urgency,
            "detected_at": self.detected_at,
        }


class EmergingIssueDetector:
    """
    Early-warning monitor for identifying new or surging complaint issues.
    """

    def __init__(
        self,
        clusterer: ComplaintClusterer,
        severity_scorer: Optional[SeverityScorer] = None,
        growth_threshold: float = EMERGING_GROWTH_THRESHOLD,
        min_recent_volume: int = EMERGING_MIN_VOLUME,
        min_coherence: float = EMERGING_MIN_COHERENCE,
    ):
        """
        Args:
            clusterer: Pre-fitted ComplaintClusterer instance.
            severity_scorer: Optional SeverityScorer to evaluate risk profile of emerging issues.
            growth_threshold: Minimum fractional volume growth (e.g. 0.25 = +25%).
            min_recent_volume: Minimum complaints in recent period to qualify as an issue.
            min_coherence: Minimum semantic coherence score to avoid random noise clusters.
        """
        self.clusterer = clusterer
        self.severity_scorer = severity_scorer or SeverityScorer()
        self.growth_threshold = growth_threshold
        self.min_recent_volume = min_recent_volume
        self.min_coherence = min_coherence

    def detect_from_batches(
        self,
        baseline_texts: List[str],
        recent_texts: List[str],
        baseline_categories: Optional[List[str]] = None,
        recent_categories: Optional[List[str]] = None,
    ) -> List[EmergingIssueAlert]:
        """
        Compare baseline time period (e.g. prior month) vs recent period (e.g. current month)
        to identify emerging clusters with significant volume growth.

        Args:
            baseline_texts: Complaints from reference/baseline period.
            recent_texts: Complaints from current/recent monitoring period.
            baseline_categories: Optional category tags for baseline period.
            recent_categories: Optional category tags for recent period.

        Returns:
            List of EmergingIssueAlert sorted by urgency and growth rate.
        """
        if not self.clusterer.topics_:
            return []

        # Predict cluster assignments for both batches
        recent_labels = self.clusterer.predict(recent_texts)
        baseline_labels = self.clusterer.predict(baseline_texts)

        # Compute volume counts per cluster (ignoring noise label -1)
        recent_counts = pd.Series(recent_labels).value_counts().to_dict()
        baseline_counts = pd.Series(baseline_labels).value_counts().to_dict()

        # Adjust for total batch sizes to compare relative frequencies
        n_recent = max(len(recent_texts), 1)
        n_baseline = max(len(baseline_texts), 1)
        volume_scale = n_recent / n_baseline

        alerts: List[EmergingIssueAlert] = []

        for cluster_id, topic in self.clusterer.topics_.items():
            if cluster_id == -1:
                continue

            r_vol = int(recent_counts.get(cluster_id, 0))
            b_vol_raw = int(baseline_counts.get(cluster_id, 0))
            # Normalized baseline volume comparable to current period scale
            b_vol_norm = max(b_vol_raw * volume_scale, 0.5)

            # Growth rate calculation: (recent - baseline_norm) / baseline_norm
            growth_rate = (r_vol - b_vol_norm) / b_vol_norm
            growth_pct = growth_rate * 100.0

            # Quality and actionability filtering
            if r_vol < self.min_recent_volume:
                continue
            if growth_rate < self.growth_threshold:
                continue
            if topic.coherence < self.min_coherence:
                continue

            # Analyze severity for recent complaints in this cluster
            cluster_indices = [i for i, lbl in enumerate(recent_labels) if lbl == cluster_id]
            cluster_recent_texts = [recent_texts[i] for i in cluster_indices]

            severity_scores = []
            for t in cluster_recent_texts[:100]:  # sample up to 100 for speed
                sev_res = self.severity_scorer.score(t)
                severity_scores.append(sev_res.score)

            avg_sev = float(np.mean(severity_scores)) if severity_scores else 0.0
            if avg_sev >= 0.60:
                sev_level = "High"
            elif avg_sev >= 0.30:
                sev_level = "Medium"
            else:
                sev_level = "Low"

            # Determine urgency based on both growth rate and severity
            if sev_level == "High" and growth_pct >= 50.0:
                urgency = "Critical"
            elif sev_level == "High" or growth_pct >= 50.0:
                urgency = "High"
            elif growth_pct >= 25.0:
                urgency = "Moderate"
            else:
                urgency = "Monitor"

            # Formulate human-readable theme name from top keywords
            theme_keywords = topic.top_terms[:3]
            theme_name = " / ".join(theme_keywords).title() if theme_keywords else f"Theme #{cluster_id}"

            alert = EmergingIssueAlert(
                cluster_id=cluster_id,
                theme_name=theme_name,
                dominant_category=topic.dominant_category or "Unknown",
                recent_volume=r_vol,
                baseline_volume=b_vol_raw,
                growth_rate=growth_rate,
                growth_percentage=growth_pct,
                coherence=topic.coherence,
                avg_severity_score=avg_sev,
                severity_level=sev_level,
                top_keywords=topic.top_terms,
                representative_complaints=topic.representative_complaints,
                urgency=urgency,
            )
            alerts.append(alert)

        # Sort alerts: Critical first, then High, then highest growth
        urgency_order = {"Critical": 0, "High": 1, "Moderate": 2, "Monitor": 3}
        alerts.sort(key=lambda a: (urgency_order.get(a.urgency, 4), -a.growth_rate))

        return alerts

    def detect_from_chronological_split(
        self,
        texts: List[str],
        split_ratio: float = 0.5,
        categories: Optional[List[str]] = None,
    ) -> List[EmergingIssueAlert]:
        """
        Convenience method for datasets with sequential/chronological ordering:
        splits into earlier baseline period and later recent period.
        """
        n = len(texts)
        split_idx = int(n * (1.0 - split_ratio))
        baseline_texts = texts[:split_idx]
        recent_texts = texts[split_idx:]

        baseline_cats = categories[:split_idx] if categories else None
        recent_cats = categories[split_idx:] if categories else None

        return self.detect_from_batches(
            baseline_texts=baseline_texts,
            recent_texts=recent_texts,
            baseline_categories=baseline_cats,
            recent_categories=recent_cats,
        )
