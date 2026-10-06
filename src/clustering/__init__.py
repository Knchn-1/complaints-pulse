"""
ComplaintsPulse — Clustering & Emerging Issue Detection Module

Exports:
  - ComplaintClusterer: Unsupervised topic discovery (HDBSCAN / KMeans + SVD + c-TF-IDF)
  - TopicSummary: Container for cluster keywords, coherence, and exemplars
  - ClusterQualityReport, evaluate_cluster_quality: Multi-metric evaluation engine
  - EmergingIssueDetector, EmergingIssueAlert: Temporal spike & early warning monitor
"""

from src.clustering.evaluation import (
    ClusterQualityReport,
    evaluate_cluster_quality,
)
from src.clustering.clusterer import (
    ComplaintClusterer,
    TopicSummary,
)
from src.clustering.detector import (
    EmergingIssueDetector,
    EmergingIssueAlert,
)

__all__ = [
    "ComplaintClusterer",
    "TopicSummary",
    "ClusterQualityReport",
    "evaluate_cluster_quality",
    "EmergingIssueDetector",
    "EmergingIssueAlert",
]
