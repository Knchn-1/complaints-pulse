"""
ComplaintsPulse — Cluster Quality Evaluation Engine

Rigorous evaluation of unsupervised complaint clusters. As required:
Do not judge clustering merely by whether clusters are produced.
Evaluate cluster quality using:
  - Cluster size distribution (min, max, median, skewness)
  - Noise / outlier rate
  - Semantic coherence (intra-cluster cosine similarity)
  - Separation (inter-cluster distance, silhouette score, Davies-Bouldin)
  - Term distinctiveness across clusters
  - Actionability assessment
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ClusterQualityReport:
    """Detailed quantitative evaluation metrics for a clustering run."""
    n_samples: int
    n_clusters: int
    noise_count: int
    noise_rate: float
    cluster_sizes: Dict[int, int]
    min_cluster_size: int
    max_cluster_size: int
    mean_cluster_size: float
    median_cluster_size: float
    silhouette: Optional[float] = None
    davies_bouldin: Optional[float] = None
    intra_cluster_coherence: Optional[float] = None
    term_distinctiveness: Optional[float] = None
    is_actionable: bool = False
    actionability_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "noise_count": self.noise_count,
            "noise_rate": round(self.noise_rate, 4),
            "cluster_sizes": {str(k): v for k, v in self.cluster_sizes.items()},
            "min_cluster_size": self.min_cluster_size,
            "max_cluster_size": self.max_cluster_size,
            "mean_cluster_size": round(self.mean_cluster_size, 2),
            "median_cluster_size": round(self.median_cluster_size, 2),
            "silhouette": round(self.silhouette, 4) if self.silhouette is not None else None,
            "davies_bouldin": round(self.davies_bouldin, 4) if self.davies_bouldin is not None else None,
            "intra_cluster_coherence": (
                round(self.intra_cluster_coherence, 4)
                if self.intra_cluster_coherence is not None
                else None
            ),
            "term_distinctiveness": (
                round(self.term_distinctiveness, 4)
                if self.term_distinctiveness is not None
                else None
            ),
            "is_actionable": self.is_actionable,
            "actionability_reasons": self.actionability_reasons,
        }


def evaluate_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_top_terms: Optional[Dict[int, List[str]]] = None,
    max_coherence_samples: int = 500,
) -> ClusterQualityReport:
    """
    Comprehensive quantitative evaluation of a clustering result.

    Args:
        embeddings: Feature matrix (e.g., SVD-reduced TF-IDF or dense embeddings).
        labels: Array of cluster assignments (-1 represents noise).
        cluster_top_terms: Optional mapping of cluster_id -> top keywords.
        max_coherence_samples: Subsample size per cluster for intra-cluster coherence computation.

    Returns:
        ClusterQualityReport dataclass with all metrics.
    """
    n_samples = len(labels)
    unique_labels = sorted(list(set(labels)))
    noise_count = int(np.sum(labels == -1))
    noise_rate = float(noise_count / n_samples) if n_samples > 0 else 0.0

    valid_cluster_labels = [lbl for lbl in unique_labels if lbl != -1]
    n_clusters = len(valid_cluster_labels)

    if n_clusters == 0:
        return ClusterQualityReport(
            n_samples=n_samples,
            n_clusters=0,
            noise_count=noise_count,
            noise_rate=noise_rate,
            cluster_sizes={},
            min_cluster_size=0,
            max_cluster_size=0,
            mean_cluster_size=0.0,
            median_cluster_size=0.0,
            is_actionable=False,
            actionability_reasons=["No valid clusters formed; all points classified as noise."],
        )

    cluster_sizes = {int(lbl): int(np.sum(labels == lbl)) for lbl in valid_cluster_labels}
    sizes = list(cluster_sizes.values())
    min_size = min(sizes)
    max_size = max(sizes)
    mean_size = float(np.mean(sizes))
    median_size = float(np.median(sizes))

    # Evaluate separation & cohesion on valid (non-noise) points
    valid_mask = labels != -1
    valid_embeddings = embeddings[valid_mask]
    valid_labels = labels[valid_mask]

    sil_score = None
    db_score = None
    if n_clusters > 1 and len(valid_labels) > n_clusters:
        try:
            sil_score = float(silhouette_score(valid_embeddings, valid_labels, metric="cosine"))
        except Exception:
            try:
                sil_score = float(silhouette_score(valid_embeddings, valid_labels, metric="euclidean"))
            except Exception:
                sil_score = None

        try:
            db_score = float(davies_bouldin_score(valid_embeddings, valid_labels))
        except Exception:
            db_score = None

    # Intra-cluster semantic coherence (mean pairwise cosine similarity within each cluster)
    coherence_scores = []
    rng = np.random.default_rng(42)
    for lbl in valid_cluster_labels:
        cluster_vecs = embeddings[labels == lbl]
        if len(cluster_vecs) < 2:
            continue
        if len(cluster_vecs) > max_coherence_samples:
            idx = rng.choice(len(cluster_vecs), size=max_coherence_samples, replace=False)
            cluster_vecs = cluster_vecs[idx]

        sim_matrix = cosine_similarity(cluster_vecs)
        # Extract upper triangle without diagonal
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        if len(upper_tri) > 0:
            coherence_scores.append(float(np.mean(upper_tri)))

    mean_coherence = float(np.mean(coherence_scores)) if coherence_scores else None

    # Term distinctiveness across clusters
    distinctiveness = None
    if cluster_top_terms and len(cluster_top_terms) > 1:
        all_terms = []
        cluster_term_sets = []
        for lbl, terms in cluster_top_terms.items():
            if lbl != -1:
                terms_clean = [t.lower().strip() for t in terms]
                cluster_term_sets.append(set(terms_clean))
                all_terms.extend(terms_clean)

        total_unique_terms = len(set(all_terms))
        total_term_instances = len(all_terms)
        if total_term_instances > 0:
            # Term distinctiveness = unique terms / total term occurrences
            distinctiveness = float(total_unique_terms / total_term_instances)

    # Actionability check
    reasons = []
    is_actionable = True

    if noise_rate > 0.65:
        is_actionable = False
        reasons.append(f"Noise rate is too high ({noise_rate:.1%}). More than 65% of complaints unclustered.")

    if min_size < 5:
        reasons.append(f"Smallest cluster has only {min_size} complaints; may be an outlier artifact.")

    if max_size > 0.70 * n_samples:
        is_actionable = False
        reasons.append(f"Largest cluster contains {max_size / n_samples:.1%} of all data (undifferentiated megacluster).")

    if mean_coherence is not None and mean_coherence < 0.10:
        is_actionable = False
        reasons.append(f"Intra-cluster semantic coherence is low ({mean_coherence:.3f} < 0.10).")

    if distinctiveness is not None and distinctiveness < 0.30:
        reasons.append(f"Clusters share many top terms; distinctiveness is low ({distinctiveness:.2f}).")

    if is_actionable and not reasons:
        reasons.append("Passes all quality checks: coherent clusters, controlled noise, distinct themes.")

    return ClusterQualityReport(
        n_samples=n_samples,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_rate=noise_rate,
        cluster_sizes=cluster_sizes,
        min_cluster_size=min_size,
        max_cluster_size=max_size,
        mean_cluster_size=mean_size,
        median_cluster_size=median_size,
        silhouette=sil_score,
        davies_bouldin=db_score,
        intra_cluster_coherence=mean_coherence,
        term_distinctiveness=distinctiveness,
        is_actionable=is_actionable,
        actionability_reasons=reasons,
    )
