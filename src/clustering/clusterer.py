"""
ComplaintsPulse — Complaint Topic Discovery & Cluster Engine

Features:
  - Scalable vectorization (TF-IDF + TruncatedSVD / LSA + Spherical Normalization)
  - HDBSCAN (density-based, automatic cluster count, noise separation)
  - MiniBatchKMeans / KMeans baseline option
  - c-TF-IDF (Class-based TF-IDF) for highly distinctive root-cause keyword extraction
  - Medoid & exemplar extraction for representative human-readable complaints
  - Model serialization and inference
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from src.config import (
    CLUSTER_MIN_SIZE,
    CLUSTER_MIN_SAMPLES,
    CLUSTER_SVD_DIMS,
    CLUSTER_N_TOP_TERMS,
    CLUSTER_N_EXEMPLARS,
    RANDOM_STATE,
)
from src.clustering.evaluation import ClusterQualityReport, evaluate_cluster_quality


@dataclass
class TopicSummary:
    """Summary information for a discovered complaint cluster."""
    cluster_id: int
    size: int
    prevalence_pct: float
    top_terms: List[str]
    representative_complaints: List[str]
    coherence: float
    dominant_category: Optional[str] = None
    category_distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "prevalence_pct": round(self.prevalence_pct, 2),
            "top_terms": self.top_terms,
            "representative_complaints": self.representative_complaints,
            "coherence": round(self.coherence, 4),
            "dominant_category": self.dominant_category,
            "category_distribution": {
                k: round(v, 4) for k, v in self.category_distribution.items()
            },
        }


class ComplaintClusterer:
    """
    Topic discovery and clustering engine for customer complaint narratives.
    """

    def __init__(
        self,
        method: str = "hdbscan",
        n_clusters: Optional[int] = 10,
        min_cluster_size: int = CLUSTER_MIN_SIZE,
        min_samples: int = CLUSTER_MIN_SAMPLES,
        svd_dims: int = CLUSTER_SVD_DIMS,
        n_top_terms: int = CLUSTER_N_TOP_TERMS,
        n_exemplars: int = CLUSTER_N_EXEMPLARS,
        random_state: int = RANDOM_STATE,
    ):
        """
        Args:
            method: 'hdbscan' or 'kmeans'.
            n_clusters: Number of clusters (used only for kmeans).
            min_cluster_size: Minimum points for a valid cluster (hdbscan).
            min_samples: Neighborhood density parameter (hdbscan).
            svd_dims: Dimensions to reduce TF-IDF space to.
            n_top_terms: Number of c-TF-IDF terms to extract per cluster.
            n_exemplars: Number of representative complaints to extract per cluster.
            random_state: Random seed for reproducibility.
        """
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.svd_dims = svd_dims
        self.n_top_terms = n_top_terms
        self.n_exemplars = n_exemplars
        self.random_state = random_state

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_df=0.85,
            stop_words="english",
        )
        self.svd = TruncatedSVD(n_components=self.svd_dims, random_state=self.random_state)
        self.normalizer = Normalizer(copy=False)

        self.cluster_model: Optional[BaseEstimator] = None
        self.labels_: Optional[np.ndarray] = None
        self.embeddings_: Optional[np.ndarray] = None
        self.topics_: Dict[int, TopicSummary] = {}
        self.quality_report_: Optional[ClusterQualityReport] = None
        self.cluster_centroids_: Dict[int, np.ndarray] = {}

    @property
    def is_fitted(self) -> bool:
        """Returns True if the clusterer has been fitted."""
        return self.cluster_model is not None

    def _extract_c_tfidf_terms(
        self, texts: List[str], labels: np.ndarray
    ) -> Dict[int, List[str]]:
        """
        Extract distinctive cluster terms using Class-based TF-IDF (c-TF-IDF).
        """
        unique_labels = sorted(list(set(labels)))
        valid_labels = [lbl for lbl in unique_labels if lbl != -1]

        if not valid_labels:
            return {}

        # Aggregate texts per cluster
        cluster_docs = []
        for lbl in valid_labels:
            doc = " ".join([texts[i] for i, l in enumerate(labels) if l == lbl])
            cluster_docs.append(doc)

        c_vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            max_features=5_000,
        )
        try:
            count_matrix = c_vectorizer.fit_transform(cluster_docs).toarray()
            words = c_vectorizer.get_feature_names_out()
        except Exception:
            return {lbl: [] for lbl in valid_labels}

        if count_matrix.shape[0] == 0 or count_matrix.shape[1] == 0:
            return {lbl: [] for lbl in valid_labels}

        # Term frequency per cluster: tf_{t, c}
        total_words_per_cluster = count_matrix.sum(axis=1, keepdims=True) + 1e-9
        tf = count_matrix / total_words_per_cluster

        # Global term frequency across all clusters: f_t
        f_t = count_matrix.sum(axis=0) + 1e-9
        avg_words = float(np.mean(total_words_per_cluster))

        # c-TF-IDF IDF: log(1 + avg_words / f_t)
        idf = np.log(1.0 + (avg_words / f_t))

        c_tfidf = tf * idf

        cluster_top_terms = {}
        for idx, lbl in enumerate(valid_labels):
            scores = c_tfidf[idx]
            top_indices = np.argsort(scores)[::-1][: self.n_top_terms]
            cluster_top_terms[lbl] = [str(words[i]) for i in top_indices if scores[i] > 0]

        return cluster_top_terms

    def fit(
        self,
        texts: Union[pd.Series, List[str]],
        categories: Optional[Union[pd.Series, List[str]]] = None,
    ) -> "ComplaintClusterer":
        """
        Fit vectorizer, SVD, and clustering model on complaint texts.
        """
        text_list = list(texts)
        cat_list = list(categories) if categories is not None else None
        n_samples = len(text_list)

        if n_samples < self.min_cluster_size:
            raise ValueError(
                f"Sample size {n_samples} is smaller than min_cluster_size {self.min_cluster_size}."
            )

        # 1. Feature extraction
        # Dynamically adjust min_df for small inputs
        adaptive_min_df = min(3, max(1, n_samples // 15))
        self.tfidf_vectorizer.set_params(min_df=adaptive_min_df)

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_list)
        n_features = tfidf_matrix.shape[1]
        max_possible_dims = min(n_features - 1, n_samples - 1)

        if max_possible_dims >= 2:
            effective_dims = min(self.svd_dims, max_possible_dims)
            self.svd = TruncatedSVD(n_components=effective_dims, random_state=self.random_state)
            reduced = self.svd.fit_transform(tfidf_matrix)
            self.embeddings_ = self.normalizer.fit_transform(reduced)
        else:
            self.svd = None
            self.embeddings_ = self.normalizer.fit_transform(tfidf_matrix.toarray())

        # 2. Fit clustering model
        if self.method == "hdbscan":
            self.cluster_model = HDBSCAN(
                min_cluster_size=min(self.min_cluster_size, max(3, n_samples // 8)),
                min_samples=min(self.min_samples, max(2, self.min_cluster_size // 3)),
                metric="euclidean",  # on normalized vectors, euclidean is monotonic with cosine
                cluster_selection_method="eom",
                copy=True,
            )
            self.labels_ = self.cluster_model.fit_predict(self.embeddings_)
        elif self.method == "kmeans":
            k = self.n_clusters or 10
            k = min(k, max(2, n_samples // 3))
            self.cluster_model = MiniBatchKMeans(
                n_clusters=k,
                random_state=self.random_state,
                batch_size=min(1024, n_samples),
                n_init=3,
            )
            self.labels_ = self.cluster_model.fit_predict(self.embeddings_)
        else:
            raise ValueError(f"Unsupported clustering method '{self.method}'. Choose 'hdbscan' or 'kmeans'.")

        # 3. Extract c-TF-IDF keywords
        cluster_terms = self._extract_c_tfidf_terms(text_list, self.labels_)

        # 4. Compute centroids and exemplars
        unique_labels = sorted(list(set(self.labels_)))
        valid_labels = [lbl for lbl in unique_labels if lbl != -1]

        self.cluster_centroids_ = {}
        self.topics_ = {}

        for lbl in valid_labels:
            indices = np.where(self.labels_ == lbl)[0]
            cluster_embeds = self.embeddings_[indices]

            # Centroid
            centroid = np.mean(cluster_embeds, axis=0, keepdims=True)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self.cluster_centroids_[lbl] = centroid[0]

            # Representative exemplars (closest to centroid)
            sims = cosine_similarity(cluster_embeds, centroid).ravel()
            exemplar_idx_order = np.argsort(sims)[::-1][: self.n_exemplars]
            exemplar_texts = [text_list[indices[i]] for i in exemplar_idx_order]

            # Coherence
            mean_coherence = float(np.mean(sims)) if len(sims) > 0 else 0.0

            # Product category distribution if provided
            dominant_cat = None
            cat_dist = {}
            if cat_list is not None:
                cluster_cats = [cat_list[i] for i in indices]
                cat_counts = pd.Series(cluster_cats).value_counts()
                dominant_cat = cat_counts.index[0]
                cat_dist = (cat_counts / len(cluster_cats)).to_dict()

            self.topics_[lbl] = TopicSummary(
                cluster_id=int(lbl),
                size=len(indices),
                prevalence_pct=(len(indices) / n_samples) * 100.0,
                top_terms=cluster_terms.get(lbl, []),
                representative_complaints=exemplar_texts,
                coherence=mean_coherence,
                dominant_category=dominant_cat,
                category_distribution=cat_dist,
            )

        # 5. Evaluate overall clustering quality
        self.quality_report_ = evaluate_cluster_quality(
            embeddings=self.embeddings_,
            labels=self.labels_,
            cluster_top_terms=cluster_terms,
        )

        return self

    def predict(self, texts: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        Assign new complaints to closest existing cluster centroids.
        Assigns -1 if similarity to nearest centroid is below threshold (< 0.25).
        """
        if not self.cluster_centroids_:
            return np.full(len(texts), -1)

        text_list = list(texts)
        tfidf = self.tfidf_vectorizer.transform(text_list)
        if self.svd is not None:
            reduced = self.svd.transform(tfidf)
            embeds = self.normalizer.transform(reduced)
        else:
            embeds = self.normalizer.transform(tfidf.toarray())

        labels = sorted(list(self.cluster_centroids_.keys()))
        centroids_matrix = np.vstack([self.cluster_centroids_[l] for l in labels])

        sims = cosine_similarity(embeds, centroids_matrix)
        best_cluster_idx = np.argmax(sims, axis=1)
        best_sims = np.max(sims, axis=1)

        assigned_labels = []
        for i, idx in enumerate(best_cluster_idx):
            if best_sims[i] < 0.25:
                assigned_labels.append(-1)
            else:
                assigned_labels.append(labels[idx])

        return np.array(assigned_labels)

    def get_topics(self) -> Dict[int, TopicSummary]:
        """Return all discovered topic summaries."""
        return self.topics_

    def save(self, output_path: Union[str, Path]) -> Path:
        """Serialize clusterer to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "ComplaintClusterer":
        """Load serialized clusterer."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Cluster model not found at {path}")
        return joblib.load(path)
