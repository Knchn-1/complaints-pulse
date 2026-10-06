"""
ComplaintsPulse — Complaint Similarity Search Engines

Implements both:
1. TF-IDF Cosine Similarity Search (Lexical baseline)
2. Sentence-Transformers Dense Semantic Retrieval (Embedding-based)

Both share a common interface for indexing, searching, and artifact persistence.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.cleaner import clean_text

# Ensure SSL bypass for huggingface if needed
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Import torch early on Windows to guarantee c10.dll initializes cleanly
try:
    import torch
except Exception:
    pass


@dataclass
class SimilarityMatch:
    """Represents a single retrieved historical complaint match."""
    complaint_id: int
    text: str
    category: str
    similarity_score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complaint_id": self.complaint_id,
            "text": self.text,
            "category": self.category,
            "similarity_score": round(self.similarity_score, 4),
            "rank": self.rank,
        }


SimilarComplaintMatch = SimilarityMatch


class BaseSimilaritySearcher:
    """Base interface for similarity search engines."""
    
    def fit(
        self,
        texts: Union[pd.Series, List[str]],
        categories: Union[pd.Series, List[str]],
    ) -> "BaseSimilaritySearcher":
        raise NotImplementedError

    def query(self, query_text: str, top_k: int = 5) -> List[SimilarityMatch]:
        raise NotImplementedError

    def save(self, filepath: Union[str, Path]) -> Path:
        raise NotImplementedError

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseSimilaritySearcher":
        raise NotImplementedError


class TfidfSimilaritySearcher(BaseSimilaritySearcher):
    """
    TF-IDF Sparse Cosine Similarity Search Engine.
    Fast, lightweight, highly accurate for exact lexical and domain-term matching.
    """

    def __init__(
        self,
        max_features: int = 25_000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=2,
            max_df=0.95,
        )
        self.corpus_vectors = None
        self.corpus_texts: List[str] = []
        self.corpus_categories: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        texts: Union[pd.Series, List[str]],
        categories: Union[pd.Series, List[str]],
    ) -> "TfidfSimilaritySearcher":
        clean_corpus = [clean_text(t) for t in texts]
        self.corpus_vectors = self.vectorizer.fit_transform(clean_corpus)
        self.corpus_texts = list(texts)
        self.corpus_categories = list(categories)
        self.is_fitted = True
        return self

    def query(self, query_text: str, top_k: int = 5) -> List[SimilarityMatch]:
        if not self.is_fitted:
            raise ValueError("Search index is not fitted.")

        cleaned = clean_text(query_text)
        query_vec = self.vectorizer.transform([cleaned])
        
        # Sparse cosine similarity
        similarities = cosine_similarity(query_vec, self.corpus_vectors).flatten()
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        matches = [
            SimilarityMatch(
                complaint_id=int(idx),
                text=self.corpus_texts[idx],
                category=self.corpus_categories[idx],
                similarity_score=round(float(similarities[idx]), 4),
                rank=i + 1,
            )
            for i, idx in enumerate(top_indices)
        ]
        return matches

    def save(self, filepath: Union[str, Path]) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TfidfSimilaritySearcher":
        return joblib.load(filepath)


class DenseSimilaritySearcher(BaseSimilaritySearcher):
    """
    Dense Semantic Search Engine powered by SentenceTransformers.
    Captures semantic meaning and synonym variations beyond exact lexical terms.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.corpus_texts: List[str] = []
        self.corpus_categories: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        texts: Union[pd.Series, List[str]],
        categories: Union[pd.Series, List[str]],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> "DenseSimilaritySearcher":
        clean_corpus = [clean_text(t) for t in texts]
        embeddings = self.model.encode(
            clean_corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.corpus_embeddings = embeddings
        self.corpus_texts = list(texts)
        self.corpus_categories = list(categories)
        self.is_fitted = True
        return self

    def query(self, query_text: str, top_k: int = 5) -> List[SimilarityMatch]:
        if not self.is_fitted:
            raise ValueError("Dense search index is not fitted.")

        cleaned = clean_text(query_text)
        query_emb = self.model.encode(
            [cleaned],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Inner product with normalized embeddings equals cosine similarity
        similarities = np.dot(self.corpus_embeddings, query_emb.T).flatten()
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        matches = [
            SimilarityMatch(
                complaint_id=int(idx),
                text=self.corpus_texts[idx],
                category=self.corpus_categories[idx],
                similarity_score=round(float(similarities[idx]), 4),
                rank=i + 1,
            )
            for i, idx in enumerate(top_indices)
        ]
        return matches

    def save(self, filepath: Union[str, Path]) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store embeddings and corpus data, reload model on load
        state = {
            "model_name": self.model_name,
            "corpus_embeddings": self.corpus_embeddings,
            "corpus_texts": self.corpus_texts,
            "corpus_categories": self.corpus_categories,
        }
        joblib.dump(state, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DenseSimilaritySearcher":
        state = joblib.load(filepath)
        instance = cls(model_name=state["model_name"])
        instance.corpus_embeddings = state["corpus_embeddings"]
        instance.corpus_texts = state["corpus_texts"]
        instance.corpus_categories = state["corpus_categories"]
        instance.is_fitted = True
        return instance
