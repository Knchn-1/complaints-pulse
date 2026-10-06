"""Similarity search module — TF-IDF and semantic retrieval over historical complaints."""

from src.similarity.searcher import (
    BaseSimilaritySearcher,
    TfidfSimilaritySearcher,
    DenseSimilaritySearcher,
    SimilarityMatch,
)
from src.similarity.evaluator import (
    evaluate_retrieval_performance,
    evaluate_paraphrase_challenge,
    PARAPHRASE_TEST_CASES,
)

__all__ = [
    "BaseSimilaritySearcher",
    "TfidfSimilaritySearcher",
    "DenseSimilaritySearcher",
    "SimilarityMatch",
    "evaluate_retrieval_performance",
    "evaluate_paraphrase_challenge",
    "PARAPHRASE_TEST_CASES",
]
