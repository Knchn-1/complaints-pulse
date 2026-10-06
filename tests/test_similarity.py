"""
Unit tests for TF-IDF and Dense similarity retrieval engines.
"""

import pytest
from src.similarity.evaluator import (
    evaluate_paraphrase_challenge,
    evaluate_retrieval_performance,
)
from src.similarity.searcher import (
    DenseSimilaritySearcher,
    TfidfSimilaritySearcher,
)


@pytest.fixture(scope="module")
def small_corpus():
    texts = [
        "Unauthorized fraud charge on credit card account stolen money",
        "Disputed incorrect debt collection phone calls demanding payment",
        "Credit reporting bureau failed to remove inaccurate collection record",
        "Mortgage interest rate escrow calculation dispute on loan",
        "Bank checking account fee charged unlawfully at branch",
    ] * 6
    categories = [
        "Credit Card",
        "Debt Collection",
        "Credit Reporting",
        "Mortgages & Loans",
        "Retail Banking",
    ] * 6
    return texts, categories


def test_tfidf_similarity_searcher(small_corpus, tmp_path):
    texts, categories = small_corpus
    searcher = TfidfSimilaritySearcher(max_features=500)
    searcher.fit(texts, categories)

    assert searcher.is_fitted
    matches = searcher.query("fraudulent credit card charge", top_k=3)
    assert len(matches) == 3
    assert matches[0].category == "Credit Card"
    assert matches[0].similarity_score > 0.0
    assert matches[0].rank == 1

    # Persistence test
    save_path = tmp_path / "tfidf_test.joblib"
    searcher.save(save_path)
    loaded = TfidfSimilaritySearcher.load(save_path)
    loaded_matches = loaded.query("fraudulent credit card charge", top_k=3)
    assert loaded_matches[0].category == "Credit Card"


def test_dense_similarity_searcher(small_corpus, tmp_path):
    texts, categories = small_corpus
    searcher = DenseSimilaritySearcher(model_name="all-MiniLM-L6-v2")
    searcher.fit(texts, categories, batch_size=16)

    assert searcher.is_fitted
    matches = searcher.query("unlawful bank fee on checking account", top_k=3)
    assert len(matches) == 3
    assert matches[0].category == "Retail Banking"
    assert matches[0].similarity_score > 0.0

    # Persistence test
    save_path = tmp_path / "dense_test.joblib"
    searcher.save(save_path)
    loaded = DenseSimilaritySearcher.load(save_path)
    loaded_matches = loaded.query("unlawful bank fee on checking account", top_k=3)
    assert loaded_matches[0].category == "Retail Banking"


def test_retrieval_evaluation(small_corpus):
    texts, categories = small_corpus
    searcher = TfidfSimilaritySearcher(max_features=500)
    searcher.fit(texts, categories)

    queries = ["credit card stolen", "debt collection call"]
    expected = ["Credit Card", "Debt Collection"]

    perf = evaluate_retrieval_performance(searcher, queries, expected, top_k=3)
    assert "precision_at_k" in perf
    assert "hit_rate_at_k" in perf
    assert perf["hit_rate_at_k"] == 1.0
    assert perf["precision_at_k"] > 0.5
