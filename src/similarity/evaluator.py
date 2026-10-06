"""
ComplaintsPulse — Retrieval Quality Benchmarking Suite

Compares TF-IDF and Sentence-Transformer retrieval quality using:
- Precision@K (Category concordance)
- Hit Rate@K
- Mean Reciprocal Rank (MRR@K)
- Average similarity score
- Query latency (ms/query)
- Semantic paraphrase challenge queries (evaluating lexical gap handling)
"""

import time
from typing import Any, Dict, List, Tuple
import numpy as np
from src.similarity.searcher import BaseSimilaritySearcher


def evaluate_retrieval_performance(
    searcher: BaseSimilaritySearcher,
    query_texts: List[str],
    query_categories: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluates a similarity search engine over a set of labeled queries.
    """
    precisions = []
    hits = []
    reciprocal_ranks = []
    top1_sims = []
    latencies = []

    for query, expected_cat in zip(query_texts, query_categories):
        t0 = time.perf_counter()
        matches = searcher.query(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        if not matches:
            precisions.append(0.0)
            hits.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        # Precision@K: Fraction of top_k that share the query's category
        match_cats = [m.category for m in matches]
        correct_count = sum(1 for c in match_cats if c == expected_cat)
        precisions.append(correct_count / len(matches))

        # Hit@K: Was at least one correct category retrieved?
        hits.append(1.0 if correct_count > 0 else 0.0)

        # MRR@K: Reciprocal rank of first correct match
        rr = 0.0
        for rank, c in enumerate(match_cats, start=1):
            if c == expected_cat:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        top1_sims.append(matches[0].similarity_score)

    return {
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "hit_rate_at_k": round(float(np.mean(hits)), 4),
        "mrr_at_k": round(float(np.mean(reciprocal_ranks)), 4),
        "mean_top1_similarity": round(float(np.mean(top1_sims)), 4),
        "avg_query_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_query_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "queries_evaluated": len(query_texts),
        "top_k": top_k,
    }


# Paraphrase challenge pairs testing lexical variation and vocabulary mismatch
PARAPHRASE_TEST_CASES = [
    {
        "query": "The automated teller terminal retained my paper currency without crediting my balance",
        "category": "Retail Banking",
        "rationale": "ATM cash deposit failure using non-standard phrasing",
    },
    {
        "query": "Third-party agency is threatening wage seizure for a disputed medical bill",
        "category": "Debt Collection",
        "rationale": "Debt collection wage garnishment using synonym phrasing",
    },
    {
        "query": "Inaccurate derogatory notation on my credit profile from an unknown bureau",
        "category": "Credit Reporting",
        "rationale": "Credit reporting error described with formal vocabulary",
    },
    {
        "query": "Lender miscalculated annual percentage rate and property tax reserves",
        "category": "Mortgages & Loans",
        "rationale": "Mortgage escrow dispute without using the word escrow",
    },
    {
        "query": "Merchant charged plastic line of credit twice for single transaction",
        "category": "Credit Card",
        "rationale": "Credit card double-charge using colloquial phrasing",
    },
]


def evaluate_paraphrase_challenge(
    searcher: BaseSimilaritySearcher,
    test_cases: List[Dict] = PARAPHRASE_TEST_CASES,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Tests semantic generalization on queries with zero or minimal exact lexical overlap.
    """
    case_results = []
    correct_top1 = 0
    correct_top_k = 0

    for item in test_cases:
        query = item["query"]
        expected = item["category"]
        matches = searcher.query(query, top_k=top_k)
        
        top1_cat = matches[0].category if matches else None
        all_cats = [m.category for m in matches]
        
        is_top1 = (top1_cat == expected)
        is_in_topk = (expected in all_cats)
        
        if is_top1:
            correct_top1 += 1
        if is_in_topk:
            correct_top_k += 1

        case_results.append({
            "query": query,
            "expected_category": expected,
            "top1_retrieved_category": top1_cat,
            "top1_similarity": matches[0].similarity_score if matches else 0.0,
            "top1_match_snippet": matches[0].text[:120] if matches else "",
            "success_top1": is_top1,
            "success_top_k": is_in_topk,
        })

    n = len(test_cases)
    return {
        "paraphrase_top1_accuracy": round(correct_top1 / n, 4),
        "paraphrase_top_k_hit_rate": round(correct_top_k / n, 4),
        "case_details": case_results,
    }
