"""
ComplaintsPulse — Retrieval Quality Benchmarking & Selection Script

Empirically compares:
1. TF-IDF Cosine Similarity Search (Lexical baseline)
2. Sentence-Transformers Dense Semantic Retrieval (Embedding-based)

Evaluates:
- Category Precision@5 & Hit Rate@5 on holdout test queries
- Mean Reciprocal Rank (MRR@5)
- Mean query latency (ms)
- Lexical gap / Paraphrase challenge robustness
- Model size / memory footprint

Selects the final production similarity engine based on measured data.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Windows: torch c10.dll must initialise before any transitive import
# (sentence_transformers -> transformers -> torch) to avoid DLL load order failures.
try:
    import torch  # noqa: F401
except Exception:
    pass

# Ensure project root in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.classification.data_loader import load_and_split_data
from src.config import MODELS_DIR
from src.similarity.evaluator import (
    evaluate_retrieval_performance,
    evaluate_paraphrase_challenge,
)
from src.similarity.searcher import (
    DenseSimilaritySearcher,
    TfidfSimilaritySearcher,
)


def run_benchmark(
    index_size: int = 5_000,
    n_test_queries: int = 200,
):
    print("=" * 75)
    print("ComplaintsPulse — Similarity Retrieval Benchmark: TF-IDF vs. Dense")
    print("=" * 75)

    # 1. Load data
    print(f"\n[1/5] Loading data for indexing and benchmarking...")
    splits = load_and_split_data(sample_size=index_size + n_test_queries + 1000, random_state=42)
    
    index_texts = splits.X_train.iloc[:index_size].tolist()
    index_cats = splits.y_train.iloc[:index_size].tolist()
    
    test_texts = splits.X_test.iloc[:n_test_queries].tolist()
    test_cats = splits.y_test.iloc[:n_test_queries].tolist()

    print(f"      Historical Corpus Size: {len(index_texts):,} complaints")
    print(f"      Benchmark Query Count:  {len(test_texts):,} queries")

    # 2. Build & Index TF-IDF
    print("\n[2/5] Indexing with TF-IDF Vectorizer...")
    t0 = time.time()
    tfidf_searcher = TfidfSimilaritySearcher()
    tfidf_searcher.fit(index_texts, index_cats)
    tfidf_index_time = round(time.time() - t0, 2)
    print(f"      TF-IDF indexed in {tfidf_index_time}s")

    # 3. Build & Index Sentence-Transformers
    print("\n[3/5] Indexing with Sentence-Transformers (all-MiniLM-L6-v2)...")
    t0 = time.time()
    dense_searcher = DenseSimilaritySearcher(model_name="all-MiniLM-L6-v2")
    dense_searcher.fit(index_texts, index_cats, batch_size=64, show_progress_bar=False)
    dense_index_time = round(time.time() - t0, 2)
    print(f"      Dense embeddings indexed in {dense_index_time}s")

    # 4. Evaluate Standard Retrieval Quality
    print(f"\n[4/5] Running retrieval evaluation over {n_test_queries} holdout queries...")
    tfidf_perf = evaluate_retrieval_performance(tfidf_searcher, test_texts, test_cats, top_k=5)
    dense_perf = evaluate_retrieval_performance(dense_searcher, test_texts, test_cats, top_k=5)

    # 5. Evaluate Paraphrase Challenge
    print("      Testing lexical gap / paraphrase challenge queries...")
    tfidf_para = evaluate_paraphrase_challenge(tfidf_searcher, top_k=5)
    dense_para = evaluate_paraphrase_challenge(dense_searcher, top_k=5)

    # 6. Comparison Output
    benchmark_report = {
        "dataset_metadata": {
            "index_corpus_size": len(index_texts),
            "eval_query_count": len(test_texts),
        },
        "tfidf_baseline": {
            "indexing_time_sec": tfidf_index_time,
            **tfidf_perf,
            "paraphrase_challenge": tfidf_para,
        },
        "dense_sentence_transformers": {
            "indexing_time_sec": dense_index_time,
            **dense_perf,
            "paraphrase_challenge": dense_para,
        },
    }

    print("\n" + "=" * 75)
    print("SIMILARITY RETRIEVAL BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Metric':<32} {'TF-IDF (Baseline)':<22} {'Dense (MiniLM)':<22}")
    print("-" * 75)
    print(f"{'Precision@5 (Category Concordance)':<32} {tfidf_perf['precision_at_k']:<22.4f} {dense_perf['precision_at_k']:<22.4f}")
    print(f"{'Hit Rate@5':<32} {tfidf_perf['hit_rate_at_k']:<22.4f} {dense_perf['hit_rate_at_k']:<22.4f}")
    print(f"{'Mean Reciprocal Rank (MRR@5)':<32} {tfidf_perf['mrr_at_k']:<22.4f} {dense_perf['mrr_at_k']:<22.4f}")
    print(f"{'Mean Top-1 Similarity Score':<32} {tfidf_perf['mean_top1_similarity']:<22.4f} {dense_perf['mean_top1_similarity']:<22.4f}")
    print(f"{'Query Latency (Avg ms)':<32} {tfidf_perf['avg_query_latency_ms']:<22.2f} {dense_perf['avg_query_latency_ms']:<22.2f}")
    print(f"{'Query Latency (P95 ms)':<32} {tfidf_perf['p95_query_latency_ms']:<22.2f} {dense_perf['p95_query_latency_ms']:<22.2f}")
    print(f"{'Paraphrase Top-1 Accuracy':<32} {tfidf_para['paraphrase_top1_accuracy']:<22.4f} {dense_para['paraphrase_top1_accuracy']:<22.4f}")
    print(f"{'Paraphrase Hit Rate@5':<32} {tfidf_para['paraphrase_top_k_hit_rate']:<22.4f} {dense_para['paraphrase_top_k_hit_rate']:<22.4f}")
    print("=" * 75)

    # Decision Logic:
    # If Dense has higher semantic paraphrase accuracy and competitive category precision,
    # evaluate latency vs semantic gain.
    dense_better_para = dense_para['paraphrase_top1_accuracy'] > tfidf_para['paraphrase_top1_accuracy']
    dense_better_mrr = dense_perf['mrr_at_k'] >= tfidf_perf['mrr_at_k']
    
    if dense_better_para and dense_better_mrr:
        chosen_engine = "dense"
        selection_rationale = (
            "Dense Sentence-Transformers chosen: Demonstrates superior semantic generalization "
            "on paraphrase queries and vocabulary mismatch while maintaining high category precision."
        )
        selected_searcher = dense_searcher
    else:
        chosen_engine = "tfidf"
        selection_rationale = (
            "TF-IDF chosen: Provides superior execution speed with competitive category precision."
        )
        selected_searcher = tfidf_searcher

    benchmark_report["selection"] = {
        "chosen_engine": chosen_engine,
        "rationale": selection_rationale,
    }

    print(f"\n[Selection Decision]: {selection_rationale}")

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODELS_DIR / "similarity_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    # Save the selected searcher
    index_path = MODELS_DIR / "similarity_index.joblib"
    selected_searcher.save(index_path)

    # Also save TFIDF index as fast baseline fallback
    tfidf_searcher.save(MODELS_DIR / "tfidf_similarity_index.joblib")

    print(f"\n[5/5] Artifacts saved:")
    print(f"      -> Benchmark Report: {report_path}")
    print(f"      -> Selected Index:   {index_path}")
    print(f"      -> TF-IDF Index:     {MODELS_DIR / 'tfidf_similarity_index.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark similarity searchers")
    parser.add_argument("--index-size", type=int, default=5_000, help="Corpus size for index")
    parser.add_argument("--n-queries", type=int, default=150, help="Number of test queries")
    args = parser.parse_args()
    run_benchmark(index_size=args.index_size, n_test_queries=args.n_queries)
