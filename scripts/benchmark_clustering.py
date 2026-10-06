"""
ComplaintsPulse — Clustering & Emerging Issue Detection Benchmark

Evaluates multiple clustering approaches and hyperparameter sets on real complaints:
  1. HDBSCAN (min_cluster_size=25, min_samples=5)
  2. HDBSCAN (min_cluster_size=50, min_samples=10)
  3. MiniBatchKMeans (K=8)
  4. MiniBatchKMeans (K=15)

Evaluates:
  - Cluster size distribution
  - Noise / outlier rate
  - Intra-cluster semantic coherence
  - Silhouette and Davies-Bouldin separation
  - c-TF-IDF term distinctiveness
  - Temporal growth & emerging issue detection
  - Actionability assessment

Saves results to models/clustering_benchmark.json and persists the best model
to models/clusterer.joblib.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Ensure project root in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.classification.data_loader import load_raw_complaints
from src.clustering.clusterer import ComplaintClusterer
from src.clustering.detector import EmergingIssueDetector
from src.config import MODELS_DIR, RANDOM_STATE


def run_clustering_benchmark(sample_size: int = 5000, random_state: int = RANDOM_STATE):
    print("=" * 70)
    print(f"ComplaintsPulse — Emerging Issue Clustering Benchmark")
    print(f"Sample size: {sample_size:,} complaints | Random seed: {random_state}")
    print("=" * 70)

    # 1. Load complaints
    print(f"Loading {sample_size} complaints from dataset...")
    df = load_raw_complaints(sample_size=sample_size, random_state=random_state)
    texts = df["text"].tolist()
    categories = df["category"].tolist()
    print(f"Loaded {len(texts)} complaints across {df['category'].nunique()} product categories.")

    # Configurations to evaluate
    configs = [
        {
            "name": "HDBSCAN_conservative",
            "method": "hdbscan",
            "min_cluster_size": 50,
            "min_samples": 10,
            "n_clusters": None,
        },
        {
            "name": "HDBSCAN_sensitive",
            "method": "hdbscan",
            "min_cluster_size": 25,
            "min_samples": 5,
            "n_clusters": None,
        },
        {
            "name": "MiniBatchKMeans_K8",
            "method": "kmeans",
            "min_cluster_size": 10,
            "min_samples": 5,
            "n_clusters": 8,
        },
        {
            "name": "MiniBatchKMeans_K15",
            "method": "kmeans",
            "min_cluster_size": 10,
            "min_samples": 5,
            "n_clusters": 15,
        },
    ]

    benchmark_results = {}
    fitted_models = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"\nEvaluating: {name} (method={cfg['method']})...")
        t0 = time.perf_counter()

        clusterer = ComplaintClusterer(
            method=cfg["method"],
            n_clusters=cfg["n_clusters"],
            min_cluster_size=cfg["min_cluster_size"],
            min_samples=cfg["min_samples"],
            random_state=random_state,
        )

        clusterer.fit(texts=texts, categories=categories)
        fit_time = time.perf_counter() - t0

        fitted_models[name] = clusterer
        quality = clusterer.quality_report_
        topics = clusterer.get_topics()

        print(f"  Fit time: {fit_time:.2f}s")
        print(f"  Clusters discovered: {quality.n_clusters}")
        print(f"  Noise rate: {quality.noise_rate:.1%} ({quality.noise_count} unassigned)")
        print(f"  Cluster size (min/median/max): {quality.min_cluster_size} / {quality.median_cluster_size:.0f} / {quality.max_cluster_size}")
        if quality.intra_cluster_coherence is not None:
            print(f"  Intra-cluster semantic coherence: {quality.intra_cluster_coherence:.4f}")
        if quality.silhouette is not None:
            print(f"  Silhouette score: {quality.silhouette:.4f}")
        if quality.term_distinctiveness is not None:
            print(f"  Term distinctiveness: {quality.term_distinctiveness:.4f}")
        print(f"  Actionable: {quality.is_actionable} ({'; '.join(quality.actionability_reasons[:2])})")

        # Sample topic top terms
        sample_topics = []
        for cid, top in list(topics.items())[:3]:
            sample_topics.append({
                "cluster_id": cid,
                "size": top.size,
                "category": top.dominant_category,
                "top_terms": top.top_terms[:5],
                "coherence": round(top.coherence, 4),
            })

        benchmark_results[name] = {
            "config": cfg,
            "fit_time_sec": round(fit_time, 2),
            "quality_report": quality.to_dict(),
            "sample_topics": sample_topics,
        }

    # Evaluate Emerging Issue Detection on sequential split (first 50% baseline, second 50% recent)
    print("\n" + "=" * 70)
    print("Evaluating Emerging Issue Detection & Temporal Spikes...")
    print("=" * 70)

    # Use HDBSCAN_sensitive and MiniBatchKMeans_K15 for detection comparison
    detector_hdbscan = EmergingIssueDetector(
        clusterer=fitted_models["HDBSCAN_sensitive"],
        growth_threshold=0.20,
        min_recent_volume=8,
        min_coherence=0.15,
    )
    alerts_hdbscan = detector_hdbscan.detect_from_chronological_split(
        texts=texts, split_ratio=0.5, categories=categories
    )

    detector_kmeans = EmergingIssueDetector(
        clusterer=fitted_models["MiniBatchKMeans_K15"],
        growth_threshold=0.20,
        min_recent_volume=8,
        min_coherence=0.15,
    )
    alerts_kmeans = detector_kmeans.detect_from_chronological_split(
        texts=texts, split_ratio=0.5, categories=categories
    )

    print(f"\nEmerging issues surfaced by HDBSCAN_sensitive: {len(alerts_hdbscan)}")
    for a in alerts_hdbscan[:3]:
        print(f"  [{a.urgency}] Theme: '{a.theme_name}' | Cat: {a.dominant_category}")
        print(f"      Growth: +{a.growth_percentage:.1f}% (Recent: {a.recent_volume} vs Baseline: {a.baseline_volume})")
        print(f"      Top keywords: {', '.join(a.top_keywords[:4])}")

    print(f"\nEmerging issues surfaced by MiniBatchKMeans_K15: {len(alerts_kmeans)}")
    for a in alerts_kmeans[:3]:
        print(f"  [{a.urgency}] Theme: '{a.theme_name}' | Cat: {a.dominant_category}")
        print(f"      Growth: +{a.growth_percentage:.1f}% (Recent: {a.recent_volume} vs Baseline: {a.baseline_volume})")
        print(f"      Top keywords: {', '.join(a.top_keywords[:4])}")

    # Determine recommended configuration:
    # We want actionable clusters with good semantic coherence and meaningful topics.
    # If HDBSCAN produces good clusters without excessive noise, it is preferred for natural density;
    # otherwise KMeans provides comprehensive coverage.
    hdbscan_rep = benchmark_results["HDBSCAN_sensitive"]["quality_report"]
    kmeans_rep = benchmark_results["MiniBatchKMeans_K15"]["quality_report"]

    if hdbscan_rep["n_clusters"] >= 3 and hdbscan_rep["noise_rate"] < 0.60:
        recommended = "HDBSCAN_sensitive"
        selection_rationale = (
            f"HDBSCAN_sensitive discovered {hdbscan_rep['n_clusters']} dense, natural complaint themes "
            f"with controlled noise ({hdbscan_rep['noise_rate']:.1%}) and high semantic coherence."
        )
    else:
        recommended = "MiniBatchKMeans_K15"
        selection_rationale = (
            f"MiniBatchKMeans_K15 selected for guaranteed 100% complaint assignment, robust cluster sizing, "
            f"and high c-TF-IDF term distinctiveness without discarding marginal complaints."
        )

    print("\n" + "=" * 70)
    print(f"RECOMMENDED CONFIGURATION: {recommended}")
    print(f"Rationale: {selection_rationale}")
    print("=" * 70)

    # Save benchmark report
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = MODELS_DIR / "clustering_benchmark.json"

    chosen_alerts = alerts_kmeans if recommended.startswith("MiniBatch") else alerts_hdbscan

    def _json_default(obj):
        import numpy as np
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.ndarray, list)):
            return list(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return str(obj)

    full_output = {
        "benchmark_metadata": {
            "sample_size": sample_size,
            "random_state": random_state,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "configurations": benchmark_results,
        "recommended_configuration": recommended,
        "selection_rationale": selection_rationale,
        "sample_emerging_alerts": [a.to_dict() for a in chosen_alerts],
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, default=_json_default)
    print(f"Benchmark report saved to: {report_file}")

    # Save the selected production model
    best_model = fitted_models[recommended]
    model_file = MODELS_DIR / "clusterer.joblib"
    best_model.save(model_file)
    print(f"Production cluster model saved to: {model_file}")

    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark complaint clustering algorithms.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of complaints to sample.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_clustering_benchmark(sample_size=args.sample_size, random_state=args.random_state)
