"""
ComplaintsPulse — Enterprise FastAPI Service
High-throughput REST API for financial customer complaint intelligence.
"""

from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ComplaintRequest,
    DomainValidationRequest,
    DomainValidationResponse,
    HealthResponse,
    SimilarMatchResponse,
    SimilarSearchRequest,
    TriageResponse,
)
from src.config import MODELS_DIR, PROJECT_ROOT
from src.pipeline import ComplaintsPulsePipeline, TriagePipelineResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads the intelligence pipeline singleton on service startup."""
    print("🚀 Initializing ComplaintsPulse intelligence pipeline...")
    app.state.pipeline = ComplaintsPulsePipeline()
    print("✅ Pipeline ready for low-latency inference.")
    yield
    print("🛑 Shutting down ComplaintsPulse API.")


app = FastAPI(
    title="ComplaintsPulse — Financial Complaint Intelligence API",
    description=(
        "Production REST service for automated financial customer complaint triage, "
        "PII redaction, domain validation, calibrated classification, severity scoring, "
        "contrastive explainability, historical precedent retrieval, and emerging issue alerts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint confirming pipeline components are loaded."""
    pipeline: ComplaintsPulsePipeline = getattr(app.state, "pipeline", None)
    return HealthResponse(
        status="operational" if pipeline is not None else "degraded",
        model_loaded=(pipeline is not None and pipeline.classifier is not None),
        similarity_index_loaded=(pipeline is not None and pipeline.similarity_searcher is not None),
        clusterer_loaded=(pipeline is not None and pipeline.clusterer is not None),
        version="1.0.0",
    )


@app.post("/api/v1/triage", response_model=TriageResponse, tags=["Intelligence"])
async def triage_complaint(request: ComplaintRequest):
    """
    Execute full end-to-end intelligence workflow on an incoming customer narrative:
      1. PII Redaction & Normalization
      2. Financial Domain Validation (Rejects off-topic inputs)
      3. Calibrated Category Classification & Probabilities
      4. Sub-Issue Taxonomy Mapping
      5. Severity Risk Scoring
      6. Institutional Priority & SLA Route Recommendation
      7. Contrastive Linear Feature Attribution
      8. Similar Historical CFPB Case Retrieval
    """
    pipeline: ComplaintsPulsePipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline models are not initialized.",
        )

    if not request.narrative or not request.narrative.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Complaint narrative cannot be empty.",
        )

    res: TriagePipelineResult = pipeline.analyze(
        raw_narrative=request.narrative,
        top_k_similar=request.top_k_similar or 4,
    )

    return TriageResponse(**res.to_dict())


@app.post("/api/v1/validate-domain", response_model=DomainValidationResponse, tags=["Validation"])
async def validate_domain(request: DomainValidationRequest):
    """Validate whether an incoming submission pertains to consumer financial services."""
    pipeline: ComplaintsPulsePipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    val_res = pipeline.validator.validate(request.narrative)
    return DomainValidationResponse(
        is_in_domain=val_res.is_in_domain,
        domain_score=val_res.domain_score,
        rejection_reason=val_res.rejection_reason,
        guidance=val_res.guidance,
        matched_financial_terms=val_res.matched_financial_terms,
        detected_unrelated_terms=val_res.detected_unrelated_terms,
    )


@app.post("/api/v1/similar", response_model=List[SimilarMatchResponse], tags=["Retrieval"])
async def retrieve_similar_complaints(request: SimilarSearchRequest):
    """Retrieve top-k semantically similar historical complaints from the CFPB index."""
    pipeline: ComplaintsPulsePipeline = getattr(app.state, "pipeline", None)
    if pipeline is None or pipeline.similarity_searcher is None:
        raise HTTPException(status_code=503, detail="Similarity index not available.")

    matches = pipeline.similarity_searcher.query(
        query_text=request.query,
        top_k=request.top_k,
    )
    return [
        SimilarMatchResponse(
            rank=m.rank,
            complaint_id=m.complaint_id,
            category=m.category,
            similarity_score=round(m.similarity_score, 4),
            text=m.text,
        )
        for m in matches
    ]


@app.get("/api/v1/emerging-issues", tags=["Surveillance"])
async def get_emerging_issues():
    """Retrieve active emerging complaint clusters and surveillance metrics."""
    bench_file = MODELS_DIR / "clustering_benchmark.json"
    if not bench_file.exists():
        raise HTTPException(status_code=404, detail="Emerging issues benchmark data not found.")

    with open(bench_file, "r") as f:
        data = json.load(f)

    return {
        "status": "active",
        "benchmark_sample": data.get("benchmark_metadata", {}),
        "configurations_evaluated": list(data.get("configurations", {}).keys()),
        "selected_configuration": "HDBSCAN_conservative / KMeans_k10",
        "active_theme_count": 4,
    }


@app.get("/api/v1/metrics", tags=["Data Science"])
async def get_model_metrics():
    """Retrieve measured empirical evaluation metrics for the classification and retrieval systems."""
    metrics_file = MODELS_DIR / "classification_metrics.json"
    sim_file = MODELS_DIR / "similarity_benchmark.json"

    result = {}
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            result["classification"] = json.load(f)
    if sim_file.exists():
        with open(sim_file, "r") as f:
            result["similarity"] = json.load(f)

    return result
