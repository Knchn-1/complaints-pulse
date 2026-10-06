"""
ComplaintsPulse — FastAPI Request & Response Schemas (Pydantic v2)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ComplaintRequest(BaseModel):
    narrative: str = Field(
        ...,
        description="Raw customer complaint text to be analyzed and triaged.",
        examples=[
            "I noticed an unauthorized wire transfer of $5,000 on my checking account statement. "
            "I alerted customer service but the bank refused to refund the stolen funds under Reg E."
        ],
    )
    top_k_similar: Optional[int] = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of similar historical CFPB complaints to retrieve.",
    )


class DomainValidationRequest(BaseModel):
    narrative: str = Field(
        ...,
        description="Raw text to check for financial complaint domain relevance.",
    )


class DomainValidationResponse(BaseModel):
    is_in_domain: bool
    domain_score: float
    rejection_reason: Optional[str] = None
    guidance: Optional[str] = None
    matched_financial_terms: List[str] = []
    detected_unrelated_terms: List[str] = []


class SimilarSearchRequest(BaseModel):
    query: str = Field(..., description="Query narrative for historical precedent search.")
    top_k: int = Field(default=5, ge=1, le=20)


class SimilarMatchResponse(BaseModel):
    rank: int
    complaint_id: Optional[Any] = None
    category: str
    similarity_score: float
    text: str


class TriageResponse(BaseModel):
    # Input validation
    raw_text: str
    cleaned_text: str
    is_valid_domain: bool
    domain_validation: Dict[str, Any]

    # Classification & Sub-Issue
    category: str
    category_confidence: float
    all_probabilities: Dict[str, float]
    is_uncertain: bool
    classification_status: str
    sub_issue: str
    sub_issue_confidence: float
    sub_issue_description: str

    # Severity & Risk
    severity_level: str
    risk_score: float
    severity_signals: Any
    detected_keywords: List[str]
    is_critical: bool

    # Smart Triage & SLA
    priority: str
    target_queue: str
    handling_tier: str
    sla_hours: int
    escalation_required: bool
    escalation_reason: Optional[str]
    suggested_actions: List[str]
    triage_rationale: str

    # Explainability & Similar Cases
    explanation: Optional[Dict[str, Any]]
    similar_complaints: List[Dict[str, Any]]

    # Topic Cluster Context
    assigned_cluster_id: Optional[int]
    cluster_topic_terms: List[str]

    # Latency
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    similarity_index_loaded: bool
    clusterer_loaded: bool
    version: str
