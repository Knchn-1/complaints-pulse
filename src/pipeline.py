"""
ComplaintsPulse — Unified Intelligence Pipeline

Implements the end-to-end operational processing flow:
  1. Input Validation + PII Protection (Redacts SSN, phone, card numbers, email, etc.)
  2. Financial Domain Validation (Rejects out-of-domain / irrelevant submissions)
  3. Complaint Classification (Calibrated Logistic Regression on tuned n-gram TF-IDF)
  4. Sub-Issue Identification (Domain-grounded CFPB sub-issue taxonomy)
  5. Severity / Risk Assessment (Heuristic 0.0-1.0 risk scoring across 5 threat signals)
  6. Smart Triage & SLA Routing (Institutional queue routing, handling tier, playbook)
  7. Feature Explainability (Contrastive linear feature attribution against runner-up)
  8. Historical Similar Case Retrieval (Semantic & lexical search against known CFPB records)
  9. Cluster & Emerging Theme Assignment (HDBSCAN/KMeans topic assignment with c-TF-IDF terms)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from src.config import MODELS_DIR
from src.preprocessing.cleaner import clean_text
from src.preprocessing.domain_validator import (
    FinancialDomainValidator,
    DomainValidationResult,
)
from src.classification.model import ComplaintClassifier, PredictionResult
from src.classification.sub_issue import SubIssueClassifier, SubIssueResult
from src.severity.scorer import SeverityScorer, SeverityResult
from src.triage.router import TriageRouter, TriageDecision
from src.explainability.explainer import ComplaintExplainer, ExplanationResult
from src.similarity.searcher import (
    TfidfSimilaritySearcher,
    SimilarComplaintMatch,
)
from src.clustering.clusterer import ComplaintClusterer


@dataclass
class TriagePipelineResult:
    """Full operational intelligence payload for a customer complaint narrative."""
    # Input & Validation
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
    severity_signals: Dict[str, Any]
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
    
    # Explainability
    explanation: Optional[Dict[str, Any]]
    
    # Similar Cases
    similar_complaints: List[Dict[str, Any]]
    
    # Cluster & Topic Context
    assigned_cluster_id: Optional[int]
    cluster_topic_terms: List[str]
    
    # Processing latency
    latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "is_valid_domain": self.is_valid_domain,
            "domain_validation": self.domain_validation,
            "category": self.category,
            "category_confidence": self.category_confidence,
            "all_probabilities": self.all_probabilities,
            "is_uncertain": self.is_uncertain,
            "classification_status": self.classification_status,
            "sub_issue": self.sub_issue,
            "sub_issue_confidence": self.sub_issue_confidence,
            "sub_issue_description": self.sub_issue_description,
            "severity_level": self.severity_level,
            "risk_score": self.risk_score,
            "severity_signals": self.severity_signals,
            "detected_keywords": self.detected_keywords,
            "is_critical": self.is_critical,
            "priority": self.priority,
            "target_queue": self.target_queue,
            "handling_tier": self.handling_tier,
            "sla_hours": self.sla_hours,
            "escalation_required": self.escalation_required,
            "escalation_reason": self.escalation_reason,
            "suggested_actions": self.suggested_actions,
            "triage_rationale": self.triage_rationale,
            "explanation": self.explanation,
            "similar_complaints": self.similar_complaints,
            "assigned_cluster_id": self.assigned_cluster_id,
            "cluster_topic_terms": self.cluster_topic_terms,
            "latency_ms": self.latency_ms,
        }


class ComplaintsPulsePipeline:
    """
    Singleton service orchestrating all analytical and operational intelligence steps.
    """

    def __init__(
        self,
        classifier: Optional[ComplaintClassifier] = None,
        similarity_searcher: Optional[TfidfSimilaritySearcher] = None,
        clusterer: Optional[ComplaintClusterer] = None,
    ):
        self.validator = FinancialDomainValidator()
        self.sub_issue_clf = SubIssueClassifier()
        self.severity_scorer = SeverityScorer()
        self.triage_router = TriageRouter()

        # Load persisted classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            clf_path = MODELS_DIR / "classifier.joblib"
            if clf_path.exists():
                self.classifier = ComplaintClassifier.load(clf_path)
            else:
                self.classifier = None

        # Build explainer from loaded classifier
        if self.classifier is not None:
            self.explainer = ComplaintExplainer(self.classifier)
        else:
            self.explainer = None

        # Load similarity searcher
        if similarity_searcher is not None:
            self.similarity_searcher = similarity_searcher
        else:
            sim_path = MODELS_DIR / "similarity_index.joblib"
            if not sim_path.exists():
                sim_path = MODELS_DIR / "tfidf_similarity_index.joblib"
            if sim_path.exists():
                self.similarity_searcher = TfidfSimilaritySearcher.load(sim_path)
            else:
                self.similarity_searcher = None

        # Load clusterer
        if clusterer is not None:
            self.clusterer = clusterer
        else:
            cl_path = MODELS_DIR / "clusterer.joblib"
            if cl_path.exists():
                self.clusterer = ComplaintClusterer.load(cl_path)
            else:
                self.clusterer = None

    def analyze(self, raw_narrative: str, top_k_similar: int = 4) -> TriagePipelineResult:
        """
        Execute full intelligence workflow on an incoming complaint text.
        """
        t0 = time.perf_counter()

        # Step 1: Input Validation + PII Protection
        cleaned_text = clean_text(raw_narrative)

        # Step 2: Financial Domain Validation
        dom_val = self.validator.validate(raw_narrative)

        # Handle Out-of-Domain gracefully
        if not dom_val.is_in_domain:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return TriagePipelineResult(
                raw_text=raw_narrative,
                cleaned_text=cleaned_text,
                is_valid_domain=False,
                domain_validation={
                    "is_in_domain": False,
                    "domain_score": dom_val.domain_score,
                    "rejection_reason": dom_val.rejection_reason,
                    "guidance": dom_val.guidance,
                    "matched_financial_terms": dom_val.matched_financial_terms,
                    "detected_unrelated_terms": dom_val.detected_unrelated_terms,
                },
                category="Out of Domain",
                category_confidence=0.0,
                all_probabilities={},
                is_uncertain=True,
                classification_status="Outside supported financial complaint domain",
                sub_issue="Non-Financial / Unsupported Request",
                sub_issue_confidence=0.0,
                sub_issue_description="Submission does not match CFPB financial product areas.",
                severity_level="Low",
                risk_score=0.0,
                severity_signals={},
                detected_keywords=[],
                is_critical=False,
                priority="P4 - Routine",
                target_queue="Unsupported Input Review",
                handling_tier="Tier 1 — Rejection / Forwarding",
                sla_hours=72,
                escalation_required=False,
                escalation_reason=None,
                suggested_actions=[
                    "Notify customer that narrative falls outside consumer financial protection mandate",
                    "Offer redirection to appropriate merchant or vendor support channel",
                ],
                triage_rationale="Input flagged as out-of-domain; blocked from ML classification to prevent false triage.",
                explanation=None,
                similar_complaints=[],
                assigned_cluster_id=None,
                cluster_topic_terms=[],
                latency_ms=latency_ms,
            )

        # Step 3: Complaint Understanding — Classification
        if self.classifier is not None:
            pred_res: PredictionResult = self.classifier.predict_single(cleaned_text)
            category = pred_res.predicted_category
            confidence = pred_res.confidence
            probabilities = pred_res.probabilities
            is_uncertain = pred_res.is_uncertain
            clf_status = pred_res.status
        else:
            category = "Unclassified"
            confidence = 0.50
            probabilities = {}
            is_uncertain = True
            clf_status = "Model unavailable"

        # Step 4: Sub-Issue Identification
        sub_res: SubIssueResult = self.sub_issue_clf.classify(cleaned_text, category)

        # Step 5: Severity / Risk Assessment
        sev_res: SeverityResult = self.severity_scorer.score(cleaned_text)

        # Step 6: Smart Priority & Operational Routing
        triage_dec: TriageDecision = self.triage_router.route(
            category=category,
            severity=sev_res,
            confidence=confidence,
            is_uncertain=is_uncertain,
        )

        # Step 7: Feature Attribution Explainability
        explanation_dict = None
        if self.explainer is not None:
            try:
                exp_res: ExplanationResult = self.explainer.explain(cleaned_text, top_n_features=6)
                explanation_dict = exp_res.to_dict()
            except Exception:
                explanation_dict = None

        # Step 8: Similar Historical Complaint Retrieval
        similar_list = []
        if self.similarity_searcher is not None:
            try:
                matches: List[SimilarComplaintMatch] = self.similarity_searcher.query(
                    query_text=cleaned_text, top_k=top_k_similar
                )
                similar_list = [m.to_dict() for m in matches]
            except Exception:
                similar_list = []

        # Step 9: Topic Cluster Assignment
        assigned_cluster = None
        cluster_terms = []
        if self.clusterer is not None and self.clusterer.is_fitted:
            try:
                clusters = self.clusterer.predict([cleaned_text])
                assigned_cluster = int(clusters[0])
                if assigned_cluster >= 0:
                    topic_info = self.clusterer.get_topic_summary(assigned_cluster)
                    if topic_info:
                        cluster_terms = topic_info.top_terms[:6]
            except Exception:
                pass

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return TriagePipelineResult(
            raw_text=raw_narrative,
            cleaned_text=cleaned_text,
            is_valid_domain=True,
            domain_validation={
                "is_in_domain": True,
                "domain_score": dom_val.domain_score,
                "rejection_reason": None,
                "guidance": None,
                "matched_financial_terms": dom_val.matched_financial_terms,
                "detected_unrelated_terms": dom_val.detected_unrelated_terms,
            },
            category=category,
            category_confidence=confidence,
            all_probabilities=probabilities,
            is_uncertain=is_uncertain,
            classification_status=clf_status,
            sub_issue=sub_res.sub_issue,
            sub_issue_confidence=sub_res.confidence,
            sub_issue_description=sub_res.description,
            severity_level=sev_res.severity_level,
            risk_score=sev_res.risk_score,
            severity_signals=sev_res.matched_signals,
            detected_keywords=sev_res.detected_keywords,
            is_critical=sev_res.is_critical,
            priority=triage_dec.priority_level,
            target_queue=triage_dec.target_queue,
            handling_tier=triage_dec.handling_tier,
            sla_hours=triage_dec.sla_hours,
            escalation_required=triage_dec.escalation_required,
            escalation_reason=triage_dec.escalation_reason,
            suggested_actions=triage_dec.suggested_actions,
            triage_rationale=triage_dec.rationale,
            explanation=explanation_dict,
            similar_complaints=similar_list,
            assigned_cluster_id=assigned_cluster,
            cluster_topic_terms=cluster_terms,
            latency_ms=latency_ms,
        )
