"""
Unit tests for ComplaintsPulse FastAPI service.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.pipeline import ComplaintsPulsePipeline


@pytest.fixture(scope="module")
def client():
    # In TestClient, lifespan runs automatically with 'with' block or when state is set
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"


def test_triage_valid_complaint(client):
    payload = {
        "narrative": (
            "An unauthorized debit card charge of $3,200 was processed on my checking account. "
            "I alerted the bank immediately but customer support refused to reverse the transaction."
        ),
        "top_k_similar": 3,
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid_domain"] is True
    assert data["category"] in ["Retail Banking", "Credit Card"]
    assert data["severity_level"] in ["Medium", "High", "Critical"]
    assert data["risk_score"] > 0.0
    assert len(data["suggested_actions"]) > 0
    assert len(data["similar_complaints"]) <= 3
    assert data["latency_ms"] > 0


def test_triage_out_of_domain(client):
    payload = {
        "narrative": "My pizza arrived cold and the delivery driver was twenty minutes late with the order.",
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid_domain"] is False
    assert data["category"] == "Out of Domain"
    assert data["target_queue"] == "Unsupported Input Review"


def test_triage_empty_text(client):
    response = client.post("/api/v1/triage", json={"narrative": "   "})
    assert response.status_code == 400


def test_validate_domain_endpoint(client):
    resp_valid = client.post("/api/v1/validate-domain", json={"narrative": "Late fee on credit card statement"})
    assert resp_valid.status_code == 200
    assert resp_valid.json()["is_in_domain"] is True

    resp_invalid = client.post("/api/v1/validate-domain", json={"narrative": "Package delivered to wrong house by fedex"})
    assert resp_invalid.status_code == 200
    assert resp_invalid.json()["is_in_domain"] is False


def test_similar_endpoint(client):
    response = client.post(
        "/api/v1/similar",
        json={"query": "unauthorized credit card fee and interest dispute", "top_k": 3},
    )
    assert response.status_code == 200
    matches = response.json()
    assert isinstance(matches, list)
    assert len(matches) <= 3
    if matches:
        assert "category" in matches[0]
        assert "similarity_score" in matches[0]


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "classification" in data
    assert "similarity" in data
    assert "improved_calibrated" in data["classification"]
