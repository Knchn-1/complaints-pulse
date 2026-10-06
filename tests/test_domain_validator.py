"""
Unit tests for FinancialDomainValidator and Out-of-Domain (OOD) Protection.
"""

import pytest
from src.preprocessing.domain_validator import FinancialDomainValidator


@pytest.fixture
def validator():
    return FinancialDomainValidator()


def test_valid_financial_complaints(validator):
    valid_samples = [
        "Unauthorized wire transfer of 2000 dollars from my checking account at Wells Fargo.",
        "Equifax has failed to remove inaccurate collection tradeline after dispute letter.",
        "Debt collector keeps calling my mobile phone demanding payment for debt not owed.",
        "Mortgage servicer miscalculated property tax escrow payment causing artificial shortage.",
        "Credit card company doubled my interest rate without prior disclosure notice.",
    ]
    for sample in valid_samples:
        res = validator.validate(sample)
        assert res.is_in_domain is True, f"Failed for: {sample}"
        assert res.domain_score > 0.15
        assert len(res.matched_financial_terms) >= 1
        assert res.rejection_reason is None


def test_out_of_domain_rejections(validator):
    ood_samples = [
        "My Amazon package was delivered late and the cardboard box was crushed.",
        "The pepperoni pizza was completely cold and the delivery driver forgot the drinks.",
        "My flight was delayed by 6 hours at JFK airport and United lost my luggage.",
        "The graphics card fan is making loud clicking noises and Windows keeps crashing.",
        "My plumber did not show up to fix the leaking pipe in the bathroom.",
    ]
    for sample in ood_samples:
        res = validator.validate(sample)
        assert res.is_in_domain is False, f"Should have rejected: {sample}"
        assert res.rejection_reason is not None
        assert res.guidance is not None


def test_empty_and_short_inputs(validator):
    res_empty = validator.validate("")
    assert res_empty.is_in_domain is False
    assert "Empty" in res_empty.rejection_reason

    res_short = validator.validate("bad day")
    assert res_short.is_in_domain is False
    assert "too brief" in res_short.rejection_reason
