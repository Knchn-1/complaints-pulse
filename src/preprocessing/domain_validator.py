"""
ComplaintsPulse — Financial Domain Validation & Out-of-Domain (OOD) Protection

Ensures user narratives belong to the supported CFPB consumer financial complaint domain.
Rejects unrelated inputs (e.g. retail deliveries, food orders, hardware problems)
before running model inference, preventing spurious classifications.
"""

from dataclasses import dataclass
from typing import List, Optional, Set
import re

# Curated financial domain anchor terms representing standard CFPB product concepts
FINANCIAL_CORE_KEYWORDS = {
    # Banking & Accounts
    "account", "bank", "checking", "savings", "deposit", "withdrawal", "overdraft",
    "wire", "transfer", "atm", "branch", "teller", "routing", "check", "funds",
    "direct deposit", "statement", "nsf", "balance", "cleared",
    
    # Credit Card & Payments
    "credit card", "debit card", "cardholder", "prepaid card", "charge", "merchant", "billing",
    "apr", "interest", "cash back", "rewards", "transaction", "reversal", "annual fee",
    "credit limit", "fraud", "unauthorized", "stolen card", "payment",
    
    # Credit Reporting & Identity
    "credit report", "credit score", "equifax", "experian", "transunion", "bureau",
    "inquiry", "derogatory", "delinquent", "dispute", "disputed", "identity theft",
    "inaccurate", "reporting", "credit file", "tradeline", "fcra",
    
    # Debt Collection
    "debt", "collector", "collection", "agency", "owed", "balance owed", "creditor",
    "fdcpa", "harass", "harassment", "garnishment", "wage", "cease and desist",
    "validation letter", "settlement", "payoff", "past due",
    
    # Mortgages & Loans
    "mortgage", "loan", "lender", "escrow", "foreclosure", "refinance", "refi",
    "servicer", "principal", "amortization", "pmi", "closing costs", "modification",
    "title", "deed", "appraisal", "heloc", "auto loan", "student loan",
}

# Non-financial overt indicator categories to quickly identify out-of-domain inputs
NON_FINANCIAL_PATTERNS = [
    r"\b(package|parcel|ups|fedex|amazon delivery|usps tracking|shipping delayed|courier)\b",
    r"\b(pizza|burger|food order|restaurant|meal|grocery|chef|waiter|tasted)\b",
    r"\b(flight|airline|boarding pass|hotel room|luggage|vacation|airbnb)\b",
    r"\b(graphics card|gpu|cpu|ram|motherboard|software bug|windows update|video game|fan|clicking noises)\b",
    r"\b(plumber|electrician|roof leak|lawn mower|mechanic oil|water pipe)\b",
]


@dataclass
class DomainValidationResult:
    """Outcome of financial domain validation check."""
    is_in_domain: bool
    domain_score: float                # 0.0 to 1.0 confidence that input is financial
    matched_financial_terms: List[str]
    detected_unrelated_terms: List[str]
    rejection_reason: Optional[str] = None
    guidance: Optional[str] = None


class FinancialDomainValidator:
    """
    Validates whether an incoming complaint text falls within the consumer financial domain.
    """

    def __init__(
        self,
        min_domain_score: float = 0.15,
        min_word_count: int = 4,
    ):
        self.min_domain_score = min_domain_score
        self.min_word_count = min_word_count

    def validate(self, text: str) -> DomainValidationResult:
        """
        Check if text belongs to supported financial complaint domain.
        """
        if not text or not text.strip():
            return DomainValidationResult(
                is_in_domain=False,
                domain_score=0.0,
                matched_financial_terms=[],
                detected_unrelated_terms=[],
                rejection_reason="Empty or blank complaint narrative.",
                guidance="Please provide details describing the financial issue.",
            )

        lower_text = text.lower()
        words = re.findall(r"\b[a-z]{3,}\b", lower_text)

        if len(words) < self.min_word_count:
            return DomainValidationResult(
                is_in_domain=False,
                domain_score=0.0,
                matched_financial_terms=[],
                detected_unrelated_terms=[],
                rejection_reason="Narrative is too brief to assess financial context.",
                guidance="Please provide a more descriptive narrative explaining your financial dispute.",
            )

        # 1. Check for explicit non-financial patterns
        detected_unrelated = []
        for pat in NON_FINANCIAL_PATTERNS:
            matches = re.findall(pat, lower_text)
            if matches:
                detected_unrelated.extend(matches)

        # 2. Match financial terms (single words and multi-word phrases)
        matched_financial = []
        for kw in FINANCIAL_CORE_KEYWORDS:
            if " " in kw:
                if kw in lower_text:
                    matched_financial.append(kw)
            else:
                if re.search(r"\b" + re.escape(kw) + r"\b", lower_text):
                    matched_financial.append(kw)

        # Deduplicate while preserving order
        matched_financial = list(dict.fromkeys(matched_financial))
        detected_unrelated = list(dict.fromkeys(detected_unrelated))

        # Calculate domain score
        # Base credit from unique financial keywords found
        term_count = len(matched_financial)
        if term_count == 0:
            domain_score = 0.0
        elif term_count == 1:
            domain_score = 0.35 if not detected_unrelated else 0.15
        elif term_count == 2:
            domain_score = 0.65 if not detected_unrelated else 0.40
        else:
            domain_score = min(1.0, 0.70 + (term_count - 3) * 0.08)

        # Penalty if overt non-financial indicators are present without strong financial framing
        if detected_unrelated and term_count <= 1:
            domain_score = min(domain_score, 0.10)

        is_in_domain = domain_score >= self.min_domain_score and term_count >= 1

        rejection_reason = None
        guidance = None

        if not is_in_domain:
            if detected_unrelated:
                rejection_reason = (
                    f"Narrative appears related to non-financial domain: "
                    f"detected terms ({', '.join(detected_unrelated[:3])})."
                )
            else:
                rejection_reason = (
                    "No recognized consumer financial concepts (e.g. bank, card, loan, credit, payment) detected."
                )
            guidance = (
                "ComplaintsPulse supports consumer financial products (Credit Cards, Credit Reporting, "
                "Debt Collection, Mortgages & Loans, Retail Banking). Please submit a financial complaint."
            )

        return DomainValidationResult(
            is_in_domain=is_in_domain,
            domain_score=round(domain_score, 3),
            matched_financial_terms=matched_financial[:10],
            detected_unrelated_terms=detected_unrelated[:5],
            rejection_reason=rejection_reason,
            guidance=guidance,
        )
