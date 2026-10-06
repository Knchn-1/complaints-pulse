"""
ComplaintsPulse — Financial Sub-Issue Taxonomy & Classifier

Maps complaints within a predicted CFPB product category to specific operational sub-issues.
Uses grounded CFPB taxonomy and pattern scoring to provide precise root-cause categorization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import re

# Domain-grounded CFPB Sub-Issue definitions per product category
SUB_ISSUES_BY_CATEGORY: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "Credit Card": {
        "Unauthorized Transactions & Fraud": {
            "keywords": ["unauthorized", "fraud", "stolen", "scam", "skimming", "identity theft", "compromised", "forged"],
            "description": "Customer reporting charges they did not authorize or fraudulent card access.",
        },
        "Billing Disputes & Overcharging": {
            "keywords": ["double charge", "charged twice", "billing error", "wrong amount", "refund missing", "reversal denied", "cancelled order"],
            "description": "Dispute over merchant charges, missing merchant credits, or calculation errors.",
        },
        "Credit Limit & Account Servicing": {
            "keywords": ["credit limit", "lower limit", "closed account", "account locked", "credit line", "decrease", "unilateral"],
            "description": "Unexplained credit line reductions, abrupt account closures, or online access issues.",
        },
        "Interest, Fees & APR Penalties": {
            "keywords": ["interest rate", "annual fee", "late fee", "overage fee", "finance charge", "apr", "penalty interest"],
            "description": "Disputed interest rates, promotional rate expirations, or recurring penalty fees.",
        },
        "General Servicing / Other": {
            "keywords": ["customer service", "agent", "phone call", "representative", "delay", "poor service"],
            "description": "General customer service and operational delivery friction.",
        },
    },
    "Credit Reporting": {
        "Inaccurate Tradeline & Account Info": {
            "keywords": ["inaccurate", "incorrect balance", "wrong account", "not mine", "never opened", "derogatory", "closed account reporting"],
            "description": "Inaccurate balances, payment history discrepancies, or accounts belonging to someone else.",
        },
        "Bureau Dispute Resolution Failure": {
            "keywords": ["dispute", "investigation", "reinvestigation", "fcra", "verified inaccurate", "failed to remove", "30 days", "frivolous"],
            "description": "Credit reporting agency failure to conduct reasonable investigation under FCRA.",
        },
        "Identity Theft & Fraudulent Inquiries": {
            "keywords": ["identity theft", "unauthorized inquiry", "hard inquiry", "fraud alert", "police report", "ftc report", "synthetic"],
            "description": "Fraudulent lines opened through identity theft or unauthorized credit checks.",
        },
        "Public Records & Bankruptcy Errors": {
            "keywords": ["bankruptcy", "public record", "judgment", "lien", "discharged debt", "court record"],
            "description": "Improperly reported discharged bankruptcies or outdated public records.",
        },
        "General Servicing / Other": {
            "keywords": ["credit score", "bureau", "transunion", "equifax", "experian", "monitoring fee"],
            "description": "General credit bureau communication or scoring inquiries.",
        },
    },
    "Debt Collection": {
        "Debt Not Owed / Disputed Amount": {
            "keywords": ["not my debt", "paid in full", "already paid", "disputed amount", "wrong person", "identity theft", "fraudulent debt"],
            "description": "Collection agency pursuing a debt that was already settled, paid, or not owed by consumer.",
        },
        "Harassment & FDCPA Prohibited Contact": {
            "keywords": ["harass", "repeated calls", "call at work", "family member", "threaten", "foul language", "fdcpa", "cease and desist"],
            "description": "Excessive calling, verbal abuse, third-party disclosure, or ignoring cease-communication notices.",
        },
        "Failure to Provide Debt Validation": {
            "keywords": ["validation letter", "verification", "proof of debt", "promissory note", "chain of title", "original creditor"],
            "description": "Collector failing to provide written verification of debt upon timely consumer dispute.",
        },
        "Threat of Legal Action / Wage Garnishment": {
            "keywords": ["wage garnishment", "sue", "lawsuit", "arrest", "sheriff", "court", "attorney", "freeze assets"],
            "description": "Unlawful threats of arrest, improper litigation threats, or illegal garnishment claims.",
        },
        "General Servicing / Other": {
            "keywords": ["collection agency", "collector", "interest rate", "added fees", "settlement"],
            "description": "General debt servicing negotiations and fees.",
        },
    },
    "Mortgages & Loans": {
        "Escrow Account & Tax/Insurance Dispute": {
            "keywords": ["escrow", "property tax", "hazard insurance", "pmi", "shortage", "escrow analysis", "surplus"],
            "description": "Mismanagement of escrow balances, late property tax disbursements, or force-placed insurance.",
        },
        "Payment Processing & Misallocation": {
            "keywords": ["payment posting", "misapplied", "principal payment", "late fee", "suspense account", "grace period"],
            "description": "Mortgage servicer holding payments in suspense or improperly applying principal curtailments.",
        },
        "Loan Modification, Refinance & Delay": {
            "keywords": ["loan modification", "refinance", "refi", "trial payment plan", "loss mitigation", "underwriting delay", "denial"],
            "description": "Prolonged loss mitigation processing, unrequested document re-submissions, or refi delays.",
        },
        "Foreclosure & Default Servicing": {
            "keywords": ["foreclosure", "dual tracking", "default notice", "acceleration", "auction", "eviction", "trustee sale"],
            "description": "Unlawful foreclosure actions, dual-tracking during active loss mitigation, or wrongful acceleration.",
        },
        "General Servicing / Other": {
            "keywords": ["closing costs", "apr", "interest rate", "payoff statement", "deed", "title"],
            "description": "General loan documentation, payoff statements, or title servicing disputes.",
        },
    },
    "Retail Banking": {
        "Unauthorized Transfers & Reg E Fraud": {
            "keywords": ["unauthorized wire", "unauthorized", "ach", "zelle", "reg e", "stolen funds", "hacked", "scam transfer", "wire fraud", "fraud"],
            "description": "Electronic fund transfers, wire fraud, or peer-to-peer unauthorized debits under Regulation E.",
        },
        "Account Lockout & Fund Availability": {
            "keywords": ["locked", "lockout", "frozen", "freeze", "hold", "access money", "cannot access", "restricted", "refusal", "unable to access"],
            "description": "Inaccessible deposits, frozen checking/savings accounts, or extended check holds under Reg CC.",
        },
        "Overdraft & Unanticipated Fees": {
            "keywords": ["overdraft fee", "nsf fee", "maintenance fee", "monthly fee", "reordering transactions", "hidden fee"],
            "description": "Disputed non-sufficient funds fees, abusive transaction posting order, or hidden maintenance charges.",
        },
        "ATM & Branch Transaction Errors": {
            "keywords": ["atm error", "cash deposit", "machine took money", "teller error", "deposit discrepancy", "cashier check"],
            "description": "ATM failure to credit deposited cash or teller deposit clerical mistakes.",
        },
        "General Servicing / Other": {
            "keywords": ["direct deposit", "routing number", "statement", "branch manager", "customer service"],
            "description": "General bank servicing inquiries and branch administration issues.",
        },
    },
}


@dataclass
class SubIssueResult:
    """Classified sub-issue with matched evidence."""
    sub_issue: str
    confidence: float
    description: str
    matched_signals: List[str]


class SubIssueClassifier:
    """
    Identifies the granular sub-issue for a complaint narrative within a confirmed product category.
    """

    def __init__(self, taxonomy: Optional[Dict] = None):
        self.taxonomy = taxonomy or SUB_ISSUES_BY_CATEGORY

    def classify(self, text: str, category: str) -> SubIssueResult:
        """
        Classifies the complaint narrative into the most relevant sub-issue.
        """
        if category not in self.taxonomy:
            return SubIssueResult(
                sub_issue="General Financial Complaint",
                confidence=0.50,
                description="Unclassified financial service issue.",
                matched_signals=[],
            )

        category_sub_issues = self.taxonomy[category]
        lower_text = text.lower()

        scores: Dict[str, Tuple[int, List[str]]] = {}
        for sub_name, details in category_sub_issues.items():
            if sub_name.startswith("General"):
                continue
            matched = []
            for kw in details["keywords"]:
                if kw in lower_text:
                    matched.append(kw)
            scores[sub_name] = (len(matched), matched)

        # Pick sub-issue with highest keyword evidence
        best_sub = "General Servicing / Other"
        best_count = 0
        best_matched = []

        for sub_name, (cnt, matched) in scores.items():
            if cnt > best_count:
                best_count = cnt
                best_sub = sub_name
                best_matched = matched

        if best_count > 0:
            confidence = min(0.95, 0.60 + best_count * 0.10)
            desc = category_sub_issues[best_sub]["description"]
        else:
            best_sub = "General Servicing / Other"
            confidence = 0.50
            desc = category_sub_issues.get(best_sub, {}).get("description", "General service issue.")
            best_matched = []

        return SubIssueResult(
            sub_issue=best_sub,
            confidence=round(confidence, 3),
            description=desc,
            matched_signals=best_matched,
        )
