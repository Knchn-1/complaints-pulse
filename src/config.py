"""
ComplaintsPulse — Central Configuration

All paths, model parameters, severity signals, triage rules,
and tuneable constants live here.  Business logic modules import
from this file so that every knob is in one place.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = PROJECT_ROOT / "complaints_processed.csv"

# ── Dataset column mapping ─────────────────────────────────────────────────
RAW_TEXT_COL = "narrative"
RAW_LABEL_COL = "product"

CATEGORY_MAP = {
    "credit_card":            "Credit Card",
    "credit_reporting":       "Credit Reporting",
    "debt_collection":        "Debt Collection",
    "mortgages_and_loans":    "Mortgages & Loans",
    "retail_banking":         "Retail Banking",
}
CATEGORIES = sorted(CATEGORY_MAP.values())

# ── Data splitting ─────────────────────────────────────────────────────────
TEST_RATIO = 0.15
VAL_RATIO  = 0.15          # fraction of the data *after* test is removed
RANDOM_STATE = 42

# ── Baseline TF-IDF + Logistic Regression ──────────────────────────────────
BASELINE_TFIDF = dict(max_features=5_000, ngram_range=(1, 1),
                      sublinear_tf=False)
BASELINE_CLF   = dict(max_iter=1_000, random_state=RANDOM_STATE,
                      C=1.0, solver="lbfgs")

# ── Improved TF-IDF + Logistic Regression ──────────────────────────────────
IMPROVED_TFIDF = dict(max_features=15_000, ngram_range=(1, 2),
                      sublinear_tf=True, min_df=3, max_df=0.95)
IMPROVED_CLF   = dict(max_iter=1_000, random_state=RANDOM_STATE,
                      class_weight="balanced", C=1.0, solver="lbfgs")

# ── Severity scoring  (transparent, content-based) ─────────────────────────
#
# Each signal group has a set of indicator keywords and a weight.
# Final severity = weighted sum of binary indicator matches, clipped to [0, 1].
# This is a heuristic score — it is NOT trained on ground-truth severity labels
# because the CFPB dataset does not provide severity annotations.
SEVERITY_SIGNALS = {
    "financial_harm": {
        "keywords": [
            "fraud", "unauthorized", "stolen", "identity theft", "scam",
            "forged", "forgery", "wire fraud", "phishing", "hacked",
            "compromised", "fraudulent", "counterfeit",
        ],
        "weight": 0.30,
    },
    "urgency": {
        "keywords": [
            "immediate", "emergency", "urgent", "locked out", "frozen",
            "cannot access", "blocked", "suspended", "closed without",
            "time sensitive", "deadline", "foreclosure", "repossession",
        ],
        "weight": 0.25,
    },
    "legal_regulatory": {
        "keywords": [
            "attorney", "lawyer", "lawsuit", "court", "violation", "arrest",
            "fcra", "fdcpa", "cfpb", "ftc", "legal", "legal action", "sue",
            "regulatory", "compliance", "fair credit", "cease and desist",
            "harass", "harassment", "unlawful", "garnishment",
        ],
        "weight": 0.20,
    },
    "unresolved_persistent": {
        "keywords": [
            "unresolved", "ignored", "multiple times", "months",
            "still not resolved", "no response", "repeatedly", "ongoing",
            "year", "several attempts", "numerous", "no resolution",
        ],
        "weight": 0.15,
    },
    "financial_magnitude": {
        "keywords": [
            "thousands", "large amount", "substantial", "significant loss",
            "life savings", "retirement", "mortgage payment", "foreclosure",
            "garnishment", "bankruptcy", "wage",
        ],
        "weight": 0.10,
    },
}

SEVERITY_TIERS = [
    # (label,    lower_bound, upper_bound)
    ("Critical", 0.65, 1.01),
    ("High",     0.40, 0.65),
    ("Medium",   0.20, 0.40),
    ("Low",      0.00, 0.20),
]

# ── Triage routing ─────────────────────────────────────────────────────────
TRIAGE_ACTIONS = {
    "Critical": dict(
        action="Immediate compliance / specialist review",
        sla_hours=4,
        tier="Tier 3 — Senior / Legal / Compliance",
    ),
    "High": dict(
        action="Priority handling by senior agent",
        sla_hours=24,
        tier="Tier 2 — Senior Agent",
    ),
    "Medium": dict(
        action="Standard priority handling",
        sla_hours=48,
        tier="Tier 1 — Standard Agent",
    ),
    "Low": dict(
        action="Standard handling / self-service eligible",
        sla_hours=72,
        tier="Tier 1 — Standard / Automated",
    ),
}

# ── Out-of-Domain detection ────────────────────────────────────────────────
OOD_PERCENTILE = 97   # training-distance percentile used as the threshold

# ── Similarity search ──────────────────────────────────────────────────────
SIMILARITY_TOP_K = 5
# Maximum number of training docs kept for the search index (keeps file size
# manageable while still providing representative results).
SIMILARITY_INDEX_SIZE = 25_000

# ── Emerging-issue clustering & early-warning ──────────────────────────────
CLUSTER_MIN_SIZE        = 30        # minimum cluster size for HDBSCAN
CLUSTER_MIN_SAMPLES     = 5         # neighborhood density threshold for HDBSCAN
CLUSTER_SVD_DIMS        = 50        # dimensionality after TruncatedSVD
CLUSTER_N_TOP_TERMS     = 8         # top distinctive terms per cluster (c-TF-IDF)
CLUSTER_N_EXEMPLARS     = 3         # representative complaints per cluster
EMERGING_GROWTH_THRESHOLD = 0.25    # minimum +25% volume surge to flag as emerging
EMERGING_MIN_VOLUME     = 10        # minimum complaints in current period to alert
EMERGING_MIN_COHERENCE  = 0.15      # minimum semantic coherence score for actionability
