"""
ComplaintsPulse — Financial Complaint Intelligence & Early-Warning Platform
Main Streamlit Application Entrypoint

Architecture Flow:
  Complaint
     ↓
  Input Validation + PII Redaction
     ↓
  Financial Domain Validation Gate
     ↓
  Complaint Understanding (Category, Sub-Issue, Calibrated Risk)
     ↓
  Smart Priority & Institutional SLA Routing
     ↓
  Similar Historical CFPB Case Retrieval
     ↓
  Emerging Issue Detection & Early Warning
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="ComplaintsPulse — Financial Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.styles import CUSTOM_CSS
from ui.pages.triage_page import render_triage_page
from ui.pages.analytics_page import render_analytics_page
from ui.pages.emerging_page import render_emerging_page
from ui.pages.model_page import render_model_page
from src.pipeline import ComplaintsPulsePipeline


@st.cache_resource
def get_pipeline() -> ComplaintsPulsePipeline:
    """Cached singleton pipeline instance across Streamlit reruns."""
    return ComplaintsPulsePipeline()


def main():
    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar Header & Navigation
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <div style="font-size: 20px; font-weight: 700; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                <span>🏦</span> ComplaintsPulse
            </div>
            <div style="font-size: 12px; color: #64748b; margin-top: 2px;">
                Financial Complaint Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_selection = st.sidebar.radio(
        "Navigation",
        options=[
            "🎯 Complaint Triage",
            "📊 Complaint Analytics",
            "🚨 Emerging Issues Radar",
            "🔬 Model & Data Science",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 12px; color: #64748b; line-height: 1.6;">
            <strong>System Health:</strong><br>
            • Classifier: <code>Calibrated Logistic</code><br>
            • Feature Eng: <code>Sublinear TF-IDF (1,2)</code><br>
            • Retrieval: <code>CFPB Precedent Index</code><br>
            • Surveillance: <code>c-TF-IDF Topic Burst</code><br>
            • PII Filter: <code>Active (Zero Leakage)</code><br>
            • Status: <span style="color: #16a34a; font-weight: bold;">Operational</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load Pipeline
    pipeline = get_pipeline()

    # Route to selected page
    if nav_selection == "🎯 Complaint Triage":
        render_triage_page(pipeline)
    elif nav_selection == "📊 Complaint Analytics":
        render_analytics_page()
    elif nav_selection == "🚨 Emerging Issues Radar":
        render_emerging_page()
    elif nav_selection == "🔬 Model & Data Science":
        render_model_page()


if __name__ == "__main__":
    main()
