"""
ComplaintsPulse — Page 2: Complaint Analytics & Portfolio Intelligence

Aggregate insights computed across the CFPB customer complaint repository:
  - Total volume and department share
  - Sub-issue granular breakdown
  - Severity and risk score distribution
  - Cross-product risk exposure
  - Temporal complaint volume trends
"""

import json
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import PROJECT_ROOT

SUMMARY_FILE = PROJECT_ROOT / "data" / "analytics_summary.json"


@st.cache_data
def load_analytics_data():
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, "r") as f:
            return json.load(f)
    return {}


def render_analytics_page():
    st.markdown("### Complaint Portfolio Analytics & Macro Insights")
    st.caption("Empirical breakdown across 162,411 verified CFPB consumer financial complaint records.")

    data = load_analytics_data()
    if not data:
        st.error("Analytics summary data not found. Please run scripts/generate_analytics_data.py.")
        return

    # Top KPI Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total CFPB Complaints", f"{data.get('total_complaints', 162411):,}")
    with k2:
        top_cat = list(data.get("category_percentages", {}).keys())[0]
        top_pct = list(data.get("category_percentages", {}).values())[0]
        st.metric("Dominant Category", top_cat, f"{top_pct:.1f}% share")
    with k3:
        high_risk_pct = data.get("severity_percentages", {}).get("High", 0) + data.get("severity_percentages", {}).get("Critical", 0)
        st.metric("Elevated Risk Cases", f"{high_risk_pct:.1f}%", "High + Critical")
    with k4:
        st.metric("Median Risk Index", f"{data.get('median_risk_score', 0.15):.2f}", "Scale: 0.0 - 1.0")

    st.markdown("---")

    # Section 1: Department Distribution & Severity Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Complaint Volume by Financial Product")
        cat_counts = data.get("category_counts", {})
        labels = list(cat_counts.keys())
        values = list(cat_counts.values())

        fig_cat = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.45,
            marker=dict(colors=["#1e40af", "#3b82f6", "#60a5fa", "#93c5fd", "#cbd5e1"]),
            textinfo='label+percent',
        )])
        fig_cat.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.markdown("#### Severity Tier Distribution")
        sev_counts = data.get("severity_counts", {})
        sev_order = ["Critical", "High", "Medium", "Low"]
        sev_vals = [sev_counts.get(s, 0) for s in sev_order]
        sev_colors = ["#ef4444", "#f97316", "#eab308", "#10b981"]

        fig_sev = go.Figure(go.Bar(
            x=sev_vals,
            y=sev_order,
            orientation='h',
            marker_color=sev_colors,
            text=[f"{v:,} ({v / sum(sev_vals) * 100:.1f}%)" for v in sev_vals],
            textposition='inside',
        ))
        fig_sev.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Complaints"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    st.markdown("---")

    # Section 2: Sub-Issue Granular Breakdown by Department
    st.markdown("#### Granular Sub-Issue Breakdown")
    sub_issues_by_cat = data.get("sub_issues_by_category", {})
    categories = list(sub_issues_by_cat.keys())

    selected_category = st.selectbox(
        "Select Department / Category to inspect top sub-issues:",
        options=categories,
        index=0,
    )

    if selected_category in sub_issues_by_cat:
        sub_data = sub_issues_by_cat[selected_category]
        sub_labels = list(sub_data.keys())
        sub_counts = list(sub_data.values())

        fig_sub = go.Figure(go.Bar(
            x=sub_counts,
            y=sub_labels,
            orientation='h',
            marker_color="#2563eb",
            text=[f"{c:,} samples" for c in sub_counts],
            textposition='auto',
        ))
        fig_sub.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Sample Frequency"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    st.markdown("---")

    # Section 3: Monthly Volume Trends (Illustrative)
    st.markdown("#### Illustrative 12-Month Intake Distribution")
    st.caption("Note: Because the processed CFPB corpus does not contain historical date attributes, this annualized distribution illustrates how the analytics module visualizes cross-department volume trends.")
    temporal = data.get("temporal_trends", {})
    months = temporal.get("months", [])
    volumes = temporal.get("volumes", {})

    if months and volumes:
        fig_trend = go.Figure()
        colors = ["#1e40af", "#0284c7", "#f59e0b", "#10b981", "#8b5cf6"]
        for idx, (cat_name, monthly_vals) in enumerate(volumes.items()):
            fig_trend.add_trace(go.Scatter(
                x=months,
                y=monthly_vals,
                mode='lines+markers',
                name=cat_name,
                line=dict(color=colors[idx % len(colors)], width=2.5),
            ))

        fig_trend.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Month"),
            yaxis=dict(title="Monthly Complaint Volume"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Narrative Length Distribution Insight
    len_stats = data.get("narrative_length_stats", {})
    if len_stats:
        st.info(
            f"📊 **Narrative Length Profile:** Average narrative length is **{len_stats.get('mean_words', 0):.0f} words** "
            f"(median: **{len_stats.get('median_words', 0)} words**, 90th percentile: **{len_stats.get('p90_words', 0)} words**). "
            f"The high variance demonstrates why sublinear TF scaling and n-gram representations were necessary to prevent long narratives from dominating predictions."
        )
