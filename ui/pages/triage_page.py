"""
ComplaintsPulse — Page 1: Complaint Triage & Live Intelligence

Interactive complaint investigation interface featuring:
  - Financial domain gating (rejects irrelevant submissions)
  - PII redaction preview
  - Calibrated multi-class classification
  - Sub-issue taxonomy mapping
  - Severity risk scoring with threat signal breakdown
  - Policy SLA queue routing & operational playbook
  - Contrastive linear feature attribution
  - Similar historical CFPB case retrieval
"""

import streamlit as st
import plotly.graph_objects as go
from src.pipeline import ComplaintsPulsePipeline, TriagePipelineResult

SAMPLE_COMPLAINTS = {
    "Select a pre-built case...": "",
    "🚨 Critical Wire Fraud (Retail Banking)": (
        "I noticed an unauthorized wire transfer of $7,500 sent from my checking account "
        "to an offshore beneficiary yesterday. I immediately alerted the branch manager, but "
        "the bank refused to initiate an emergency recall or credit my account under Regulation E. "
        "My account is now frozen and I am facing severe financial hardship."
    ),
    "⚖️ Debt Collection Harassment (Debt Collection)": (
        "A collection agency has been repeatedly calling my employer and family members at 6 AM, "
        "threatening legal arrest and wage garnishment for an old hospital bill that was discharged "
        "in bankruptcy three years ago. They refused to provide a debt validation notice under FDCPA."
    ),
    "📋 Credit Bureau Inaccuracy (Credit Reporting)": (
        "I submitted multiple written disputes to the credit bureau regarding an inaccurate late payment "
        "tradeline from an auto lender that was reported in error. More than 45 days have passed with zero "
        "investigation response, violating FCRA guidelines and depressing my credit score by 80 points."
    ),
    "💳 Credit Card Double Charge (Credit Card)": (
        "A merchant billed my credit card twice for a single transaction. When I opened a billing dispute, "
        "the card issuer closed the dispute without reviewing my purchase receipts and charged an unwarranted "
        "dispute penalty fee plus late interest charges."
    ),
    "🏠 Mortgage Escrow Miscalculation (Mortgages & Loans)": (
        "My loan servicer improperly adjusted my monthly mortgage escrow calculation, increasing my payment by "
        "$600 per month due to an erroneous property tax estimate. Despite sending proof of local tax assessment, "
        "they are threatening default."
    ),
    "⛔ Out-of-Domain Example (E-Commerce Delivery)": (
        "My Amazon Prime delivery was delayed by three days and when the box arrived, the ceramic plates "
        "were shattered. Customer support was rude and refused to issue a return shipping label for the shoes."
    ),
}


def render_triage_page(pipeline: ComplaintsPulsePipeline):
    st.markdown("### Complaint Triage & Operational Intelligence")
    st.caption("Live financial domain validation, classification, severity scoring, SLA routing, and historical case retrieval.")

    # Sample Selector
    col_sample, col_clear = st.columns([4, 1])
    with col_sample:
        selected_sample = st.selectbox(
            "Load an example complaint scenario:",
            options=list(SAMPLE_COMPLAINTS.keys()),
            index=0,
            key="sample_selector",
        )
    with col_clear:
        st.write("")
        st.write("")
        if st.button("Clear Input", use_container_width=True):
            st.session_state["complaint_input"] = ""
            st.rerun()

    # Pre-populate text if sample picked
    default_text = SAMPLE_COMPLAINTS.get(selected_sample, "")
    if default_text and ("last_sample" not in st.session_state or st.session_state["last_sample"] != selected_sample):
        st.session_state["complaint_input"] = default_text
        st.session_state["last_sample"] = selected_sample

    narrative_input = st.text_area(
        "Enter customer complaint narrative:",
        value=st.session_state.get("complaint_input", ""),
        height=140,
        placeholder="Type or paste customer complaint text here...",
        key="complaint_input",
    )

    col_btn, col_k = st.columns([3, 1])
    with col_btn:
        analyze_clicked = st.button("Run Intelligence Analysis", type="primary", use_container_width=True)
    with col_k:
        top_k = st.slider("Similar Cases", min_value=2, max_value=8, value=4)

    if not narrative_input.strip():
        if analyze_clicked:
            st.warning("Please enter a customer complaint text to analyze.")
        st.info("💡 **Quick Start:** Select a pre-built case from the dropdown above or paste a custom complaint.")
        return

    if analyze_clicked or "last_result" in st.session_state:
        # Run pipeline
        with st.spinner("Processing complaint through analytical pipeline..."):
            result: TriagePipelineResult = pipeline.analyze(narrative_input, top_k_similar=top_k)
            st.session_state["last_result"] = result

        res = st.session_state["last_result"]

        # Latency metric banner
        st.caption(f"⚡ Pipeline execution time: **{res.latency_ms:.1f} ms**")

        # ── Domain Gate Handling ──────────────────────────────────────────────
        if not res.is_valid_domain:
            st.error("### ⚠️ Submission Outside Financial Complaint Mandate")
            st.markdown(
                f"""
                <div class="cp-priority-box p1-critical">
                    <strong style="color: #b91c1c; font-size: 15px;">Out-of-Domain Rejection</strong><br>
                    <span><strong>Reason:</strong> {res.domain_validation.get('rejection_reason', 'Unsupported subject matter')}</span><br>
                    <span><strong>Guidance:</strong> {res.domain_validation.get('guidance', 'Redirect customer to merchant')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.metric("Domain Score", f"{res.domain_validation.get('domain_score', 0):.2f} / 1.0 (Threshold: 0.15)")
            with col_d2:
                unrelated = res.domain_validation.get('detected_unrelated_terms', [])
                st.write("**Detected Irrelevant Terms:**", ", ".join(unrelated) if unrelated else "None")

            st.info("ℹ️ **Why was this rejected?** The system incorporates strict Out-of-Domain protection to ensure non-financial requests (e.g., e-commerce, delivery, utilities) are not erroneously triaged into banking compliance workflows.")
            return

        # ── Domain Accepted: Main Results ─────────────────────────────────────
        st.markdown("---")

        # Row 1: Core Summary Badges
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Department / Product**")
            st.markdown(f"### {res.category}")
            st.caption(f"Confidence: **{res.category_confidence * 100:.1f}%**")
        with c2:
            st.markdown("**Severity Level**")
            badge_class = f"badge-{res.severity_level.lower()}"
            st.markdown(f'<span class="cp-badge {badge_class}" style="font-size: 15px; padding: 6px 12px;">{res.severity_level}</span>', unsafe_allow_html=True)
            st.caption(f"Risk Score: **{res.risk_score:.2f} / 1.0**")
        with c3:
            st.markdown("**Operational Priority**")
            priority_class = res.priority.split()[0].lower()
            st.markdown(f'<span class="cp-badge badge-critical" style="font-size: 14px;">{res.priority}</span>', unsafe_allow_html=True)
            st.caption(f"Target SLA: **{res.sla_hours} hours**")
        with c4:
            st.markdown("**Assigned Queue**")
            st.markdown(f"**{res.target_queue}**")
            st.caption(f"Handling: {res.handling_tier}")

        # Row 2: Sub-Issue & Operational Triage Card
        st.markdown("#### Operational Triage & Routing Directive")
        priority_css_class = "p1-critical" if "Critical" in res.priority else ("p2-high" if "High" in res.priority else ("p3-medium" if "Medium" in res.priority else "p4-routine"))
        
        st.markdown(
            f"""
            <div class="cp-priority-box {priority_css_class}">
                <div style="font-weight: 700; font-size: 15px; margin-bottom: 4px;">
                    🎯 Sub-Issue: <span style="color: #1e293b;">{res.sub_issue}</span>
                </div>
                <div style="color: #475569; font-size: 13.5px; margin-bottom: 8px;">
                    {res.sub_issue_description}
                </div>
                <div style="font-size: 13.5px; color: #334155;">
                    <strong>Triage Rationale:</strong> {res.triage_rationale}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested Actions Checklist
        if res.suggested_actions:
            st.markdown("**Mandated Playbook Action Items:**")
            for action in res.suggested_actions:
                st.markdown(f'<div class="action-item"><span class="action-check">✔</span> {action}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Row 3: Two Columns — Probability Distribution & Severity Risk Signals
        col_probs, col_sev = st.columns(2)

        with col_probs:
            st.markdown("#### Category Probabilities")
            if res.all_probabilities:
                cats = list(res.all_probabilities.keys())
                probs = [res.all_probabilities[c] * 100 for c in cats]
                colors = ["#2563eb" if c == res.category else "#94a3b8" for c in cats]

                fig_prob = go.Figure(go.Bar(
                    x=probs,
                    y=cats,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{p:.1f}%" for p in probs],
                    textposition='outside',
                ))
                fig_prob.update_layout(
                    height=240,
                    margin=dict(l=10, r=40, t=10, b=10),
                    xaxis=dict(title="Probability (%)", range=[0, 100]),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        with col_sev:
            st.markdown("#### Severity Risk Breakdown")
            st.markdown(f"**Heuristic Policy Risk Index (0.0 – 1.0):** `{res.risk_score:.2f}`")
            st.caption("Rule-based regulatory policy heuristic grounded in FCRA, FDCPA, and Reg E threat signals. (Not a trained ML prediction, as CFPB dataset lacks ground-truth severity annotations).")
            
            signals = res.severity_signals
            if signals:
                for sig_name, sig_info in signals.items():
                    label = sig_name.replace("_", " ").title()
                    weight = sig_info.get("weight", 0.0)
                    kw = sig_info.get("matched_keywords", [])
                    st.markdown(f"- **{label}** *(Weight: {weight:.2f})*: Found `{', '.join(kw)}`")
            else:
                st.caption("No critical severity trigger keywords detected. Standard routine profile.")

            if res.detected_keywords:
                st.write("**Trigger Keywords Detected:**", ", ".join(f"`{k}`" for k in res.detected_keywords[:10]))

        # Row 4: Explainability (Contrastive Feature Attribution)
        st.markdown("---")
        st.markdown("#### Explainability: Model Decision Drivers")
        if res.explanation and "top_features" in res.explanation:
            exp = res.explanation
            runner_up = exp.get("runner_up_category", "Other")
            st.caption(f"Linear feature attribution comparing predicted category **{res.category}** against runner-up **{runner_up}**.")

            top_feats = exp.get("top_features", [])
            pos_feats = [f for f in top_feats if f.get("weight", 0) > 0]
            neg_feats = [f for f in top_feats if f.get("weight", 0) < 0]

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.markdown("**Evidence Supporting Prediction:**")
                if pos_feats:
                    for f in pos_feats[:6]:
                        w = f.get("weight", 0)
                        term = f.get("feature", "")
                        st.markdown(f'<span class="attribution-pill-pos">{term} (+{w:.2f})</span>', unsafe_allow_html=True)
                else:
                    st.caption("No positive contrastive tokens found.")

            with col_exp2:
                st.markdown(f"**Evidence Favoring {runner_up}:**")
                if neg_feats:
                    for f in neg_feats[:6]:
                        w = f.get("weight", 0)
                        term = f.get("feature", "")
                        st.markdown(f'<span class="attribution-pill-neg">{term} ({w:.2f})</span>', unsafe_allow_html=True)
                else:
                    st.caption(f"No opposing tokens favoring {runner_up}.")

        # Row 5: Similar Historical Complaints
        st.markdown("---")
        st.markdown("#### Similar Historical Complaints (CFPB Precedents)")
        st.caption("Retrieved from CFPB historical resolution index using calibrated TF-IDF cosine similarity.")

        if res.similar_complaints:
            for idx, match in enumerate(res.similar_complaints, start=1):
                sim_pct = match.get("similarity_score", 0.0) * 100
                cat = match.get("category", "General")
                snippet = match.get("snippet", match.get("text", ""))[:280]

                st.markdown(
                    f"""
                    <div class="similar-card">
                        <div class="similar-meta">
                            <span><strong>Case #{idx}</strong> — Category: <span style="color: #2563eb;">{cat}</span></span>
                            <span>Semantic Similarity: <strong>{sim_pct:.1f}%</strong></span>
                        </div>
                        <div style="color: #334155; line-height: 1.45;">
                            "{snippet}..."
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No similar historical complaints retrieved.")
