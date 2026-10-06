"""
ComplaintsPulse — Custom Design System & CSS Styling
Provides clean, modern, professional enterprise aesthetics for Streamlit.
"""

CUSTOM_CSS = """
<style>
/* Main container styling */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
    max-width: 1200px;
}

/* Header & Banner */
.cp-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 24px;
    color: #f8fafc;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.cp-header h1 {
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.cp-header p {
    font-size: 14px;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
}

/* Card Container */
.cp-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}

.cp-card-dark {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
    color: #f8fafc;
}

/* Badge System */
.cp-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-right: 6px;
}

.badge-critical {
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
.badge-high {
    background: #ffedd5;
    color: #9a3412;
    border: 1px solid #fed7aa;
}
.badge-medium {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}
.badge-low {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.badge-neutral {
    background: #f1f5f9;
    color: #334155;
    border: 1px solid #cbd5e1;
}

/* Priority & Queue Callout */
.cp-priority-box {
    border-left: 5px solid;
    border-radius: 4px 8px 8px 4px;
    padding: 14px 18px;
    margin: 12px 0;
    background: #f8fafc;
}
.p1-critical { border-left-color: #ef4444; background: #fff5f5; }
.p2-high { border-left-color: #f97316; background: #fffaf0; }
.p3-medium { border-left-color: #eab308; background: #fefce8; }
.p4-routine { border-left-color: #3b82f6; background: #eff6ff; }

/* Similar Complaint Card */
.similar-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 13px;
}
.similar-card:hover {
    border-color: #cbd5e1;
    background: #f1f5f9;
}
.similar-meta {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    font-size: 11.5px;
    color: #64748b;
    font-weight: 500;
}

/* Feature Attribution Pill */
.attribution-pill-pos {
    display: inline-block;
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
    border-radius: 14px;
    padding: 3px 10px;
    font-size: 12px;
    margin: 3px;
    font-weight: 500;
}
.attribution-pill-neg {
    display: inline-block;
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
    border-radius: 14px;
    padding: 3px 10px;
    font-size: 12px;
    margin: 3px;
    font-weight: 500;
}

/* Action item check */
.action-item {
    display: flex;
    align-items: flex-start;
    padding: 6px 0;
    font-size: 13.5px;
    color: #1e293b;
}
.action-check {
    color: #2563eb;
    margin-right: 8px;
    font-weight: bold;
}
</style>
"""
