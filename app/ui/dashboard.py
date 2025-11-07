"""
CodeSage — AI Code Reviewer (Enhanced Dashboard v3)
----------------------------------------------------
Modern Streamlit dashboard for CodeSage.
Features:
- File upload + code preview
- Run Static + AI analysis (L1, L2, L3)
- Interactive summary, metrics, and function feedback
- Download JSON / PDF reports
"""

import streamlit as st
import tempfile
import json
from pathlib import Path
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer, SimpleDocTemplate

# Ensure project root is on sys.path when run via `streamlit run app/ui/dashboard.py`
try:
    from app.ai_engine.llm_feedback import generate_feedback
    from app.core.report_generator import generate_code_report
    from app import config
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add '<repo>/codesage' to path
    from app.ai_engine.llm_feedback import generate_feedback
    from app.core.report_generator import generate_code_report
    from app import config

# -------------------------------------------------------
# Page Config & Theme
# -------------------------------------------------------
st.set_page_config(
    page_title="CodeSage — AI Code Reviewer",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Custom CSS (for better visuals)
# -------------------------------------------------------
st.markdown("""
    <style>
    /* Global layout */
    .main {
        background-color: #0d1117;
        color: #e6edf3;
    }
    h1, h2, h3, h4 {
        color: #00e0b0 !important;
    }
    .stCodeBlock {
        border-radius: 10px !important;
        background-color: #161b22 !important;
    }
    .css-1d391kg p {
        color: #c9d1d9 !important;
    }
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(90deg, #00e0b0, #008cff);
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #00ffc6, #00aaff);
    }
    .stDownloadButton>button {
        background: #1f6feb !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stDownloadButton>button:hover {
        background: #238636 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00ffc6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------
def make_json_bytes(report: dict) -> bytes:
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")

def make_pdf_bytes(report: dict, title: str = "CodeSage Report") -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    story.append(Paragraph(f"Generated: {ts}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    story.append(Paragraph(report.get("summary", "No summary available."), styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Function Feedback</b>", styles["Heading2"]))
    for f in report.get("function_feedback", []):
        story.append(Paragraph(f"- {f}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Recommendations</b>", styles["Heading2"]))
    for r in report.get("general_recommendations", []):
        story.append(Paragraph(f"- {r}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Score:</b> {report.get('score', 'N/A')}", styles["Normal"]))

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf

# -------------------------------------------------------
# UI — Header
# -------------------------------------------------------
st.title("🤖 CodeSage — AI Code Reviewer")
st.caption("AI-Powered Python Code Quality Analysis • Feature Engineering + LLM Review")

st.markdown("---")

uploaded = st.file_uploader("📤 Upload a Python file for review", type=["py"])

if uploaded:
    code_bytes = uploaded.read()
    try:
        code_text = code_bytes.decode("utf-8")
    except Exception:
        code_text = code_bytes.decode("latin-1")

    st.subheader("📜 Source Code Preview")
    st.code(code_text, language="python")

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / uploaded.name
    temp_path.write_text(code_text, encoding="utf-8")

    st.sidebar.header("⚙️ Controls")
    mode = st.sidebar.selectbox("Review Mode", ["AUTO", "L1", "L2", "L3"])
    st.sidebar.markdown("### ")
    if st.sidebar.button("🚀 Run CodeSage Review"):
        st.info("🔍 Running static analysis...")
        report = generate_code_report(temp_path)
        wrapped = {"summary": report, "details": {"ast_analysis": report.get("ast_analysis", [])}}
        st.success("✅ Static analysis complete!")

        st.info("🤖 Generating AI Feedback...")
        # Pass selected mode directly to the feedback generator (avoids stale globals)
        feedback = generate_feedback(wrapped, mode=mode)
        st.success("✅ AI feedback ready!")

        # -------------------------------------------------------
        # Display Results (Tabs)
        # -------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["🧠 Overview", "🔍 Function Feedback", "💡 Recommendations"])

        with tab1:
            st.markdown("### Code Review Summary")
            st.write(feedback.get("summary", "No summary available."))

            score = feedback.get("score", 0)
            st.markdown(f"### Code Quality Score: **{score}/100**")

            # Visual progress bar
            st.progress(min(score / 100, 1.0))

            # Basic metrics grid
            col1, col2, col3 = st.columns(3)
            col1.metric("Maintainability", f"{report.get('maintainability_index', 0)}")
            col2.metric("Avg Complexity", f"{report.get('avg_complexity', 0)}")
            col3.metric("Lint Issues", f"{report.get('lint_issues', 0)}")

        with tab2:
            st.markdown("### Function-Level Insights")
            for f in feedback.get("function_feedback", []):
                st.success(f"✅ {f}")

        with tab3:
            st.markdown("### General Recommendations")
            for r in feedback.get("general_recommendations", []):
                st.warning(f"💡 {r}")

        # -------------------------------------------------------
        # Export Section
        # -------------------------------------------------------
        st.markdown("---")
        st.subheader("📁 Export Your Report")

        json_bytes = make_json_bytes(feedback)
        pdf_bytes = make_pdf_bytes(feedback)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download JSON", data=json_bytes, file_name="codesage_report.json", mime="application/json")
        with c2:
            st.download_button("🧾 Download PDF", data=pdf_bytes, file_name="codesage_report.pdf", mime="application/pdf")

else:
    st.info("Upload a Python file to start your AI code review 🚀")
