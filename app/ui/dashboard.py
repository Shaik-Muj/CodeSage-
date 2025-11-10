"""
CodeSage — AI Code Reviewer (Enhanced Dashboard v3)
"""

import streamlit as st
import tempfile
import json
import time
from pathlib import Path
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer, SimpleDocTemplate

# Add repo to path if running as script
try:
    from app.ai_engine.llm_feedback import generate_feedback, load_phi_model
    from app.core.report_generator import generate_code_report
    from app import config
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.ai_engine.llm_feedback import generate_feedback, load_phi_model
    from app.core.report_generator import generate_code_report
    from app import config

# Page config
st.set_page_config(page_title="CodeSage — AI Code Reviewer", layout="wide", page_icon="🤖", initial_sidebar_state="expanded")

# Minimal CSS (kept from your previous version)
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #e6edf3; }
    h1,h2,h3,h4 { color: #00e0b0 !important; }
    .stCodeBlock { border-radius: 10px !important; background-color: #161b22 !important; }
    .stButton>button { border-radius: 8px; background: linear-gradient(90deg,#00e0b0,#008cff); color:white; font-weight:600; }
    .stDownloadButton>button { background: #1f6feb !important; color: white !important; }
    div[data-testid="stMetricValue"] { color: #00ffc6 !important; }
    </style>
""", unsafe_allow_html=True)

# Utility exporters
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

st.title("🤖 CodeSage — AI Code Reviewer")
st.caption("AI-Powered Python Code Quality Analysis • Local LLM Feedback")

st.markdown("---")
uploaded = st.file_uploader("📤 Upload a Python file for review", type=["py"])

# Streamlit cache to warm/load the model once (only used for L2/L3)
@st.cache_resource
def get_cached_model():
    # load_phi_model will do a safe load (quantized if possible)
    # We just call it to ensure model is initialized and cached by Streamlit
    try:
        load_phi_model()
        return True
    except Exception as e:
        print("[CACHE] model warm-up failed:", e)
        return False

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
        # 1) Static analysis
        st.info("🔍 Running static analysis...")
        t0 = time.perf_counter()
        report = generate_code_report(temp_path)
        t1 = time.perf_counter()
        static_time = t1 - t0
        st.success(f"✅ Static analysis complete ({static_time:.1f}s)")

        # 2) Warm model if needed (L2/L3/AUTO when GPU available)
        if mode.upper() in ("AUTO", "L2", "L3"):
            st.info("🔁 Warming model (cached)...")
            get_cached_model()

        # 3) AI feedback
        st.info("🤖 Generating AI Feedback...")
        t0 = time.perf_counter()
        feedback = generate_feedback(report, mode=mode)
        t1 = time.perf_counter()
        ai_time = t1 - t0
        st.success(f"✅ AI feedback ready ({ai_time:.1f}s)")

        # Display results
        tab1, tab2, tab3 = st.tabs(["🧠 Overview", "🔍 Function Feedback", "💡 Recommendations"])

        with tab1:
            st.markdown("### Code Review Summary")
            st.write(feedback.get("summary", "No summary available."))

            score = feedback.get("score", 0)
            st.markdown(f"### Code Quality Score: **{score}/100**")
            st.progress(min(score / 100, 1.0))

            # Use correct summary dict
            summary = report.get("summary", {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Maintainability", f"{summary.get('maintainability_index', 0)}")
            with col2:
                st.metric("Avg Complexity", f"{summary.get('avg_complexity', 0)}")
            with col3:
                st.metric("Lint Issues", f"{summary.get('lint_issues', 0)}")

            st.markdown(f"""
            **Total Functions:** {summary.get('total_functions', 0)}  
            **Security Issues:** {summary.get('security_issues', 0)}  
            **Static analysis time:** {static_time:.1f}s  
            **AI run time:** {ai_time:.1f}s
            """)

        with tab2:
            st.markdown("### Function-Level Insights")
            functions = feedback.get("function_feedback", [])
            if not functions:
                st.info("No function-level feedback available.")
            for f in functions:
                st.success(f"✅ {f}")

        with tab3:
            st.markdown("### General Recommendations")
            recs = feedback.get("general_recommendations", [])
            if not recs:
                st.info("No recommendations.")
            for r in recs:
                st.warning(f"💡 {r}")

        # Export section
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
