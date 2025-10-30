"""Streamlit dashboard placeholder.

Minimal Streamlit dashboard for MVP v1.0.
Features implemented here:
- File upload for single Python file
- Display raw code
- Placeholder panels for static analysis + AI feedback

Run with:
	streamlit run app/ui/dashboard.py
or
	python run_app.py
"""

import streamlit as st
from pathlib import Path
import tempfile


st.set_page_config(page_title="CodeSage - MVP", layout="wide")


st.title("CodeSage — AI Code Reviewer (MVP)")
st.markdown("Upload a Python file to run a quick static + AI review (MVP).")


uploaded = st.file_uploader("Upload a single .py file", type=["py"])


if uploaded is not None:
	code_bytes = uploaded.read()
	try:
		code_text = code_bytes.decode("utf-8")
	except Exception:
		code_text = code_bytes.decode("latin-1")

	st.subheader("Source code")
	st.code(code_text, language="python")

	# Save a temp copy for downstream modules
	temp_dir = tempfile.mkdtemp()
	temp_path = Path(temp_dir) / uploaded.name
	temp_path.write_text(code_text, encoding="utf-8")

	st.sidebar.subheader("Actions")
	if st.sidebar.button("Run Static Analysis"):
		st.sidebar.info("Static analysis will run (not implemented yet).")
		st.warning(
			"Static analyzer not implemented in this scaffold. Next step: add ast_utils and analyzer_utils."
		)

	if st.sidebar.button("Ask AI for feedback"):
		st.sidebar.info("AI feedback will run (not implemented yet).")
		st.warning(
			"LLM feedback not implemented in this scaffold. Next step: add model loader + llm_feedback."
		)
else:
	st.info("Upload a Python file to start a review.")
