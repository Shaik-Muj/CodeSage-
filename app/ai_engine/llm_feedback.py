import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.config_fb.settings import FEEDBACK_MODE


# =========================================================
# L1: Heuristic Feedback (Fast, Offline)
# =========================================================

def compute_quality_score(report: dict) -> int:
    """Compute an overall quality score from static metrics."""
    summary = report.get("summary", {})
    maintainability = summary.get("maintainability_index", 100)
    complexity = summary.get("avg_complexity", 0)
    issues = summary.get("lint_issues", 0)

    score = (
        (maintainability * 0.5)
        + max(0, 100 - complexity * 3) * 0.3
        + max(0, 100 - issues * 5) * 0.2
    )
    return round(max(0, min(score, 100)))


def generate_l1_feedback(report: dict) -> dict:
    """Lightweight rule-based feedback (Heuristic)."""
    summary = report.get("summary", {})
    ast_data = report.get("details", {}).get("ast_analysis", [])

    maintain = summary.get("maintainability_index", 0)
    complexity = summary.get("avg_complexity", 0)
    issues = summary.get("lint_issues", 0)

    if maintain > 85 and complexity < 8 and issues < 3:
        summary_text = "Outstanding code quality — clean, efficient, and very well structured."
    elif maintain > 70:
        summary_text = "Good code quality overall, though a few areas can be simplified."
    elif maintain > 50:
        summary_text = "Moderate code quality — consider improving readability and modularity."
    else:
        summary_text = "Code quality appears low — refactoring is recommended."

    feedback = []
    for func in ast_data:
        name = func.get("name", "unknown_function")
        comp = func.get("complexity", 0)

        if comp > 15:
            feedback.append(f"Function '{name}' is too complex (complexity {comp}). Consider breaking it up.")
        elif comp > 10:
            feedback.append(f"'{name}' could be simplified; complexity {comp} suggests too much logic.")
        elif comp > 5:
            feedback.append(f"'{name}' is manageable but could be refactored slightly.")
        else:
            feedback.append(f"'{name}' looks clean and simple — maintain this level of readability.")

    recs = []
    if issues > 5:
        recs.append("Resolve linting warnings to improve consistency.")
    elif issues > 0:
        recs.append("Fix minor linting issues for better code quality.")
    if maintain < 70:
        recs.append("Improve maintainability by reducing nested loops or adding docstrings.")
    if not recs:
        recs.append("Your code meets high standards of clarity, security, and maintainability.")

    random.shuffle(feedback)
    random.shuffle(recs)

    score = compute_quality_score(report)
    return {
        "score": score,
        "summary": summary_text,
        "function_feedback": feedback[:5],
        "general_recommendations": recs,
    }


# =========================================================
# L2: Local AI Model Feedback (Phi-3-Mini-4K-Instruct)
# =========================================================

_model, _tokenizer, _text_gen = None, None, None


def load_l2_model():
    """Load local Phi-3-Mini-4K-Instruct model once."""
    global _model, _tokenizer, _text_gen

    if _model is not None and _text_gen is not None:
        return _text_gen

    print("[INFO] Loading Phi-3-Mini-4K-Instruct model... this may take a few minutes on first run.")

    model_id = "microsoft/Phi-3-mini-4k-instruct"

    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        load_in_4bit=True
    )

    _text_gen = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer
    )

    print("[INFO] Phi-3-Mini-4K-Instruct model successfully loaded ✅")
    return _text_gen


def generate_l2_feedback(report: dict) -> dict:
    """Generate deep AI feedback using local Phi-3-Mini-4K-Instruct model."""
    text_gen = load_l2_model()
    summary = report.get("summary", {})
    functions = report.get("details", {}).get("ast_analysis", [])

    func_names = [f["name"] for f in functions]
    code_summary = json.dumps(summary, indent=2)

    prompt = f"""
You are CodeSage, an expert AI code reviewer. Respond ONLY with valid JSON.

Task:
Analyze the following Python code report and provide constructive insights.

Code Report:
{code_summary}

Functions found: {', '.join(func_names) if func_names else 'None'}

Expected JSON format (no explanations, no markdown):

{{
  "summary": "Brief overall evaluation of code quality.",
  "function_feedback": [
    "Short feedback line about each function"
  ],
  "general_recommendations": [
    "2–3 concise improvement suggestions"
  ]
}}

Now return only the JSON response.
"""

    response = text_gen(
        prompt,
        max_new_tokens=400,
        temperature=0.4,
        top_p=0.9,
        do_sample=True
    )

    text_output = response[0]["generated_text"].strip()

    import re

    def extract_json_block(text: str):
        import json, re
        matches = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text, re.DOTALL)
        for m in reversed(matches):  # prefer last complete JSON block
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue
        return None

    feedback = extract_json_block(text_output)
    if not feedback:
        print("[WARN] Model returned malformed JSON. Using fallback.")
        feedback = {
            "summary": text_output.strip(),
            "function_feedback": [],
            "general_recommendations": []
        }

    feedback["function_feedback"] = [f.strip() for f in feedback.get("function_feedback", []) if f.strip()]
    feedback["general_recommendations"] = [r.strip() for r in feedback.get("general_recommendations", []) if r.strip()]

    feedback["score"] = summary.get("overall_score", 0)
    return feedback



# =========================================================
# MAIN INTERFACE — AUTO SELECT
# =========================================================

def generate_feedback(report: dict) -> dict:
    """Main entrypoint — auto-select feedback engine."""
    if FEEDBACK_MODE.upper() == "L2":
        print("[INFO] Using L2: Local AI Feedback (Phi-3-Mini-4K-Instruct)")
        return generate_l2_feedback(report)
    else:
        print("[INFO] Using L1: Heuristic Feedback")
        return generate_l1_feedback(report)


# =========================================================
# TESTING ENTRY POINT
# =========================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from app.core.report_generator import generate_code_report

    if len(sys.argv) < 2:
        print("Usage: python -m app.ai_engine.llm_feedback <target_file.py>")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    print(f"[INFO] Generating static report for {target_path.name} ...")
    report = generate_code_report(target_path)

    # Wrap it into the correct structure
    wrapped_report = {
        "summary": report,
        "details": {
            "ast_analysis": report.get("ast_analysis", [])
        }
    }

    print(f"[INFO] Generating AI feedback (mode: {FEEDBACK_MODE}) ... this may take a while on first run.")
    feedback = generate_feedback(report)

    print("\n=== CODE REVIEW FEEDBACK ===\n")
    print(json.dumps(feedback, indent=2, ensure_ascii=False))
    print("\n[INFO] Done.")
