import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app import config


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
        summary_text = "Outstanding code quality — clean, efficient, and well structured."
    elif maintain > 70:
        summary_text = "Good overall quality, though a few parts can be simplified."
    elif maintain > 50:
        summary_text = "Moderate code quality — consider improving readability and modularity."
    else:
        summary_text = "Low code quality — refactoring is recommended."

    feedback, recs = [], []
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
            feedback.append(f"'{name}' looks clean and simple — maintain this readability.")

    if issues > 5:
        recs.append("Resolve linting warnings to improve consistency.")
    elif issues > 0:
        recs.append("Fix minor linting issues for better code quality.")
    if maintain < 70:
        recs.append("Improve maintainability by reducing nested loops or adding docstrings.")
    if not recs:
        recs.append("Your code meets high standards of clarity and maintainability.")

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
# Shared Model Loader (GPU-Preferred Phi-3)
# =========================================================

_phi_model, _phi_tokenizer, _phi_pipeline = None, None, None


def load_phi_model():
    """Load Phi-3 Mini optimized for GPU; fallback to CPU if necessary."""
    global _phi_model, _phi_tokenizer, _phi_pipeline

    if _phi_model is not None and _phi_pipeline is not None:
        return _phi_pipeline

    print("[INFO] Loading Phi-3-Mini-4K-Instruct model...")
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    _phi_tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Prefer GPU if available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[INFO] GPU detected: {gpu_name} ({total_vram:.1f} GB VRAM)")

        try:
            _phi_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("[INFO] Model successfully loaded on GPU ✅")
        except Exception as e:
            print(f"[WARN] GPU load failed ({e}); falling back to CPU.")
            _phi_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            )
    else:
        print("[INFO] No GPU available → Loading on CPU (float32).")
        _phi_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )

    _phi_pipeline = pipeline("text-generation", model=_phi_model, tokenizer=_phi_tokenizer)
    print(f"[INFO] Model device: {_phi_model.device}")
    print("[INFO] Phi-3-Mini-4K-Instruct loaded and ready ✅")
    return _phi_pipeline


# =========================================================
# L2 & L3 AI Feedback
# =========================================================

def generate_l2_feedback(report: dict) -> dict:
    """Generate structured AI feedback."""
    text_gen = load_phi_model()
    summary = report.get("summary", {})
    functions = report.get("details", {}).get("ast_analysis", [])
    func_names = [f["name"] for f in functions]
    code_summary = json.dumps(summary, indent=2)

    prompt = f"""
You are CodeSage, an expert AI code reviewer. Respond ONLY with valid JSON.

Analyze this Python code report:
{code_summary}

Functions: {', '.join(func_names) if func_names else 'None'}

Return JSON only:
{{
  "summary": "Brief overall evaluation.",
  "function_feedback": ["Feedback for each function"],
  "general_recommendations": ["2–3 concise suggestions"]
}}
"""

    response = text_gen(prompt, max_new_tokens=400, temperature=0.4, top_p=0.9, do_sample=True)
    text_output = response[0]["generated_text"].strip()

    import re
    matches = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text_output, re.DOTALL)
    for m in reversed(matches):
        try:
            return {**json.loads(m), "score": summary.get("overall_score", 0)}
        except json.JSONDecodeError:
            continue

    return {"summary": text_output, "function_feedback": [], "general_recommendations": [], "score": 0}


def generate_l3_feedback(report: dict) -> dict:
    """Refine and polish L2 output."""
    text_gen = load_phi_model()
    l2_output = generate_l2_feedback(report)
    l2_json = json.dumps(l2_output, indent=2)

    prompt = f"""
You are CodeSage's expert refinement AI.
Analyze the input JSON, which includes function-level feedback and insights.

Task:
- Refine wording for clarity and professionalism.
- Merge redundant points.
- Ensure JSON validity and keep structure intact.

Input JSON:
{l2_json}

Return only valid JSON.
"""

    response = text_gen(prompt, max_new_tokens=400, temperature=0.3, top_p=0.9)
    text_output = response[0]["generated_text"].strip()

    import re
    matches = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text_output, re.DOTALL)
    for m in reversed(matches):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return l2_output


# =========================================================
# AUTO SELECTION ENGINE
# =========================================================

def auto_select_feedback_engine(report: dict) -> dict:
    """Automatically pick L1/L2/L3 based on GPU VRAM."""
    try:
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[AUTO] GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")

            if vram >= 6:
                print("[AUTO] Using L3 (Refined Feedback)")
                return generate_l3_feedback(report)
            elif vram >= 3:
                print("[AUTO] Using L2 (AI Analysis)")
                return generate_l2_feedback(report)
            else:
                print("[AUTO] Using L1 (Heuristic Feedback)")
                return generate_l1_feedback(report)
        else:
            print("[AUTO] No GPU detected → Using L1 (Heuristic)")
            return generate_l1_feedback(report)
    except Exception as e:
        print(f"[AUTO] Error: {e} → Falling back to L1")
        return generate_l1_feedback(report)


# =========================================================
# MAIN INTERFACE
# =========================================================

def generate_feedback(report: dict, mode: str | None = None) -> dict:
    """Select feedback engine manually or automatically."""
    mode = (mode or getattr(config, "FEEDBACK_MODE", "L1")).upper()
    if mode == "AUTO":
        return auto_select_feedback_engine(report)
    elif mode == "L3":
        return generate_l3_feedback(report)
    elif mode == "L2":
        return generate_l2_feedback(report)
    else:
        return generate_l1_feedback(report)


# =========================================================
# TEST ENTRY POINT
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

    print("[INFO] Running auto-tier feedback selection...")
    feedback = generate_feedback(report)

    print("\n=== CODE REVIEW FEEDBACK ===\n")
    print(json.dumps(feedback, indent=2, ensure_ascii=False))
    print("\n[INFO] Done ✅")
