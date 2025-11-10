import json
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app import config

# -----------------------
# Heuristic (L1)
# -----------------------
def compute_quality_score(report: dict) -> int:
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
            feedback.append(f"'{name}' looks clean and simple — maintain this readability.")

    recs = []
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


# -----------------------
# Shared model state (cached per-process)
# -----------------------
_phi_model = None
_phi_tokenizer = None
_phi_pipeline = None
_model_id = "microsoft/Phi-3-mini-4k-instruct"

# prompt / tokenization helpers
_MAX_PROMPT_TOKENS = 3200  # conservative truncation for 4k models
_MAX_NEW_TOKENS = 180      # reduced generation size for speed


def _trim_prompt(prompt: str, max_chars: int = 24_000) -> str:
    """
    Trim prompt to a reasonable char length (not tokens) so we don't hit huge input costs.
    For safety keep the trailing context (most relevant).
    """
    if len(prompt) <= max_chars:
        return prompt
    # keep last max_chars characters (likely contains summary/most-recent items)
    return prompt[-max_chars:]


def load_phi_model():
    """
    Safe loader for Phi-3 Mini (quantized if possible). Decides at runtime:
    - If bitsandbytes is available and GPU present: try 4-bit quantization with CPU offload
    - Else if GPU present: use float16 on GPU
    - Else fallback to float32 on CPU
    """
    global _phi_model, _phi_tokenizer, _phi_pipeline, _model_id

    if _phi_pipeline is not None:
        return _phi_pipeline

    print("[INFO] Loading model:", _model_id)
    _phi_tokenizer = AutoTokenizer.from_pretrained(_model_id, use_fast=True)

    # If GPU available try quantized path (bitsandbytes)
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig

            print("[INFO] bitsandbytes present: attempting 4-bit quantization with CPU offload.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,  # allow offload of fp32 params to CPU
            )

            _phi_model = AutoModelForCausalLM.from_pretrained(
                _model_id,
                device_map="auto",
                quantization_config=bnb_config,
                low_cpu_mem_usage=True
            )
            print("[INFO] Loaded quantized model with bitsandbytes (device_map=auto).")

        except Exception as exc:
            # if quantization fails, fallback to float16 on GPU
            print(f"[WARN] Quantized load failed ({exc}). Falling back to float16 on GPU.")
            _phi_model = AutoModelForCausalLM.from_pretrained(
                _model_id,
                device_map="cuda",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("[INFO] Loaded model on GPU with float16.")
    else:
        # No GPU: load on CPU (float32)
        print("[INFO] No GPU available — loading model on CPU (float32).")
        _phi_model = AutoModelForCausalLM.from_pretrained(
            _model_id,
            torch_dtype=torch.float32
        )

    # Create a pipeline around the loaded model for convenience
    # Use explicit device: the model will already be placed by device_map/auto
    _phi_pipeline = pipeline("text-generation", model=_phi_model, tokenizer=_phi_tokenizer)
    print(f"[INFO] Model ready. Main device: {_phi_model.device}")
    return _phi_pipeline


# -----------------------
# JSON extraction helper
# -----------------------
_json_block_re = re.compile(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', re.DOTALL)


def _extract_json_from_text(text: str):
    """
    Find the last well-formed JSON object in 'text' and return its parsed value.
    Return None if not found/parseable.
    """
    matches = _json_block_re.findall(text)
    if not matches:
        return None
    # Try the last match first (most likely the model's final JSON block)
    for m in reversed(matches):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return None


# -----------------------
# L2: generate with local model
# -----------------------
def generate_l2_feedback(report: dict) -> dict:
    """Generate structured AI feedback (JSON-enforced)."""
    text_gen = load_phi_model()
    summary = report.get("summary", {})
    functions = report.get("details", {}).get("ast_analysis", [])
    func_names = [f.get("name", "unknown") for f in functions]

    code_summary = json.dumps(summary, indent=2)

    prompt = f"""
You are CodeSage, an expert AI code reviewer.

Analyze the following Python code quality report and provide a constructive review.
Be concise, professional, and insightful.

Respond ONLY with a valid JSON object using these EXACT keys:
  - summary: (1–2 lines describing overall code quality)
  - function_feedback: (a list of 2–5 short comments, one per function)
  - general_recommendations: (a list of 2–3 broader improvement suggestions)
  - score: (overall numeric score out of 100)

Example JSON:
{{
  "summary": "The code is clean and efficient, but could use better validation.",
  "function_feedback": [
    "Function 'foo' is well optimized.",
    "Function 'bar' should include input validation."
  ],
  "general_recommendations": [
    "Add type hints.",
    "Use logging instead of print statements."
  ],
  "score": 89.5
}}

Code Report:
{json.dumps(summary, indent=2)}

Functions: {', '.join(func_names) if func_names else 'None'}
"""

    prompt = _trim_prompt(prompt)

    with torch.inference_mode():
        response = _phi_model.generate(
            **_phi_tokenizer(prompt, return_tensors="pt").to(_phi_model.device),
            max_new_tokens=280,
            temperature=0.4,
            top_p=0.9,
            do_sample=True
        )

    text_output = _phi_tokenizer.decode(response[0], skip_special_tokens=True).strip()
    parsed = _extract_json_from_text(text_output)

    if parsed:
        parsed.setdefault("summary", "No summary provided.")
        parsed.setdefault("function_feedback", [])
        parsed.setdefault("general_recommendations", [])
        parsed.setdefault("score", summary.get("overall_score", 0))
        return parsed

    # Fallback JSON to prevent UI blanks
    return {
        "summary": "Could not extract structured feedback, but model response was:",
        "function_feedback": [text_output],
        "general_recommendations": [],
        "score": summary.get("overall_score", 0)
    }



# -----------------------
# L3: refinement (polish L2 output)
# -----------------------
def generate_l3_feedback(report: dict) -> dict:
    """Refine and polish L2 feedback while enforcing proper JSON schema."""
    text_gen = load_phi_model()
    l2_output = generate_l2_feedback(report)
    l2_json = json.dumps(l2_output, indent=2)

    prompt = f"""
You are CodeSage's refinement AI.
You will rewrite the provided JSON feedback with improved clarity,
but keep the same structure and keys.

ALWAYS respond ONLY with valid JSON containing:
  - summary
  - function_feedback
  - general_recommendations
  - score

Input JSON:
{l2_json}
"""
    prompt = _trim_prompt(prompt)

    with torch.inference_mode():
        response = _phi_model.generate(
            **_phi_tokenizer(prompt, return_tensors="pt").to(_phi_model.device),
            max_new_tokens=220,
            temperature=0.35,
            top_p=0.9,
            do_sample=True
        )

    text_output = _phi_tokenizer.decode(response[0], skip_special_tokens=True).strip()
    parsed = _extract_json_from_text(text_output)

    if parsed:
        parsed.setdefault("summary", "No summary provided.")
        parsed.setdefault("function_feedback", [])
        parsed.setdefault("general_recommendations", [])
        parsed.setdefault("score", l2_output.get("score", 0))
        return parsed

    # fallback if model failed to produce JSON
    return l2_output



# -----------------------
# auto selection
# -----------------------
def auto_select_feedback_engine(report: dict) -> dict:
    try:
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[AUTO] GPU detected: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
            if vram >= 6:
                return generate_l3_feedback(report)
            elif vram >= 3:
                return generate_l2_feedback(report)
            else:
                return generate_l1_feedback(report)
        else:
            return generate_l1_feedback(report)
    except Exception as e:
        print("[AUTO] error:", e)
        return generate_l1_feedback(report)


# -----------------------
# main entry
# -----------------------
def generate_feedback(report: dict, mode: str | None = None) -> dict:
    mode = (mode or getattr(config, "FEEDBACK_MODE", "L1")).upper()
    if mode == "AUTO":
        return auto_select_feedback_engine(report)
    elif mode == "L3":
        return generate_l3_feedback(report)
    elif mode == "L2":
        return generate_l2_feedback(report)
    else:
        return generate_l1_feedback(report)


# CLI
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from app.core.report_generator import generate_code_report

    if len(sys.argv) < 2:
        print("Usage: python -m app.ai_engine.llm_feedback <target_file.py>")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    report = generate_code_report(target_path)
    feedback = generate_feedback(report, mode="L2")
    print(json.dumps(feedback, indent=2, ensure_ascii=False))
