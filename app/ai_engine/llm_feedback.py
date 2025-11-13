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
    
    result = {
        "score": score,
        "summary": summary_text,
        "function_feedback": feedback[:5],
        "general_recommendations": recs,
    }
    
    # Debug: Print L1 result
    print(f"[DEBUG L1] Generated feedback: score={score}, functions={len(feedback)}, recs={len(recs)}")
    
    return result


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
You are a practical code quality expert. Analyze this Python code and provide actionable feedback.

Code Statistics:
- Functions: {summary.get("total_functions", 0)}
- Avg Complexity: {summary.get("avg_complexity", 0):.1f}
- Security Issues: {summary.get("security_issues", 0)}
- Style Issues: {summary.get("lint_issues", 0)}

Provide practical feedback as JSON:
{{
  "summary": "Brief practical assessment of code quality and immediate fixes needed",
  "function_feedback": ["Specific actionable feedback for each function"],
  "general_recommendations": ["Practical improvements and best practices"],
  "score": {summary.get("overall_score", 0)}
}}

Functions to review: {', '.join(func_names[:3]) if func_names else 'None'}
"""

    prompt = _trim_prompt(prompt)

    with torch.inference_mode():
        response = _phi_model.generate(
            **_phi_tokenizer(prompt, return_tensors="pt").to(_phi_model.device),
            max_new_tokens=400,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            pad_token_id=_phi_tokenizer.eos_token_id
        )

    text_output = _phi_tokenizer.decode(response[0], skip_special_tokens=True).strip()
    
    # Debug: Print the raw model output
    print(f"[DEBUG L2] Raw model output: {text_output[:500]}...")
    
    parsed = _extract_json_from_text(text_output)
    
    # Debug: Print parsing result
    print(f"[DEBUG L2] Parsed JSON: {parsed is not None}")

    if parsed:
        parsed.setdefault("summary", "No summary provided.")
        parsed.setdefault("function_feedback", [])
        parsed.setdefault("general_recommendations", [])
        parsed.setdefault("score", summary.get("overall_score", 0))
        
        # Check if we got valid content or just placeholders
        valid_summary = parsed.get("summary", "") and "No summary provided" not in parsed.get("summary", "")
        valid_feedback = parsed.get("function_feedback") and len(parsed.get("function_feedback", [])) > 0
        valid_recs = parsed.get("general_recommendations") and len(parsed.get("general_recommendations", [])) > 0
        
        # If we don't have good content, force fallback
        if not (valid_summary and valid_feedback and valid_recs):
            print(f"[DEBUG L2] AI response incomplete, using fallback instead")
            parsed = None  # Force fallback
        else:
            # Add L2-specific practical markers
            if "summary" in parsed and parsed["summary"]:
                parsed["summary"] = f"[L2 Practical] 🛠️ {parsed['summary']}"
            
            # Add practical context to function feedback
            if len(parsed.get("function_feedback", [])) > 0:
                enhanced_feedback = []
                for feedback in parsed["function_feedback"]:
                    enhanced_feedback.append(f"🔧 [Code Quality] {feedback}")
                parsed["function_feedback"] = enhanced_feedback
            
            # Add actionable context to recommendations
            if len(parsed.get("general_recommendations", [])) > 0:
                enhanced_recs = []
                for rec in parsed["general_recommendations"]:
                    enhanced_recs.append(f"⚙️ [Best Practice] {rec}")
                parsed["general_recommendations"] = enhanced_recs
            
            print(f"[DEBUG L2] Returning enhanced parsed result")
            return parsed

    # Enhanced fallback with practical focus and actual analysis
    print(f"[DEBUG L2] JSON extraction failed, using enhanced practical fallback")
    
    # Analyze the actual code for practical issues
    function_issues = []
    practical_recs = []
    
    ast_funcs = report.get("details", {}).get("ast_analysis", [])
    
    for func in ast_funcs[:3]:
        fname = func.get('name', 'unknown')
        complexity = func.get('complexity', 0)
        lines = func.get('lines', 0)
        has_docs = func.get('has_docstring', False)
        
        if complexity > 10:
            function_issues.append(f"🔧 [Code Quality] {fname}: High complexity ({complexity}) - break into smaller functions")
        elif lines > 50:
            function_issues.append(f"🔧 [Code Quality] {fname}: Large function ({lines} lines) - consider refactoring")
        elif not has_docs:
            function_issues.append(f"🔧 [Code Quality] {fname}: Missing docstring - add documentation")
        else:
            function_issues.append(f"🔧 [Code Quality] {fname}: Well-structured, consider adding type hints")
    
    # Generate practical recommendations based on issues found
    if summary.get("security_issues", 0) > 0:
        practical_recs.append("⚙️ [Best Practice] Address security vulnerabilities immediately")
    if summary.get("lint_issues", 0) > 5:
        practical_recs.append("⚙️ [Best Practice] Fix linting issues to improve code readability")
    if summary.get("avg_complexity", 0) > 5:
        practical_recs.append("⚙️ [Best Practice] Reduce function complexity through decomposition")
    
    if not practical_recs:
        practical_recs = [
            "⚙️ [Best Practice] Add comprehensive error handling and input validation",
            "⚙️ [Best Practice] Implement proper logging for debugging",
            "⚙️ [Best Practice] Add type hints for better code documentation"
        ]
    
    return {
        "summary": f"[L2 Practical] 🛠️ Code analysis reveals {len(ast_funcs)} functions with average complexity {summary.get('avg_complexity', 0):.1f}. Found {summary.get('security_issues', 0)} security issues and {summary.get('lint_issues', 0)} style violations requiring immediate attention.",
        "function_feedback": function_issues,
        "general_recommendations": practical_recs,
        "score": summary.get("overall_score", 0)
    }



# -----------------------
# L3: refinement (polish L2 output)
# -----------------------
def generate_l3_feedback(report: dict) -> dict:
    """Generate sophisticated L3 analysis with advanced architectural insights."""
    text_gen = load_phi_model()
    
    # Get the original report data for deeper analysis
    summary = report.get("summary", {})
    ast_data = report.get("details", {}).get("ast_analysis", [])
    static_analysis = report.get("details", {}).get("static_analysis", {})
    
    # Extract deeper metrics for L3 analysis
    total_functions = summary.get("total_functions", 0)
    avg_complexity = summary.get("avg_complexity", 0)
    maintainability = summary.get("maintainability_index", 0)
    security_issues = summary.get("security_issues", 0)
    lint_issues = summary.get("lint_issues", 0)
    
    # Analyze function patterns for L3
    complex_functions = [f for f in ast_data if f.get("complexity", 0) > 10]
    undocumented_functions = [f for f in ast_data if not f.get("has_docstring", False)]
    large_functions = [f for f in ast_data if f.get("lines", 0) > 30]

    prompt = f"""
You are a senior software architect. Analyze this codebase for architectural insights.

Architecture Analysis:
- Functions: {total_functions}
- Avg Complexity: {avg_complexity:.1f}  
- Complex Functions: {len(complex_functions)}
- Undocumented: {len(undocumented_functions)}
- Security Issues: {security_issues}

Provide architectural assessment as JSON:
{{
  "summary": "Architectural assessment focusing on design patterns and technical debt",
  "function_feedback": ["Architectural insights for each function"],
  "general_recommendations": ["Strategic architectural improvements"],
  "score": {summary.get("overall_score", 0)}
}}

Key functions: {', '.join([f.get('name', 'unknown') for f in ast_data[:3]])}
"""
    prompt = _trim_prompt(prompt)

    with torch.inference_mode():
        response = _phi_model.generate(
            **_phi_tokenizer(prompt, return_tensors="pt").to(_phi_model.device),
            max_new_tokens=500,
            temperature=0.35,
            top_p=0.9,
            do_sample=True,
            pad_token_id=_phi_tokenizer.eos_token_id
        )

    text_output = _phi_tokenizer.decode(response[0], skip_special_tokens=True).strip()
    
    # Debug: Print the raw L3 model output
    print(f"[DEBUG L3] Raw model output: {text_output[:500]}...")
    
    parsed = _extract_json_from_text(text_output)
    
    # Debug: Print parsing result
    print(f"[DEBUG L3] Parsed JSON: {parsed is not None}")

    if parsed:
        parsed.setdefault("summary", "No summary provided.")
        parsed.setdefault("function_feedback", [])
        parsed.setdefault("general_recommendations", [])
        parsed.setdefault("score", summary.get("overall_score", 0))
        
        # Check if we got valid architectural content or just placeholders
        valid_summary = parsed.get("summary", "") and "No summary provided" not in parsed.get("summary", "")
        valid_feedback = parsed.get("function_feedback") and len(parsed.get("function_feedback", [])) > 0
        valid_recs = parsed.get("general_recommendations") and len(parsed.get("general_recommendations", [])) > 0
        
        # If we don't have good content, force fallback
        if not (valid_summary and valid_feedback and valid_recs):
            print(f"[DEBUG L3] AI response incomplete, using enhanced fallback instead")
            parsed = None  # Force fallback
        else:
            # Enhance L3 with sophisticated architectural content
            if "summary" in parsed and parsed["summary"]:
                parsed["summary"] = f"[L3 Enhanced] 🔍 {parsed['summary']}"
            
            # Add architectural context to function feedback for L3
            if len(parsed.get("function_feedback", [])) > 0:
                enhanced_feedback = []
                for feedback in parsed["function_feedback"]:
                    enhanced_feedback.append(f"⚡ [Architectural] {feedback}")
                parsed["function_feedback"] = enhanced_feedback
            
            # Add strategic context to recommendations for L3
            if len(parsed.get("general_recommendations", [])) > 0:
                enhanced_recs = []
                for rec in parsed["general_recommendations"]:
                    enhanced_recs.append(f"🏗️ [Strategic] {rec}")
                parsed["general_recommendations"] = enhanced_recs
            
            print(f"[DEBUG L3] Returning enhanced architectural result")
            return parsed

    # Enhanced fallback for L3 mode with sophisticated architectural analysis
    print(f"[DEBUG L3] JSON extraction failed, using enhanced architectural fallback")
    
    # Perform architectural analysis
    architectural_feedback = []
    strategic_recs = []
    
    # Analyze architectural patterns
    high_complexity_funcs = [f for f in ast_data if f.get("complexity", 0) > 10]
    undocumented_funcs = [f for f in ast_data if not f.get("has_docstring", False)]
    large_funcs = [f for f in ast_data if f.get("lines", 0) > 30]
    
    # Generate architectural insights
    for func in ast_data[:3]:
        fname = func.get('name', 'unknown')
        complexity = func.get('complexity', 0)
        
        if complexity > 15:
            architectural_feedback.append(f"⚡ [Architectural] {fname}: Critical complexity violation - requires immediate decomposition using Strategy pattern")
        elif complexity > 10:
            architectural_feedback.append(f"⚡ [Architectural] {fname}: High complexity suggests Single Responsibility Principle violation - consider modular refactoring")
        elif func.get('lines', 0) > 50:
            architectural_feedback.append(f"⚡ [Architectural] {fname}: Large function indicates procedural design - apply Extract Method refactoring")
        else:
            architectural_feedback.append(f"⚡ [Architectural] {fname}: Well-bounded function adheres to clean architecture principles")
    
    # Generate strategic recommendations based on patterns
    technical_debt_score = len(high_complexity_funcs) + len(undocumented_funcs) + len(large_funcs)
    
    if len(undocumented_funcs) > len(ast_data) * 0.5:
        strategic_recs.append("🏗️ [Strategic] Implement documentation-driven development to establish architectural contracts")
    
    if len(high_complexity_funcs) > 0:
        strategic_recs.append("🏗️ [Strategic] Apply Domain-Driven Design to reduce cognitive complexity and improve maintainability")
    
    if technical_debt_score > len(ast_data):
        strategic_recs.append("🏗️ [Strategic] Establish architectural layers with dependency injection to improve testability")
    else:
        strategic_recs.append("🏗️ [Strategic] Consider implementing hexagonal architecture for better separation of concerns")
    
    if len(strategic_recs) == 1:
        strategic_recs.extend([
            "🏗️ [Strategic] Implement CQRS pattern for scalable data operations",
            "🏗️ [Strategic] Establish monitoring and observability for architectural health"
        ])
    
    return {
        "summary": f"[L3 Enhanced] 🔍 Architectural analysis reveals {total_functions} functions with {len(high_complexity_funcs)} high-complexity modules and {len(undocumented_funcs)} undocumented components. Technical debt assessment indicates {technical_debt_score} architectural violations requiring strategic refactoring.",
        "function_feedback": architectural_feedback,
        "general_recommendations": strategic_recs,
        "score": summary.get("overall_score", 0)
    }



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
