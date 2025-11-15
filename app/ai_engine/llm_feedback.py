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
You are an experienced software engineer conducting a thorough code review. Analyze this Python codebase and provide detailed, actionable feedback focused on immediate improvements and code quality.

Codebase Analysis:
- Total Functions: {summary.get("total_functions", 0)}
- Average Complexity Score: {summary.get("avg_complexity", 0):.2f}
- Maintainability Index: {summary.get("maintainability_index", 0):.1f}
- Security Vulnerabilities: {summary.get("security_issues", 0)}
- Style/Lint Issues: {summary.get("lint_issues", 0)}

Provide comprehensive practical analysis as JSON:
{{
  "summary": "Detailed 3-4 sentence assessment covering code quality, maintainability concerns, and priority issues that need immediate attention",
  "function_feedback": ["Specific detailed feedback for each function covering complexity, readability, error handling, and optimization opportunities"],
  "general_recommendations": ["Comprehensive improvement suggestions covering testing, documentation, performance, security, and code organization"],
  "score": {summary.get("overall_score", 0)}
}}

Key functions to analyze: {', '.join(func_names[:5]) if func_names else 'None'}
Focus on practical issues like error handling, input validation, performance bottlenecks, and maintainability concerns.
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
        valid_summary = (parsed.get("summary", "") and 
                        "No summary provided" not in parsed.get("summary", "") and
                        "Brief practical assessment" not in parsed.get("summary", "") and
                        len(parsed.get("summary", "")) > 20)
        valid_feedback = (parsed.get("function_feedback") and 
                         len(parsed.get("function_feedback", [])) > 0 and
                         "Specific actionable feedback" not in str(parsed.get("function_feedback", [])) and
                         "each function" not in str(parsed.get("function_feedback", [])))
        valid_recs = (parsed.get("general_recommendations") and 
                     len(parsed.get("general_recommendations", [])) > 0 and
                     "Practical improvements" not in str(parsed.get("general_recommendations", [])) and
                     "best practices" not in str(parsed.get("general_recommendations", [])))
        
        # If we don't have good content, force fallback
        if not (valid_summary and valid_feedback and valid_recs):
            print(f"[DEBUG L2] AI response incomplete, using fallback instead")
            parsed = None  # Force fallback
        else:
            # Add L2-specific practical markers
            if "summary" in parsed and parsed["summary"]:
                parsed["summary"] = f"[L2 Practical Analysis] {parsed['summary']}"
            
            # Add practical context to function feedback
            if len(parsed.get("function_feedback", [])) > 0:
                enhanced_feedback = []
                for feedback in parsed["function_feedback"]:
                    enhanced_feedback.append(f"Code Quality Review: {feedback}")
                parsed["function_feedback"] = enhanced_feedback
            
            # Add actionable context to recommendations
            if len(parsed.get("general_recommendations", [])) > 0:
                enhanced_recs = []
                for rec in parsed["general_recommendations"]:
                    enhanced_recs.append(f"Best Practice: {rec}")
                parsed["general_recommendations"] = enhanced_recs
            
            print(f"[DEBUG L2] Returning enhanced parsed result")
            return parsed

    # Enhanced fallback with comprehensive practical analysis
    print(f"[DEBUG L2] JSON extraction failed, using comprehensive practical analysis")
    
    # Perform detailed code analysis
    function_issues = []
    practical_recs = []
    
    ast_funcs = report.get("details", {}).get("ast_analysis", [])
    
    # Detailed function analysis
    for func in ast_funcs:
        fname = func.get('name', 'unknown')
        complexity = func.get('complexity', 0)
        lines = func.get('lines', 0)
        has_docs = func.get('has_docstring', False)
        params = func.get('parameters', [])
        returns = func.get('returns', False)
        
        issues = []
        improvements = []
        
        # Complexity analysis with detailed context
        if complexity > 15:
            issues.append(f"excessive cyclomatic complexity ({complexity}) indicates multiple responsibilities and makes testing difficult")
        elif complexity > 10:
            issues.append(f"high complexity ({complexity}) suggests the function is doing too much and should be decomposed")
        elif complexity > 7:
            issues.append(f"moderate complexity ({complexity}) may benefit from simplification or better organization")
        
        # Function size analysis
        if lines > 100:
            issues.append(f"function is very large ({lines} lines) which violates the single responsibility principle")
        elif lines > 50:
            issues.append(f"function length ({lines} lines) is above recommended limits and may be hard to understand")
        elif lines > 30:
            improvements.append(f"function length ({lines} lines) is getting substantial, consider breaking into smaller pieces")
        
        # Documentation analysis
        if not has_docs:
            if complexity > 5:
                issues.append("lacks documentation which is critical given its complexity")
            else:
                improvements.append("would benefit from a descriptive docstring explaining its purpose")
        
        # Parameter analysis
        param_count = len(params)
        if param_count > 5:
            issues.append(f"has too many parameters ({param_count}) which suggests it may be doing too much")
        elif param_count > 3:
            improvements.append(f"parameter count ({param_count}) could be reduced using parameter objects or configuration")
        
        # Type hint analysis
        if not returns and complexity > 1:
            improvements.append("should include return type annotations for better code clarity")
        
        if param_count > 0:
            improvements.append("would benefit from parameter type hints to improve IDE support and catch errors early")
        
        # Generate comprehensive feedback
        if issues:
            primary_issues = "; ".join(issues[:2])
            function_issues.append(f"Code Quality Review: Function '{fname}' {primary_issues}")
        elif improvements:
            suggested_improvements = "; ".join(improvements[:2])
            function_issues.append(f"Code Quality Review: Function '{fname}' is well-structured but {suggested_improvements}")
        else:
            function_issues.append(f"Code Quality Review: Function '{fname}' follows good practices with appropriate complexity and structure")
    
    # Generate comprehensive practical recommendations based on analysis
    security_issues = summary.get("security_issues", 0)
    lint_issues = summary.get("lint_issues", 0)
    avg_complexity = summary.get("avg_complexity", 0)
    maintainability = summary.get("maintainability_index", 0)
    
    # Critical issues first
    if security_issues > 0:
        practical_recs.append(f"Best Practice: Address {security_issues} security vulnerability(s) immediately using tools like bandit for Python security scanning and implement proper input validation")
    
    # Code quality issues
    if lint_issues > 15:
        practical_recs.append(f"Best Practice: Resolve {lint_issues} code style violations using automated tools like black for formatting, flake8 for linting, and isort for import organization")
    elif lint_issues > 5:
        practical_recs.append(f"Best Practice: Address {lint_issues} style issues to improve code readability and maintainability using standard Python style guides")
    
    # Complexity management
    if avg_complexity > 8:
        practical_recs.append("Best Practice: Implement systematic complexity reduction using Extract Method refactoring, single responsibility principle, and consider using design patterns like Strategy or State")
    elif avg_complexity > 5:
        practical_recs.append("Best Practice: Monitor complexity growth and apply early refactoring techniques to prevent technical debt accumulation")
    
    # Documentation assessment
    undocumented_count = len([f for f in ast_funcs if not f.get('has_docstring', False)])
    total_funcs = len(ast_funcs)
    if total_funcs > 0:
        doc_percentage = (total_funcs - undocumented_count) / total_funcs * 100
        if doc_percentage < 50:
            practical_recs.append(f"Best Practice: Improve documentation coverage from {doc_percentage:.1f}% by adding comprehensive docstrings using Google or NumPy documentation style")
        elif doc_percentage < 80:
            practical_recs.append(f"Best Practice: Enhance documentation completeness from {doc_percentage:.1f}% to improve code maintainability and team collaboration")
    
    # Function design issues
    complex_funcs = len([f for f in ast_funcs if f.get('complexity', 0) > 10])
    large_funcs = len([f for f in ast_funcs if f.get('lines', 0) > 50])
    
    if complex_funcs > 0:
        practical_recs.append(f"Best Practice: Refactor {complex_funcs} high-complexity function(s) using SOLID principles, particularly Single Responsibility and Open/Closed principles")
    
    if large_funcs > 0:
        practical_recs.append(f"Best Practice: Decompose {large_funcs} large function(s) into smaller, focused units to improve testability and maintainability")
    
    # Maintainability considerations
    if maintainability < 70:
        practical_recs.append("Best Practice: Improve overall maintainability by reducing complexity, adding comprehensive tests, and implementing consistent error handling patterns")
    
    # General best practices if no major issues
    if len(practical_recs) < 3:
        practical_recs.extend([
            "Best Practice: Implement comprehensive error handling with try-except blocks, proper logging, and graceful failure modes",
            "Best Practice: Add type hints throughout the codebase to improve IDE support, catch errors early, and enhance code documentation",
            "Best Practice: Establish automated testing with pytest, including unit tests, integration tests, and code coverage monitoring",
            "Best Practice: Set up continuous integration with automated code quality checks, security scanning, and dependency vulnerability assessment"
        ])
    
    # Limit recommendations to most important ones
    practical_recs = practical_recs[:5]
    
    # Generate comprehensive summary
    total_issues = security_issues + lint_issues + complex_funcs + large_funcs + undocumented_count
    quality_assessment = "excellent" if total_issues == 0 else "good" if total_issues <= 3 else "moderate" if total_issues <= 7 else "needs improvement"
    
    summary_text = f"[L2 Practical Analysis] Comprehensive code review of {len(ast_funcs)} functions reveals {quality_assessment} overall quality with average complexity of {avg_complexity:.1f}. "
    
    if security_issues > 0:
        summary_text += f"Critical attention needed for {security_issues} security vulnerability(s). "
    
    if lint_issues > 10:
        summary_text += f"Code style requires significant cleanup with {lint_issues} violations. "
    elif lint_issues > 0:
        summary_text += f"Minor style improvements needed with {lint_issues} formatting issues. "
    
    if complex_funcs > 0 or large_funcs > 0:
        summary_text += f"Refactoring recommended for {complex_funcs} complex and {large_funcs} oversized functions. "
    
    if undocumented_count > len(ast_funcs) * 0.5:
        summary_text += f"Documentation coverage is low with {undocumented_count} undocumented functions. "
    
    if maintainability < 70:
        summary_text += f"Maintainability index of {maintainability:.1f} indicates technical debt accumulation."
    else:
        summary_text += f"Maintainability index of {maintainability:.1f} shows good code organization."
    
    return {
        "summary": summary_text,
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
You are a senior software architect conducting a comprehensive design review. Analyze this codebase for architectural quality, design patterns, and long-term maintainability.

Architectural Metrics:
- Total Functions: {total_functions}
- Average Complexity Score: {avg_complexity:.2f}
- High-Complexity Functions: {len(complex_functions)}
- Undocumented Components: {len(undocumented_functions)}
- Large Functions (>30 lines): {len(large_functions)}
- Security Concerns: {security_issues}
- Maintainability Index: {maintainability:.1f}

Provide detailed architectural analysis as JSON:
{{
  "summary": "Comprehensive 4-5 sentence architectural assessment covering design quality, modularity, coupling/cohesion, scalability concerns, and technical debt impact on long-term maintenance",
  "function_feedback": ["Detailed architectural analysis for each function covering design patterns, SOLID principles compliance, coupling issues, and refactoring opportunities"],
  "general_recommendations": ["Strategic architectural improvements covering design patterns, modularization, dependency management, testing architecture, and scalability considerations"],
  "score": {summary.get("overall_score", 0)}
}}

Functions for architectural review: {', '.join([f.get('name', 'unknown') for f in ast_data[:5]])}
Focus on design patterns, SOLID principles, modularity, coupling/cohesion, and long-term architectural sustainability.
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
        valid_summary = (parsed.get("summary", "") and 
                        "No summary provided" not in parsed.get("summary", "") and
                        "Architectural assessment focusing" not in parsed.get("summary", "") and
                        len(parsed.get("summary", "")) > 20)
        valid_feedback = (parsed.get("function_feedback") and 
                         len(parsed.get("function_feedback", [])) > 0 and
                         "Architectural insights" not in str(parsed.get("function_feedback", [])) and
                         "each function" not in str(parsed.get("function_feedback", [])))
        valid_recs = (parsed.get("general_recommendations") and 
                     len(parsed.get("general_recommendations", [])) > 0 and
                     "Strategic architectural improvements" not in str(parsed.get("general_recommendations", [])))
        
        # If we don't have good content, force fallback
        if not (valid_summary and valid_feedback and valid_recs):
            print(f"[DEBUG L3] AI response incomplete, using enhanced fallback instead")
            parsed = None  # Force fallback
        else:
            # Enhance L3 with sophisticated architectural content
            if "summary" in parsed and parsed["summary"]:
                parsed["summary"] = f"[L3 Architectural Analysis] {parsed['summary']}"
            
            # Add architectural context to function feedback for L3
            if len(parsed.get("function_feedback", [])) > 0:
                enhanced_feedback = []
                for feedback in parsed["function_feedback"]:
                    enhanced_feedback.append(f"Architectural Review: {feedback}")
                parsed["function_feedback"] = enhanced_feedback
            
            # Add strategic context to recommendations for L3
            if len(parsed.get("general_recommendations", [])) > 0:
                enhanced_recs = []
                for rec in parsed["general_recommendations"]:
                    enhanced_recs.append(f"Strategic Recommendation: {rec}")
                parsed["general_recommendations"] = enhanced_recs
            
            print(f"[DEBUG L3] Returning enhanced architectural result")
            return parsed

    # Enhanced fallback for L3 mode with comprehensive architectural analysis
    print(f"[DEBUG L3] JSON extraction failed, using comprehensive architectural analysis")
    
    # Perform detailed architectural analysis
    architectural_feedback = []
    strategic_recs = []
    
    # Comprehensive architectural pattern analysis
    high_complexity_funcs = [f for f in ast_data if f.get("complexity", 0) > 10]
    undocumented_funcs = [f for f in ast_data if not f.get("has_docstring", False)]
    large_funcs = [f for f in ast_data if f.get("lines", 0) > 30]
    very_large_funcs = [f for f in ast_data if f.get("lines", 0) > 50]
    
    # Generate detailed architectural insights for each function
    for func in ast_data:
        fname = func.get('name', 'unknown')
        complexity = func.get('complexity', 0)
        lines = func.get('lines', 0)
        has_docs = func.get('has_docstring', False)
        params = func.get('parameters', [])
        
        # Comprehensive architectural assessment based on design principles
        if complexity > 20:
            architectural_feedback.append(f"Architectural Review: Function '{fname}' exhibits severe complexity ({complexity}) indicating multiple architectural violations including Single Responsibility Principle breaches, requiring immediate decomposition using Strategy, Command, or State patterns")
        elif complexity > 15:
            architectural_feedback.append(f"Architectural Review: Function '{fname}' shows high cognitive complexity ({complexity}) suggesting god-object anti-pattern, recommend applying Extract Class refactoring and introducing proper abstraction layers")
        elif complexity > 10:
            architectural_feedback.append(f"Architectural Review: Function '{fname}' has elevated complexity ({complexity}) indicating procedural design approach, consider object-oriented refactoring with dependency injection")
        elif lines > 100:
            architectural_feedback.append(f"Architectural Review: Function '{fname}' is monolithic ({lines} lines) violating cohesion principles, implement Facade pattern for better abstraction and consider modular decomposition")
        elif lines > 50:
            architectural_feedback.append(f"Architectural Review: Function '{fname}' is substantial ({lines} lines) and may violate single responsibility, apply Extract Method pattern to improve modularity")
        elif not has_docs and complexity > 5:
            architectural_feedback.append(f"Architectural Review: Function '{fname}' lacks architectural documentation for complex logic (complexity {complexity}), establish clear API contracts and interface specifications")
        elif len(params) > 5:
            architectural_feedback.append(f"Architectural Review: Function '{fname}' has high parameter coupling ({len(params)} parameters) suggesting insufficient abstraction, consider Parameter Object or Builder patterns")
        else:
            # Positive architectural assessment
            if complexity <= 3 and lines <= 20:
                architectural_feedback.append(f"Architectural Review: Function '{fname}' demonstrates excellent architectural design with low complexity ({complexity}) and appropriate size, adheres to clean architecture principles")
            elif complexity <= 5 and lines <= 30:
                architectural_feedback.append(f"Architectural Review: Function '{fname}' shows good architectural design with manageable complexity ({complexity}), maintains appropriate cohesion and coupling")
            else:
                architectural_feedback.append(f"Architectural Review: Function '{fname}' exhibits reasonable architectural design but could benefit from further modularization to improve maintainability")
    
    # Generate comprehensive strategic recommendations based on architectural patterns
    technical_debt_score = len(high_complexity_funcs) + len(undocumented_funcs) + len(large_funcs)
    
    # Documentation architecture assessment
    if len(undocumented_funcs) > len(ast_data) * 0.7:
        strategic_recs.append("Strategic Recommendation: Establish comprehensive documentation architecture using documentation-driven development principles to create clear architectural contracts and improve system understanding")
    elif len(undocumented_funcs) > len(ast_data) * 0.4:
        strategic_recs.append("Strategic Recommendation: Implement systematic documentation strategy focusing on architectural interfaces and complex business logic to support long-term maintainability")
    
    # Complexity management strategy
    if len(high_complexity_funcs) > len(ast_data) * 0.3:
        strategic_recs.append("Strategic Recommendation: Apply Domain-Driven Design principles to reduce cognitive complexity, implement bounded contexts, and establish clear architectural boundaries")
    elif len(high_complexity_funcs) > 0:
        strategic_recs.append("Strategic Recommendation: Implement strategic refactoring using SOLID principles, focusing on Single Responsibility and Dependency Inversion to improve code modularity")
    
    # Modularization strategy
    if len(large_funcs) > len(ast_data) * 0.4:
        strategic_recs.append("Strategic Recommendation: Implement hexagonal architecture pattern to establish clear separation of concerns and improve system modularity through proper layering")
    elif len(large_funcs) > 0:
        strategic_recs.append("Strategic Recommendation: Apply systematic function decomposition using Extract Method and Extract Class patterns to improve code organization")
    
    # Overall architectural strategy
    if technical_debt_score > len(ast_data):
        strategic_recs.append("Strategic Recommendation: Establish architectural governance including design reviews, coding standards enforcement, and automated architecture compliance checking")
    
    # Ensure comprehensive recommendations
    if len(strategic_recs) < 3:
        strategic_recs.extend([
            "Strategic Recommendation: Consider implementing microservices architecture or modular monolith pattern to support future scalability requirements",
            "Strategic Recommendation: Establish monitoring and observability architecture including application metrics and distributed tracing for operational excellence",
            "Strategic Recommendation: Design API-first architecture with comprehensive interface specifications to support system integration and extensibility"
        ])
    
    # Generate comprehensive architectural summary
    architecture_quality = "excellent" if technical_debt_score == 0 else "good" if technical_debt_score <= 2 else "moderate" if technical_debt_score <= 5 else "requires significant improvement"
    
    summary_text = f"[L3 Architectural Analysis] Comprehensive architectural review of {total_functions} functions reveals {architecture_quality} overall design quality. "
    summary_text += f"System exhibits {len(high_complexity_funcs)} high-complexity components and {len(undocumented_funcs)} undocumented architectural elements. "
    summary_text += f"Technical debt assessment indicates {technical_debt_score} architectural violations impacting long-term maintainability. "
    
    if avg_complexity > 10:
        summary_text += f"Average complexity of {avg_complexity:.1f} suggests significant architectural refactoring opportunities using design patterns and modular decomposition. "
    elif avg_complexity > 7:
        summary_text += f"Average complexity of {avg_complexity:.1f} indicates moderate architectural complexity requiring strategic attention. "
    else:
        summary_text += f"Average complexity of {avg_complexity:.1f} shows well-designed architectural structure with appropriate modularity. "
    
    if len(large_funcs) > total_functions * 0.3:
        summary_text += "Modularization strategy needed to address monolithic function patterns."
    else:
        summary_text += "Function sizing generally adheres to good architectural principles."
    
    return {
        "summary": summary_text,
        "function_feedback": architectural_feedback,
        "general_recommendations": strategic_recs[:5],
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
