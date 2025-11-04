# CodeSage — Local AI Feedback (Heuristic Mode)
"""
This module generates human-like AI feedback from static analysis reports,
without using any external APIs or paid models.

The goal: simulate intelligent review using local logic, scoring, and phrasing
to provide dynamic feedback and actionable recommendations.
"""

import random
import json


# -------------------------------------
# Feedback Generators
# -------------------------------------

def generate_summary_text(summary: dict) -> str:
    """Generate a natural summary of the overall metrics."""
    score = summary.get("overall_score", 0)

    if score >= 90:
        return "Outstanding code quality — clean, efficient, and very well structured."
    elif score >= 75:
        return "Good code quality overall. Just minor improvements needed in readability or structure."
    elif score >= 60:
        return "Average quality — consider simplifying logic and improving maintainability."
    elif score >= 40:
        return "Below-average quality — issues with complexity or structure should be addressed."
    else:
        return "Code quality appears low — extensive refactoring and cleanup are recommended."



def generate_function_feedback(ast_data: list[dict]) -> list[str]:
    """Generate feedback for individual functions based on complexity."""
    if not ast_data:
        return ["No function-level information found."]

    feedback = []
    for func in ast_data:
        name = func.get("name", "unknown_function")
        comp = func.get("complexity", 0)

        if comp > 15:
            feedback.append(f"Function '{name}' is highly complex (complexity {comp}). Consider breaking it into smaller parts.")
        elif comp > 10:
            feedback.append(f"'{name}' could be simplified — its complexity of {comp} suggests too much nested logic.")
        elif comp > 5:
            feedback.append(f"'{name}' has moderate complexity ({comp}). Adding comments could improve clarity.")
        else:
            feedback.append(f"'{name}' looks clean and simple — maintain this level of readability.")
    
    return feedback


def generate_general_recommendations(report: dict) -> list[str]:
    """Generate general advice based on analysis results."""
    analysis = report.get("details", {}).get("analysis", {})
    pylint_issues = analysis.get("pylint", {}).get("issue_count", 0)
    bandit_issues = analysis.get("bandit", {}).get("issue_count", 0)
    maintainability = analysis.get("radon", {}).get("avg_maintainability", 100)

    recs = []

    if pylint_issues > 5:
        recs.append("Resolve linting warnings from Pylint to improve code readability and style consistency.")
    elif pylint_issues > 0:
        recs.append("Fix remaining linting issues to ensure clean and maintainable code.")

    if bandit_issues > 0:
        recs.append("Address potential security vulnerabilities flagged by Bandit.")
    
    if maintainability < 70:
        recs.append("Improve maintainability by reducing nested loops and adding docstrings.")

    if not recs:
        recs.append("Your code meets high standards of clarity, security, and maintainability.")
    
    return recs


# -------------------------------------
# Core Interface
# -------------------------------------

def generate_feedback(report: dict, mode: str = "heuristic") -> dict:
    """
    Generate complete feedback from report data (local mode only).
    Uses the overall_score computed in report_generator.py.
    """
    if not report or "summary" not in report:
        raise ValueError("Invalid report format. Expected structured output from report_generator.py")

    summary = report["summary"]
    ast_data = report.get("details", {}).get("ast_analysis", [])
    
    # Use score from report_generator instead of recomputing
    quality_score = summary.get("overall_score", 0)

    summary_text = generate_summary_text(summary)
    func_feedback = generate_function_feedback(ast_data)
    recs = generate_general_recommendations(report)

    # Add some mild randomization to phrasing (so it feels AI-ish)
    random.shuffle(func_feedback)
    random.shuffle(recs)

    return {
        "score": quality_score,
        "summary": summary_text,
        "function_feedback": func_feedback[:5],  # limit to top 5
        "general_recommendations": recs,
    }


# -------------------------------------
# Save/Load
# -------------------------------------

def save_feedback(feedback: dict, output_path: str):
    """Save AI feedback as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2)
    return output_path


# -------------------------------------
# CLI Entry
# -------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from app.core.report_generator import generate_code_report

    if len(sys.argv) < 2:
        print("Usage: python app/ai_engine/llm_feedback.py <target_file.py>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    report = generate_code_report(file_path)
    feedback = generate_feedback(report)

    print(json.dumps(feedback, indent=2))
