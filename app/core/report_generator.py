"""
CodeSage - Report Generator (Refactored)
----------------------------------------
Combines AST-based structural data and static analysis results
into a unified report consumed by the AI Feedback Engine (L1–L3).
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from app.utils.ast_utils import extract_code_structure
from app.utils.analyzer_utils import analyze_file


# =========================================================
# Core: Report Generation
# =========================================================

def generate_code_report(path: str | Path) -> Dict[str, Any]:
    """
    Generate a combined code quality report for a given Python file.

    Returns a structured dictionary with:
    - File metadata
    - AST structure summary
    - Static analyzer results
    - Overall summary metrics (complexity, maintainability, issues, score)
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"❌ File not found: {path}")

    # ---------------------------------------------------------
    # Run Static + Structural Analysis
    # ---------------------------------------------------------
    try:
        ast_data = extract_code_structure(path)
    except Exception as e:
        print(f"[WARN] AST parsing failed for {path.name}: {e}")
        ast_data = []

    try:
        analyzer_data = analyze_file(path)
    except Exception as e:
        print(f"[WARN] Static analysis failed for {path.name}: {e}")
        analyzer_data = {}

    # ---------------------------------------------------------
    # Compute Derived Metrics
    # ---------------------------------------------------------
    total_functions = len(ast_data)
    avg_complexity = (
        sum(f.get("complexity", 0) for f in ast_data) / total_functions
        if total_functions else 0
    )

    pylint_issues = analyzer_data.get("pylint", {}).get("issue_count", 0)
    bandit_issues = analyzer_data.get("bandit", {}).get("issue_count", 0)
    maintainability = analyzer_data.get("radon", {}).get("avg_maintainability", 100)

    overall_score = compute_overall_score(
        avg_complexity=avg_complexity,
        maintainability=maintainability,
        issues=(pylint_issues + bandit_issues)
    )

    # ---------------------------------------------------------
    # Build Final Report
    # ---------------------------------------------------------
    report = {
        "file": str(path.name),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_functions": total_functions,
            "avg_complexity": round(avg_complexity, 2),
            "maintainability_index": round(maintainability, 2),
            "security_issues": bandit_issues,
            "lint_issues": pylint_issues,
            "overall_score": overall_score
        },
        "details": {
            "ast_analysis": ast_data,
            "static_analysis": analyzer_data
        },
    }

    return report


# =========================================================
# Helper: Overall Score Calculation
# =========================================================

def compute_overall_score(avg_complexity: float, maintainability: float, issues: int) -> float:
    """
    Compute a weighted overall quality score (0–100).

    Formula:
        - Lower complexity = better (40%)
        - Higher maintainability = better (40%)
        - Fewer issues = better (20%)
    """

    # Normalize and weight
    complexity_score = max(0, 100 - min(avg_complexity * 5, 50))
    maintain_score = max(0, maintainability)
    issue_penalty = max(0, 100 - (issues * 3))

    weighted_score = (
        complexity_score * 0.4 +
        maintain_score * 0.4 +
        issue_penalty * 0.2
    )

    return round(min(max(weighted_score, 0), 100), 2)


# =========================================================
# Utility: Text Summary
# =========================================================

def summarize_report(report: Dict[str, Any]) -> str:
    """Return a human-readable summary for console or dashboard logs."""
    s = report.get("summary", {})
    return (
        f"\n📄 Code Report for: {report.get('file', 'unknown')}\n"
        f"───────────────────────────────────────\n"
        f"Total Functions    : {s.get('total_functions', 0)}\n"
        f"Avg Complexity     : {s.get('avg_complexity', 0)}\n"
        f"Maintainability    : {s.get('maintainability_index', 0)}\n"
        f"Security Issues    : {s.get('security_issues', 0)}\n"
        f"Lint Issues        : {s.get('lint_issues', 0)}\n"
        f"Overall Score      : {s.get('overall_score', 0)}/100\n"
    )


# =========================================================
# Utility: Save Report
# =========================================================

def save_report(report: Dict[str, Any], output_path: str | Path) -> Path:
    """Save the generated report as a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path


# =========================================================
# CLI Entry Point
# =========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python app/core/report_generator.py <target_file.py>")
        sys.exit(1)

    target_file = Path(sys.argv[1])
    print(f"[INFO] Analyzing {target_file.name} ...")

    report = generate_code_report(target_file)
    print(summarize_report(report))

    save_path = Path("reports") / f"{target_file.stem}_report.json"
    save_report(report, save_path)
    print(f"✅ Report saved to: {save_path}")
