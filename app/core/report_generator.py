"""
Combines AST-based structural data and static analysis results
into a single, standardized code quality report.
This report is later consumed by the LLM feedback module.
"""

import json
from pathlib import Path
from datetime import datetime

from app.utils.ast_utils import extract_code_structure
from app.utils.analyzer_utils import analyze_file


def generate_code_report(path: str | Path) -> dict:
    """
    Generate a combined analysis report for a given Python file.

    Returns a structured dictionary with:
    - File metadata
    - AST structure summary
    - Static analyzer results
    - Overall summary metrics
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Run both utilities
    ast_data = extract_code_structure(path)
    analyzer_data = analyze_file(path)

    # Compute summary stats
    total_functions = len(ast_data)
    avg_complexity = (
        sum([f["complexity"] for f in ast_data]) / total_functions
        if total_functions else 0
    )

    pylint_issues = analyzer_data.get("pylint", {}).get("issue_count", 0)
    bandit_issues = analyzer_data.get("bandit", {}).get("issue_count", 0)
    maintainability = analyzer_data.get("radon", {}).get("avg_maintainability", 100)

    report = {
        "file": str(path.name),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_functions": total_functions,
            "avg_complexity": round(avg_complexity, 2),
            "maintainability_index": maintainability,
            "security_issues": bandit_issues,
            "lint_issues": pylint_issues,
            "overall_score": compute_overall_score(
                avg_complexity, maintainability, pylint_issues + bandit_issues
            ),
        },
        "details": {
            "ast_analysis": ast_data,
            "static_analysis": analyzer_data,
        },
    }

    return report


def compute_overall_score(avg_complexity: float, maintainability: float, issues: int) -> float:
    """
    Compute an overall quality score (0–100) based on simple weighted metrics.
    """
    # Lower complexity + higher maintainability + fewer issues = better score
    score = (
        (100 - min(avg_complexity * 5, 50)) * 0.4 +  # complexity weight
        (maintainability * 0.4) +                    # maintainability weight
        (max(0, 100 - issues * 3) * 0.2)             # issue penalty
    )
    return round(min(max(score, 0), 100), 2)


def summarize_report(report: dict) -> str:
    """
    Generate a short textual summary of the report.
    Used for display in terminal or Streamlit app.
    """
    s = report["summary"]
    return (
        f"File: {report['file']}\n"
        f"Functions: {s['total_functions']}\n"
        f"Avg Complexity: {s['avg_complexity']}\n"
        f"Maintainability Index: {s['maintainability_index']}\n"
        f"Security Issues: {s['security_issues']}\n"
        f"Lint Issues: {s['lint_issues']}\n"
        f"Overall Quality Score: {s['overall_score']}/100"
    )


def save_report(report: dict, output_path: str | Path) -> Path:
    """
    Save the combined report as a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python app/core/report_generator.py <file.py>")
        sys.exit(1)

    file_path = sys.argv[1]
    report = generate_code_report(file_path)
    print(summarize_report(report))

    save_path = Path("reports") / (Path(file_path).stem + "_report.json")
    save_report(report, save_path)
    print(f"\nReport saved to: {save_path}")