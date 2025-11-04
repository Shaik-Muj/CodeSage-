# CodeSage — AST Utilities
"""
Parses Python files to extract function-level structure,
docstring presence, and cyclomatic complexity using AST.
"""

import ast
from pathlib import Path
from radon.complexity import cc_visit


def extract_code_structure(path: str | Path):
    """
    Extracts structural information (functions, docstrings, complexity)
    from the given Python file.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")

    # Parse AST
    tree = ast.parse(source)

    # Collect function-level details
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = getattr(node, 'lineno', None)
            end_line = getattr(node, 'end_lineno', None) or start_line
            docstring = ast.get_docstring(node)
            functions.append({
                "name": node.name,
                "lines": (end_line - start_line + 1) if start_line else 0,
                "has_docstring": bool(docstring),
            })

    # Add complexity data
    complexities = cc_visit(source)
    complexity_map = {c.name: c.complexity for c in complexities}

    for f in functions:
        f["complexity"] = complexity_map.get(f["name"], 0)

    return functions


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python app/utils/ast_utils.py <file.py>")
        sys.exit(1)

    data = extract_code_structure(sys.argv[1])
    print(json.dumps(data, indent=2))
