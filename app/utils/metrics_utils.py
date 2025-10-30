"""Simple code metrics helpers (placeholders)."""


def cyclomatic_complexity_placeholder(source: str) -> int:
    """Return a tiny placeholder metric for complexity."""
    return source.count("if") + source.count("for") + source.count("while")
