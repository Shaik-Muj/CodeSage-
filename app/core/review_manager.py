"""Coordinate AI + static analyzers to produce reviews."""
from ..utils.ast_utils import parse_source
from ..ai_engine.llm_feedback import critique_code_placeholder


def review_source(source: str) -> dict:
    tree = parse_source(source)
    critique = critique_code_placeholder(source)
    return {"critique": critique, "functions": sum(1 for _ in [n for n in []])}
