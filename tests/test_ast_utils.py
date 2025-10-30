import pytest
from app.utils.ast_utils import parse_source, count_functions


def test_count_functions():
    src = """\ndef a():\n    pass\n\ndef b():\n    pass\n"""
    tree = parse_source(src)
    assert count_functions(tree) >= 2
