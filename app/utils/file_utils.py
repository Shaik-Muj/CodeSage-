"""File and path helper utilities"""
import os
from typing import List


def list_py_files(path: str) -> List[str]:
    """Return a list of .py files under path (non-recursive)."""
    try:
        return [f for f in os.listdir(path) if f.endswith('.py')]
    except FileNotFoundError:
        return []
