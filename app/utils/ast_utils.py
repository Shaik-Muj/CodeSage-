"""
AST utilities for analyzing Python source code.
Provides helpers to extract function metadata, docstring coverage,
and complexity metrics.
"""
import ast
from pathlib import Path
from radon.complexity import cc_visit
from radon.visitors import ComplexityVisitor

def parse_code(source: str):
	"""Parse Python code to an AST node."""
	try:
		return ast.parse(source)
	except SyntaxError as e:
		raise ValueError(f"Invalid Python syntax: {e}")



def get_function_info(source: str):
	"""Extract functions and metadata from Python source code."""
	tree = parse_code(source)
	results = []


	for node in ast.walk(tree):
		if isinstance(node, ast.FunctionDef):
			func_name = node.name
			docstring = ast.get_docstring(node)
			lineno = getattr(node, 'lineno', None)
			end_lineno = getattr(node, 'end_lineno', None)
			lines = (end_lineno - lineno + 1) if (lineno and end_lineno) else None


			results.append({
				'function': func_name,
				'has_docstring': bool(docstring),
				'lineno': lineno,
				'lines': lines
			})


	return results



def get_cyclomatic_complexity(source: str):
	"""Compute cyclomatic complexity for functions in the given code."""
	results = []
	try:
		blocks = cc_visit(source)
		for block in blocks:
			results.append({
				'name': block.name,
				'complexity': block.complexity,
				'lineno': block.lineno
			})
	except Exception as e:
		results.append({'error': str(e)})
	return results



def analyze_file(path: str | Path):
	"""Read a file and return combined AST + complexity info."""
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(path)


	source = p.read_text(encoding='utf-8')
	info = get_function_info(source)
	complexity = get_cyclomatic_complexity(source)


	return {
		'path': str(path),
		'functions': info,
		'complexity': complexity
	}



if __name__ == '__main__':
	import json
	import sys
	if len(sys.argv) < 2:
		print('Usage: python app/utils/ast_utils.py <file.py>')
		sys.exit(1)
	data = analyze_file(sys.argv[1])
	print(json.dumps(data, indent=2))
data = analyze_file(sys.argv[1])
print(json.dumps(data, indent=2))