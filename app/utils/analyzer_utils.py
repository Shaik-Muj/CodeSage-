# CodeSage — Static Analyzer Utilities
"""
Analyzer utilities for static analysis using Pylint, Bandit, and Radon.
Runs open-source tools locally to evaluate code quality, maintainability,
and security vulnerabilities.
"""
import json
import subprocess
from pathlib import Path
from radon.metrics import mi_visit


def run_subprocess(cmd: list[str]):
    """Run a command and capture stdout/stderr as text."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1


# -----------------------------
# PYLINT
# -----------------------------
def analyze_with_pylint(path: str | Path):
    """Run pylint on the target file and return its output as JSON."""
    cmd = [
        'pylint',
        '--output-format=json',
        '--disable=R,C',  # disable refactor & convention warnings for brevity
        str(path)
    ]
    stdout, stderr, code = run_subprocess(cmd)

    if stderr:
        return {'tool': 'pylint', 'error': stderr}

    try:
        data = json.loads(stdout) if stdout else []
    except json.JSONDecodeError:
        data = []

    issues = [{
        'type': item.get('type'),
        'message': item.get('message'),
        'symbol': item.get('symbol'),
        'line': item.get('line')
    } for item in data]

    return {
        'tool': 'pylint',
        'exit_code': code,
        'issue_count': len(issues),
        'issues': issues
    }


# -----------------------------
# BANDIT
# -----------------------------
def analyze_with_bandit(path: str | Path):
    """Run bandit for security analysis."""
    cmd = [
        'bandit',
        '-f', 'json',
        '-q',  # quiet mode
        str(path)
    ]
    stdout, stderr, code = run_subprocess(cmd)

    if stderr:
        return {'tool': 'bandit', 'error': stderr}

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        data = {'results': []}

    results = [{
        'filename': item.get('filename'),
        'line_number': item.get('line_number'),
        'issue_severity': item.get('issue_severity'),
        'issue_text': item.get('issue_text'),
        'test_id': item.get('test_id')
    } for item in data.get('results', [])]

    return {
        'tool': 'bandit',
        'exit_code': code,
        'issue_count': len(results),
        'issues': results
    }


# -----------------------------
# RADON
# -----------------------------
def analyze_with_radon(path: str | Path):
    """Compute Maintainability Index using Radon."""
    try:
        source = Path(path).read_text(encoding='utf-8')
        score = mi_visit(source, multi=True)
        avg_score = sum([s.mi for s in score]) / len(score) if score else 100
        return {
            'tool': 'radon',
            'avg_maintainability': round(avg_score, 2),
            'details': [s._asdict() for s in score]
        }
    except Exception as e:
        return {'tool': 'radon', 'error': str(e)}


# -----------------------------
# COMBINED ENTRY
# -----------------------------
def analyze_file(path: str | Path):
    """Run all static analyzers and return combined results."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    results = {
        'path': str(path),
        'pylint': analyze_with_pylint(path),
        'bandit': analyze_with_bandit(path),
        'radon': analyze_with_radon(path)
    }
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python app/utils/analyzer_utils.py <file.py>')
        sys.exit(1)

    data = analyze_file(sys.argv[1])
    print(json.dumps(data, indent=2))
