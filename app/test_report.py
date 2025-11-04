from app.core.report_generator import generate_code_report
import json

report = generate_code_report("sample_code.py")
print(json.dumps(report, indent=2))
