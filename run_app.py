"""Simple launcher for the Streamlit UI.

Usage:
	python run_app.py  # attempts to launch the Streamlit dashboard

This script assumes you have streamlit installed in the active environment.
"""

import os
import sys
import subprocess


def main() -> None:
	"""Locate the dashboard and launch Streamlit to run it.

	Exits with non-zero status if the dashboard file isn't found or Streamlit
	fails to start.
	"""
	dashboard_path = os.path.join("app", "ui", "dashboard.py")
	if not os.path.exists(dashboard_path):
		print(
			f"Dashboard not found at {dashboard_path}. Ensure you are in the project root."
		)
		sys.exit(1)

	# Launch Streamlit
	cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
	try:
		subprocess.check_call(cmd)
	except subprocess.CalledProcessError as e:
		print("Failed to start Streamlit:", e)
		sys.exit(e.returncode)


if __name__ == "__main__":
	main()