"""Configuration and constants for CodeSage"""
"""
Centralized configuration values for CodeSage.
Keep values simple and environment-agnostic for the MVP.
"""
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'


# Model config (placeholders for now)
MODEL_NAME = 'Salesforce/codet5-small' # default model; replace if desired


# Streamlit
STREAMLIT_PAGE_TITLE = 'CodeSage - AI Code Reviewer (MVP)'
