
# CodeSage — Open-Source AI Code Reviewer

CodeSage is an open-source, fully local AI-powered code review assistant. It combines
traditional static analysis with lightweight open-source language models and agentic
reasoning to automatically evaluate code quality, style, complexity, and maintainability —
without relying on any paid APIs or cloud services.

Key goals:
- Run entirely offline with no external API calls
- Combine static analyzers (pylint, bandit, radon) with LLM-based critique (CodeT5/StarCoder)
- Provide a lightweight Streamlit dashboard for interactive reviews
- Offer a configurable, rule-based Code Quality Index

Table of contents
- Features
- Project structure
- Installation
- Quick start
- Usage
- Development
- Roadmap
- License
- Contributing

Features
--------
- Automated static code analysis using pylint, bandit, and radon
- AI-based feedback powered by open-source models (CodeT5 / StarCoder — optional)
- Code structure analysis using Python’s AST (ast)
- Multi-agent system for syntax, security, and quality evaluation (agentic patterns)
- Streamlit dashboard for visualization and interactive review
- Configurable rule-based scoring system (Code Quality Index)

Project structure
-----------------
```
codesage/
│
├── app/
│   ├── main.py                # Entry point for Streamlit or FastAPI
│   ├── config.py              # Configuration and settings
│   ├── utils/                 # Helper functions
│   ├── ai_engine/             # AI models, agents, and reasoning layer
│   ├── core/                  # Core review orchestration logic
│   ├── ui/                    # Streamlit UI components
│   └── data/                  # Sample or demo code files
│
├── tests/                     # Unit and integration tests
├── notebooks/                 # Experiments and prototypes
├── requirements.txt           # Dependencies
├── run_app.py                 # Quick launcher
└── README.md                  # Project documentation
```

Installation
------------
Clone the repository and create a virtual environment.

Note: commands shown for both Unix-like shells and Windows PowerShell where appropriate.

```bash
git clone https://github.com/yourusername/codesage.git
cd codesage

# create a venv (cross-platform)
python -m venv .venv

# On macOS / Linux
source .venv/bin/activate

# On Windows PowerShell
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

Quick start
-----------
Start the Streamlit UI (recommended for interactive use):

```powershell
python -m streamlit run app\ui\dashboard.py
# OR via the launcher
python run_app.py
```

Running tests
-------------
Run the test suite with pytest:

```bash
pytest
```

Usage
-----
- Upload a single Python file through the Streamlit UI to run a lightweight static analysis and request AI feedback.
- Use the sidebar actions to trigger static analysis and AI critique (placeholders in the scaffold).

Development notes
-----------------
- The UI is implemented with Streamlit under `app/ui/`.
- Static-analysis helpers live in `app/utils/` (AST parsing, radon wrappers, lint integration).
- AI model integration belongs in `app/ai_engine/` (model loader, embeddings, LLM critique functions).
- Core orchestration is in `app/core/` (review manager, scoring).

Recommended developer workflow
1. Create a branch for your feature: `git checkout -b feat/your-feature`
2. Run tests, implement small focused changes and open a PR.

Making the project runnable
- To enable radon-based cyclomatic complexity, add `radon` to `requirements.txt`.
- To enable LLM-based feedback locally, install `transformers` and a small code model (note: models may be large).

Roadmap
-------
MVP v1.0
- Single-file Python analysis
- AST parsing + pylint + LLM critique
- Streamlit dashboard for results

v2.0
- Multi-agent feedback (syntax, security, readability)
- Code quality scoring and visualization

v3.0
- Multi-language support
- GitHub Actions integration for automated PR reviews

Contributing
------------
Contributions are welcome. Please open an issue to discuss larger changes before implementing.

1. Fork the repository
2. Create a feature branch
3. Open a pull request with a clear description and tests (where applicable)

Authors & Acknowledgements
--------------------------
Developed by S. Mujtaba Hussain. Open to collaboration and contributions.

Acknowledgements
- Portions inspired by open-source linting and code-quality tooling.

Contact
-------
If you'd like to collaborate or have questions, open an issue or contact the maintainer through the GitHub repository.

---


