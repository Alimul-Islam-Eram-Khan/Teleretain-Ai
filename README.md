# TeleRetain AI — Starter (Stage 1 only)

**Status: Environment/setup starter, NOT the finished TeleRetain AI project.**
No datasets, trained models, predictions, SHAP results, verified Teletalk offers, or Groq credentials are included. Work through the implementation guide step by step.

## Core rules
- Use public **UCI Iranian Churn** as the primary track and public **IBM Telco Customer Churn** as a separate benchmark.
- Never route Iranian columns to an IBM model or IBM columns to an Iranian model.
- The AI assistant explains validated outputs; it must not invent package names/prices or imply real Teletalk subscriber predictions.
- The RAG knowledge base must use only verified public Teletalk information with source URLs and retrieval dates.
- No real Teletalk customer records and no synthetic customer records.

## Windows setup (PowerShell in VS Code)

```powershell
py install 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
python scripts/check_environment.py
python -m streamlit run app/streamlit_app.py
```

Later, when implementing the LLM and RAG stages:

```powershell
python -m pip install -r requirements-ai.txt
```

Download datasets yourself from:
- Iranian: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- IBM: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place extracted CSVs respectively under `data/raw/iranian/` and `data/raw/ibm/` (the zip deliberately does not include datasets).

## Implementation sequence
1. Environment and datasets.
2. EDA/preprocessing separately for Iranian and IBM.
3. Leakage-safe training, metrics, calibration, thresholds, and artifact saving separately.
4. SHAP explanations for both tracks.
5. Input validation, strict Pydantic schemas, and fail-closed routing (cross-track input rejected).
6. Groq/LangChain copilot that calls tools only after validated input, and supports English/Bangla/Banglish.
7. Official public Teletalk offer knowledge, FAISS search, citations, and deterministic offer filtering.
8. Complete Streamlit app and dashboard.
9. Unit/integration tests, technical report, README, GitHub, presentation.

`app/streamlit_app.py` is **only a setup smoke-test UI**. Features are placeholders until implemented with actual data and verified sources.
