"""Check stage-1 project environment without loading or inventing customer data."""
import importlib.util
import sys

packages = {
    "pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
    "xgboost": "xgboost", "shap": "shap", "matplotlib": "matplotlib",
    "seaborn": "seaborn", "plotly": "plotly", "streamlit": "streamlit",
    "joblib": "joblib", "pydantic": "pydantic", "pytest": "pytest",
    "dotenv": "python-dotenv",
}
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
if sys.version_info[:2] != (3, 12):
    print("WARNING: Use the Python 3.12 virtual environment for this project.")
missing = []
for module, pkg in packages.items():
    ok = importlib.util.find_spec(module) is not None
    print(f"{'OK' if ok else 'MISSING'}: {pkg}")
    if not ok:
        missing.append(pkg)
if missing:
    print("Install missing packages with: python -m pip install -r requirements-ml.txt")
    raise SystemExit(1)
print("Stage-1 environment checks passed. No datasets or models checked yet.")
