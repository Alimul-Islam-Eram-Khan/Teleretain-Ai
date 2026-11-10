"""Preflight: no API call and no secrets printed."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib.util
from scripts.stage5_samples import validation_request
from src.chatbot import ask_copilot


def main():
    required = ("langchain_core", "langchain_google_genai", "langchain_groq", "dotenv")
    missing = [x for x in required if importlib.util.find_spec(x) is None]
    if missing:
        raise SystemExit(f"Missing Stage 6 dependencies: {missing}. Run: python -m pip install -r requirements-stage6.txt")
    from langchain_google_genai import ChatGoogleGenerativeAI  # pylint: disable=unused-import
    from langchain_groq import ChatGroq  # pylint: disable=unused-import
    print("PASS: LangChain/Gemini/Groq Python integrations import successfully")
    for track in ("iranian", "ibm"):
        req = validation_request(track)
        out = ask_copilot("Explain this prediction", track=track, features=req["features"], live=False)
        assert out.result and out.mode == "offline" and out.result["track"] == track
        print(f"PASS: {track} | validated prediction & source-grounded offline explanation | model={out.result['selected_model']}")
    print("SUCCESS: Stage 6 local preflight passed (no external LLM API call made).")


if __name__ == "__main__":
    main()
