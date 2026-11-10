"""No real API call and no customer fixture data. """
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src.chatbot as cb


def _fake_payload(t):
    return {"track": t, "data_source_label": cb.LABELS[t], "data_source_description": "public source",
            "selected_model": "random_forest", "churn_probability": 0.25,
            "decision_threshold": 0.48, "predicted_class": 0, "predicted_label": "No churn",
            "disclaimer": "Not Teletalk data"}


def test_live_gated_by_free_tier_acknowledgement(monkeypatch):
    monkeypatch.setattr(cb, "predict_customer", lambda req: SimpleNamespace(model_dump=lambda: _fake_payload(req["track"])))
    monkeypatch.setattr(cb, "missing_fields_for", lambda track, features: [])
    monkeypatch.delenv("GEMINI_FREE_TIER_CONFIRMED", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-real")
    answer=cb.ask_copilot("Explain", track="iranian", features={"mocked_input": 1}, live=True)
    assert answer.mode == "offline" and "Live LLM explanation unavailable" in answer.message


def test_response_text_content_blocks():
    msg=SimpleNamespace(content=[{"type":"reasoning","text":"private"}, {"type":"text","text":"Public explanation."}])
    assert cb._response_text(msg) == "Public explanation."


def test_unapproved_model_is_not_allowlisted():
    assert "gemini-4-argon" not in cb.ALLOWED_FREE_MODELS
    assert "gemini-3.8-flash" in cb.ALLOWED_FREE_MODELS
