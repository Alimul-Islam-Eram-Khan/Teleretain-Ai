"""Offline mock of the 1-request LangChain local-tool economy flow; no network/API key."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.chatbot as cb


def test_economy_one_llm_request_after_local_validation(monkeypatch):
    observed = {"calls": 0, "tools": 0}
    expected = {
        "track": "iranian",
        "data_source_label": cb.LABELS["iranian"],
        "data_source_description": "public Iranian source",
        "selected_model": "random_forest",
        "churn_probability": 0.73,
        "decision_threshold": 0.48,
        "predicted_class": 1,
        "predicted_label": "Churn",
        "disclaimer": "Not a Teletalk subscriber prediction",
    }

    def fake_predict(request):
        assert request == {"track": "iranian", "features": {"secret_input_name": 37}}
        return SimpleNamespace(model_dump=lambda: dict(expected))

    monkeypatch.setattr(cb, "predict_customer", fake_predict)
    monkeypatch.setattr(cb, "missing_fields_for", lambda track, f: [])
    monkeypatch.setenv("GEMINI_API_KEY", "not-real-only-in-local-test")
    monkeypatch.setenv("GEMINI_FREE_TIER_CONFIRMED", "true")

    fake_messages = ModuleType("langchain_core.messages")

    class Message:
        def __init__(self, content, **kw):
            self.content = content

    fake_messages.HumanMessage = Message
    fake_messages.SystemMessage = Message

    fake_tools = ModuleType("langchain_core.tools")

    class FakeTool:
        def __init__(self, fn):
            self.name = fn.__name__
            self.fn = fn

        def invoke(self, args):
            assert args == {}
            observed["tools"] += 1
            return self.fn()

    fake_tools.tool = lambda fn: FakeTool(fn)

    fake_gemini = ModuleType("langchain_google_genai")

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kw):
            assert kw["model"] == cb.DEFAULT_GEMINI_MODEL
            assert kw["max_retries"] == 0
            assert kw["max_output_tokens"] <= 1024
            assert kw["vertexai"] is False

        def invoke(self, messages):
            observed["calls"] += 1
            assert len(messages) == 2
            assert "secret_input_name" not in str(messages[1].content)
            assert "73" not in str(messages[0].content)
            assert "random_forest" in messages[1].content
            return SimpleNamespace(content="This is a public dataset estimate. It cannot establish cause.")

    fake_gemini.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
    monkeypatch.setitem(sys.modules, "langchain_core.messages", fake_messages)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_gemini)

    answer = cb.ask_copilot("Explain the result", track="iranian", features={"secret_input_name": 37}, live=True)
    assert answer.mode == "live"
    assert answer.source_label == cb.LABELS["iranian"]
    assert "73.00%" in answer.message
    assert observed == {"calls": 1, "tools": 1}


def test_economy_hallucination_rejected(monkeypatch):
    dummy = {
        "track": "ibm", "data_source_label": cb.LABELS["ibm"],
        "data_source_description": "public", "selected_model": "xgboost",
        "churn_probability": .15, "decision_threshold": .31,
        "predicted_class": 0, "predicted_label": "No churn", "disclaimer": "Not Teletalk",
    }
    monkeypatch.setattr(cb, "predict_customer", lambda request: SimpleNamespace(model_dump=lambda: dict(dummy)))
    monkeypatch.setattr(cb, "missing_fields_for", lambda t, f: [])
    monkeypatch.setattr(cb, "_economy_explanation", lambda *args: "Purchase a Tk 99 Teletalk offer")
    # This is the helper's semantic output test: runtime path checks the live model's output.
    assert not cb._is_safe_llm_text("Purchase a Tk 99 Teletalk offer")
