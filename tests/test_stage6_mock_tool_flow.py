"""Exercise LangChain/Gemini message-tool orchestration with in-process fakes.

No network calls, secrets, or customer fixture records used.
"""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.chatbot as cb


def test_single_authorized_tool_call_and_grounded_message(monkeypatch):
    observed = {"tools": [], "messages": []}

    def fake_predict(request):
        assert request["track"] == "iranian"
        assert request["features"] == {"public_test_field": 1}
        return SimpleNamespace(model_dump=lambda: {
            "track": "iranian",
            "data_source_label": cb.LABELS["iranian"],
            "data_source_description": "Iranian public source",
            "selected_model": "random_forest",
            "churn_probability": 0.73,
            "decision_threshold": 0.48,
            "predicted_class": 1,
            "predicted_label": "Churn",
            "disclaimer": "Not Teletalk subscriber data",
        })
    monkeypatch.setattr(cb, "predict_customer", fake_predict)
    monkeypatch.setattr(cb, "missing_fields_for", lambda track, features: [])
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-local-only")
    monkeypatch.setenv("GEMINI_FREE_TIER_CONFIRMED", "true")

    messages = ModuleType("langchain_core.messages")
    class Message:
        def __init__(self, content, **kwargs):
            self.content = content
            self.extra = kwargs
    messages.HumanMessage = Message
    messages.SystemMessage = Message
    messages.ToolMessage = Message

    tools = ModuleType("langchain_core.tools")
    class FakeTool:
        def __init__(self, fn):
            self.name = fn.__name__
            self.fn = fn
        def invoke(self, args):
            assert args == {}
            return self.fn()
    tools.tool = lambda fn: FakeTool(fn)

    gemini = ModuleType("langchain_google_genai")
    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            assert kwargs["model"] == cb.DEFAULT_GEMINI_MODEL
        def bind_tools(self, available, tool_choice):
            assert len(available) == 1
            assert available[0].name == "predict_iranian"
            assert tool_choice == "predict_iranian"
            observed["tools"] = available
            return self
        def invoke(self, content):
            observed["messages"].append(content)
            if len(observed["messages"]) == 1:
                # The model can only invoke the exactly matching no-argument tool.
                return SimpleNamespace(tool_calls=[{"name": "predict_iranian", "args": {}, "id": "call_1"}])
            assert isinstance(content[-1], Message)
            # Only verified structured Stage 5 output, not raw features, is in this message.
            assert "public_test_field" not in content[-1].content
            return SimpleNamespace(content="This is a public dataset model estimate, not a causal conclusion.")
    gemini.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI

    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", gemini)
    reply = cb.ask_copilot("Explain this", track="iranian", features={"public_test_field": 1}, live=True, agent_tools=True)
    assert reply.mode == "live"
    assert "73.00%" in reply.message
    assert "This is a public dataset model estimate" in reply.message
    assert len(observed["messages"]) == 2
