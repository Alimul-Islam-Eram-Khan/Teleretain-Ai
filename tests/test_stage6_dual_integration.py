"""End-to-end mocked chatbot: Gemini 429 -> Groq; no external API used."""
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.chatbot as cb
from src import provider_failover as pf

class QuotaError(Exception):
    status_code = 429


def setup(monkeypatch, *, bad_reply=False):
    pf.clear_cooldowns_for_testing()
    monkeypatch.setenv('GEMINI_FREE_TIER_CONFIRMED','true')
    monkeypatch.setenv('GROQ_FREE_TIER_CONFIRMED','true')
    monkeypatch.setenv('GEMINI_API_KEY','fake-gemini-key')
    monkeypatch.setenv('GROQ_API_KEY','fake-groq-key')
    monkeypatch.setenv('COPILOT_PRIMARY','gemini')
    monkeypatch.delenv('GEMINI_MODEL',raising=False)
    monkeypatch.delenv('GROQ_MODEL',raising=False)
    verified = {
        'track':'iranian', 'data_source_label': cb.LABELS['iranian'],
        'data_source_description': 'Public dataset',
        'selected_model':'random_forest', 'churn_probability':0.73,
        'decision_threshold':0.48, 'predicted_class':1,
        'predicted_label':'Churn', 'disclaimer':'Not a Teletalk customer',
    }
    monkeypatch.setattr(cb,'predict_customer',lambda req: SimpleNamespace(model_dump=lambda:verified))
    monkeypatch.setattr(cb,'missing_fields_for',lambda *args:[])
    messages_mod=ModuleType('langchain_core.messages')
    class Msg:
        def __init__(self, content, **kw): self.content=content
    messages_mod.HumanMessage=Msg
    messages_mod.SystemMessage=Msg
    messages_mod.ToolMessage=Msg
    monkeypatch.setitem(sys.modules, 'langchain_core.messages',messages_mod)
    tool_mod=ModuleType('langchain_core.tools')
    class FakeTool:
        def __init__(self, fn):
            self.name=fn.__name__
            self.fn=fn
        def invoke(self, args):
            assert args=={}
            return self.fn()
    tool_mod.tool=lambda f: FakeTool(f)
    monkeypatch.setitem(sys.modules, 'langchain_core.tools', tool_mod)
    calls=[]
    gemini_mod=ModuleType('langchain_google_genai')
    class Gemini:
        def __init__(self, **kw):
            assert kw['model']=='gemini-3.8-flash'
        def invoke(self, messages):
            calls.append('gemini')
            if bad_reply:
                return SimpleNamespace(content='Please buy Tk 999 offer')
            raise QuotaError('test 429')
    gemini_mod.ChatGoogleGenerativeAI=Gemini
    monkeypatch.setitem(sys.modules,'langchain_google_genai',gemini_mod)
    groq_mod=ModuleType('langchain_groq')
    class Groq:
        def __init__(self, **kw):
            assert kw['model']=='openai/gpt-oss-20b'
            assert kw['max_retries']==0
        def invoke(self, messages):
            calls.append('groq')
            assert all('secret_feature' not in str(m.content) for m in messages)
            return SimpleNamespace(content='This is a public dataset model estimate, not causal evidence.')
    groq_mod.ChatGroq=Groq
    monkeypatch.setitem(sys.modules,'langchain_groq',groq_mod)
    return calls


def test_gemini_quota_fallback_groq_with_verified_prediction(monkeypatch):
    calls=setup(monkeypatch)
    out=cb.ask_copilot('Please explain',track='iranian',features={'secret_feature':123},live=True)
    assert calls==['gemini','groq']
    assert out.mode=='live'
    assert out.provider=='groq'
    assert out.model_id=='openai/gpt-oss-20b'
    assert '73.00%' in out.message and cb.LABELS['iranian'] in out.message


def test_unsafe_gemini_text_is_offline_not_groq(monkeypatch):
    calls=setup(monkeypatch, bad_reply=True)
    out=cb.ask_copilot('Explain',track='iranian',features={'secret_feature':123},live=True)
    assert calls==['gemini']
    assert out.mode=='offline' and out.provider is None
    assert '73.00%' in out.message
