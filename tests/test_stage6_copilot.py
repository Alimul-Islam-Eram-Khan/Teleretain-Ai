"""Deterministic unit tests: no network, no Gemini key, no model pickle required."""
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest
import src.chatbot as cb


@pytest.fixture(autouse=True)
def fake_stage5(monkeypatch):
    def pred(request):
        if request['track'] == 'iranian' and set(request['features']) != {'x'}:
            raise ValueError('bad features')
        if request['track'] == 'ibm' and set(request['features']) != {'y'}:
            raise ValueError('bad features')
        return SimpleNamespace(model_dump=lambda: {
            'track': request['track'],
            'data_source_label': cb.LABELS[request['track']],
            'data_source_description': 'public dataset only',
            'selected_model': 'random_forest' if request['track'] == 'iranian' else 'xgboost',
            'churn_probability': 0.3,
            'decision_threshold': 0.48 if request['track'] == 'iranian' else 0.31,
            'predicted_class': 0,
            'predicted_label': 'No churn',
            'disclaimer': 'Public dataset proof of concept; not Teletalk.',
        })
    monkeypatch.setattr(cb, 'predict_customer', pred)
    monkeypatch.setattr(cb, 'missing_fields_for', lambda t, f: [] if ('x' if t == 'iranian' else 'y') in f else ['x' if t == 'iranian' else 'y'])


def test_offline_result_uses_verification():
    r = cb.ask_copilot('Explain my prediction', track='iranian', features={'x': 1})
    assert r.mode == 'offline' and '30.00%' in r.message and '0.48' in r.message
    assert cb.LABELS['iranian'] in r.message


def test_missing_no_llm(monkeypatch):
    monkeypatch.setattr(cb, '_economy_explanation', lambda *a: pytest.fail('LLM should not be called'))
    r = cb.ask_copilot('amar risk koto?', track='iranian', features={}, live=True)
    assert r.mode == 'needs_input' and 'x' in r.message


def test_extra_input_rejected_no_llm(monkeypatch):
    monkeypatch.setattr(cb, '_economy_explanation', lambda *a: pytest.fail('LLM should not be called'))
    r = cb.ask_copilot('Predict', track='iranian', features={'x': 1, 'Churn': 0}, live=True)
    assert r.mode == 'invalid'


def test_offer_refused_no_llm(monkeypatch):
    monkeypatch.setattr(cb, '_economy_explanation', lambda *a: pytest.fail('LLM should not be called'))
    r = cb.ask_copilot('Teletalk package price?', track='iranian', features={'x': 1}, live=True)
    assert r.mode == 'unavailable' and 'Stage 7' in r.message


def test_cross_track_input_rejected():
    r = cb.ask_copilot('Explain', track='ibm', features={'x': 1}, live=True)
    assert r.mode == 'needs_input'


def test_no_track_rejected():
    r = cb.ask_copilot('Explain', features={'x': 1})
    assert r.mode == 'needs_input' and r.result is None


def test_llm_untrusted_numbers_rejected():
    assert not cb._is_safe_llm_text('Model risk is 95%')
    assert not cb._is_safe_llm_text('Buy this Tk 99 package')
    assert not cb._is_safe_llm_text('Visit https://fake.example')
    assert cb._is_safe_llm_text('This is a noncausal public benchmark model estimate.')


def test_live_error_fallback(monkeypatch):
    def fail(*args):
        raise RuntimeError('key missing')
    monkeypatch.setattr(cb, '_economy_explanation', fail)
    r = cb.ask_copilot('Explain', track='ibm', features={'y': 1}, live=True)
    assert r.mode == 'offline' and r.result['track'] == 'ibm'
    assert 'Live LLM explanation unavailable' in r.message


def test_bangla_lang():
    assert cb._language('আমাকে বুঝিয়ে দাও') == 'Bangla'
    assert cb._language('amar churn keno?') == 'Banglish'
    assert cb._language('Explain this') == 'English'


def test_too_long_question_rejected():
    r=cb.ask_copilot("x"*601, track="iranian", features={"x": 1}, live=True)
    assert r.mode == "invalid" and r.result is None

def test_unknown_offer_terms_rejected():
    assert not cb._is_safe_llm_text("Please buy the cheap recharge offer")
    assert not cb._is_safe_llm_text("Get an internet package")
