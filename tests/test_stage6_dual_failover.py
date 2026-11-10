"""Provider failover tests: mocked HTTP and LLM responses; no customer data or keys."""
import os
from types import SimpleNamespace
import pytest
from src import provider_failover as pf


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__('dummy mocked network error')
        self.status_code = status_code


@pytest.fixture(autouse=True)
def reset(monkeypatch, tmp_path):
    pf.clear_cooldowns_for_testing()
    # Unit tests must not reload the user's real .env or API keys.
    # In production, configured_providers() still reads the real .env.
    monkeypatch.setattr(pf, "ROOT", tmp_path)
    for name in ('GEMINI_API_KEY', 'GROQ_API_KEY', 'GEMINI_FREE_TIER_CONFIRMED',
                 'GROQ_FREE_TIER_CONFIRMED','COPILOT_PRIMARY', 'GEMINI_MODEL', 'GROQ_MODEL'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('COPILOT_PRIMARY', 'gemini')


def configured(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-unit-test-key')
    monkeypatch.setenv('GROQ_API_KEY', 'fake-unit-test-key')
    monkeypatch.setenv('GEMINI_FREE_TIER_CONFIRMED', 'true')
    monkeypatch.setenv('GROQ_FREE_TIER_CONFIRMED', 'true')


def test_primary_success_no_backup_invoked(monkeypatch):
    configured(monkeypatch)
    calls = []
    monkeypatch.setattr(pf, 'create_llm', lambda p: p.name)
    def action(provider, client):
        calls.append(provider.name)
        return 'Grounded explanation.'
    r = pf.run_with_failover(action)
    assert (r.provider, r.model) == ('gemini', 'gemini-3.8-flash')
    assert calls == ['gemini']


def test_quota_triggers_backup(monkeypatch):
    configured(monkeypatch)
    calls = []
    monkeypatch.setattr(pf, 'create_llm', lambda p: p.name)
    def action(provider, client):
        calls.append(provider.name)
        if provider.name == 'gemini':
            raise FakeHTTPError(429)
        return 'Verified answer.'
    r = pf.run_with_failover(action)
    assert r.provider == 'groq'
    assert r.model == 'openai/gpt-oss-20b'
    assert calls == ['gemini', 'groq']
    # A second chat turn in this same process should bypass cooling-down Gemini.
    assert pf.run_with_failover(action).provider == 'groq'
    assert calls == ['gemini', 'groq', 'groq']


def test_reprobe_primary_after_cooldown(monkeypatch):
    configured(monkeypatch)
    clock = [300.0]
    monkeypatch.setattr(pf.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(pf, 'create_llm', lambda p: p.name)
    seen = []
    def action(provider, client):
        seen.append(provider.name)
        if len(seen) == 1:
            raise FakeHTTPError(429)
        return 'Model estimate.'
    assert pf.run_with_failover(action).provider == 'groq'
    clock[0] += pf.COOLDOWN_SECONDS + 1
    assert pf.run_with_failover(action).provider == 'gemini'
    assert seen == ['gemini','groq','gemini']


def test_provider_server_error_fails_over(monkeypatch):
    configured(monkeypatch)
    monkeypatch.setattr(pf, 'create_llm', lambda p: p.name)
    ans = pf.run_with_failover(lambda p, _ : (_ for _ in ()).throw(FakeHTTPError(503)) if p.name == 'gemini' else 'safe')
    assert ans.provider == 'groq'


@pytest.mark.parametrize('status', [400,401,403,404,422])
def test_permanent_errors_do_not_fallback(monkeypatch, status):
    configured(monkeypatch)
    called = []
    monkeypatch.setattr(pf, 'create_llm', lambda p: p.name)
    def action(p, _):
        called.append(p.name)
        raise FakeHTTPError(status)
    with pytest.raises(FakeHTTPError):
        pf.run_with_failover(action)
    assert called == ['gemini']


def test_safety_refusal_not_sent_to_backup(monkeypatch):
    configured(monkeypatch)
    seen=[]
    monkeypatch.setattr(pf, 'create_llm', lambda p: p.name)
    def action(p, _):
        seen.append(p.name)
        raise ValueError('unsafe output')
    with pytest.raises(ValueError):
        pf.run_with_failover(action)
    assert seen == ['gemini']


def test_both_unavailable_fails_closed(monkeypatch):
    configured(monkeypatch)
    monkeypatch.setattr(pf, 'create_llm', lambda p: p.name)
    with pytest.raises(pf.ProvidersUnavailable):
        pf.run_with_failover(lambda p, _: (_ for _ in ()).throw(FakeHTTPError(429)))


def test_unconfirmed_tiers_never_called(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'fake')
    monkeypatch.setenv('GROQ_API_KEY', 'fake')
    assert pf.configured_providers() == []
    with pytest.raises(pf.ProvidersUnavailable):
        pf.run_with_failover(lambda *_: 'never')


def test_only_confirmed_groq_works(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','fake')
    monkeypatch.setenv('GROQ_API_KEY','fake')
    monkeypatch.setenv('GROQ_FREE_TIER_CONFIRMED','true')
    assert [p.name for p in pf.configured_providers()] == ['groq']


def test_unreviewed_model_is_skipped(monkeypatch):
    configured(monkeypatch)
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-4-argon')
    assert [p.name for p in pf.configured_providers()] == ['groq']
    monkeypatch.setenv('GROQ_MODEL', 'some-unreviewed-paid-model')
    assert pf.configured_providers() == []


def test_choice_of_primary_respected(monkeypatch):
    configured(monkeypatch)
    monkeypatch.setenv('COPILOT_PRIMARY','groq')
    assert [p.name for p in pf.configured_providers()] == ['groq','gemini']


def test_config_flags_are_not_billing_verification(monkeypatch):
    configured(monkeypatch)
    assert [p.name for p in pf.configured_providers()] == ['gemini','groq']
    # Intentional: no provider account/billing API is called.


def test_timeout_error_retryable_but_unsafe_output_not():
    assert pf._retryable(TimeoutError('connection timeout'))
    assert pf._retryable(FakeHTTPError(429))
    assert not pf._retryable(FakeHTTPError(403))
    assert not pf._retryable(ValueError('unsafe user request'))


def test_fake_dotenv_still_supported_without_real_keys(monkeypatch, tmp_path):
    """Production .env loading works, while this test never reads real API keys."""
    from unittest.mock import patch

    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=dummy-test-key\n"
        "GEMINI_FREE_TIER_CONFIRMED=true\n"
        "GROQ_API_KEY=dummy-test-key\n"
        "GROQ_FREE_TIER_CONFIRMED=false\n",
        encoding="utf-8",
    )
    # Restore the entire environment at the end so fake .env values
    # cannot contaminate later tests.
    with patch.dict(os.environ, {"COPILOT_PRIMARY": "gemini"}, clear=True):
        assert [provider.name for provider in pf.configured_providers()] == ["gemini"]
