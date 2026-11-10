import os
import ssl
import pytest
from src import teletalk_tls
from src import teletalk_live


def test_default_session_is_regular_requests():
    session = teletalk_tls.official_session(mode='certifi')
    try:
        assert not isinstance(session.get_adapter('https://www.teletalk.com.bd/robots.txt'),teletalk_tls.WindowsTrustAdapter)
    finally: session.close()


def test_invalid_modes_refused():
    for mode in ('insecure','verify=False','off'):
        with pytest.raises(ValueError):
            teletalk_tls.official_session(mode=mode)


def test_strict_tls_adapter_refuses_nonchecking_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    with pytest.raises(ValueError): teletalk_tls.WindowsTrustAdapter(ctx)


def test_custom_mock_session_is_not_swapped():
    class Stub: pass
    stub=Stub()
    assert teletalk_live.OfficialFetcher(session=stub).session is stub


def test_real_fetcher_uses_selected_session(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(teletalk_live,'official_session',lambda:sentinel)
    assert teletalk_live.OfficialFetcher().session is sentinel


def test_nonwindows_rejects_windows_mode(monkeypatch):
    monkeypatch.setattr(teletalk_tls.sys,'platform','linux')
    with pytest.raises(RuntimeError): teletalk_tls.official_session(mode='windows')


def test_windows_mode_mounts_only_official_hosts(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setattr(teletalk_tls.sys, 'platform', 'win32')
    native_context = SimpleNamespace(verify_mode=ssl.CERT_REQUIRED, check_hostname=True)
    monkeypatch.setitem(sys.modules, 'truststore', SimpleNamespace(SSLContext=lambda _:native_context))
    session=teletalk_tls.official_session(mode='windows')
    try:
        assert isinstance(session.get_adapter('https://www.teletalk.com.bd/internet-offers'),teletalk_tls.WindowsTrustAdapter)
        assert isinstance(session.get_adapter('https://teletalk.com.bd/robots.txt'),teletalk_tls.WindowsTrustAdapter)
        assert not isinstance(session.get_adapter('https://example.org/'),teletalk_tls.WindowsTrustAdapter)
    finally:
        session.close()
