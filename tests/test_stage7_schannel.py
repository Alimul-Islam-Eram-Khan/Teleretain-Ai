"""Deterministic simulated Windows Schannel tests; no network / credentials."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
import requests
from src import teletalk_schannel as sch
from src import teletalk_live as live


class FakeCurl:
    def __init__(self, status=200, html=b'<html><body>hello Teletalk testing</body></html>',
                 headers=None, exit_code=0):
        self.status = status
        self.html = html
        self.headers = headers
        self.exit_code = exit_code
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        if '--version' in cmd:
            return SimpleNamespace(returncode=0, stdout='curl 8.17 libcurl Schannel', stderr='')
        assert kw['shell'] is False
        assert '-k' not in cmd and '--insecure' not in cmd
        assert '--location' not in cmd and '-L' not in cmd
        assert cmd[1] == '-q'
        assert '--proto' in cmd and '=https' in cmd
        assert '--tlsv1.2' in cmd
        assert cmd[-2] == '--'
        assert cmd[-1].startswith('https://www.teletalk.com.bd/')
        header = Path(cmd[cmd.index('--dump-header') + 1]); body = Path(cmd[cmd.index('--output') + 1])
        header.write_bytes(self.headers or f'HTTP/1.1 {self.status} Status\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n'.encode())
        body.write_bytes(self.html)
        return SimpleNamespace(returncode=self.exit_code, stdout='', stderr='mock TLS error' if self.exit_code else '')


def checked(url, fake, **kwargs):
    return sch.fetch_via_schannel(url, headers={'User-Agent':'Demo','Accept':'text/html'},
                                  which=lambda s:'C:/Windows/System32/curl.exe',run=fake,
                                  platform='win32',**kwargs)


def test_successful_verified_get_is_not_head():
    fake = FakeCurl()
    resp = checked('https://www.teletalk.com.bd/internet-offers', fake)
    assert resp.status_code == 200
    assert b'hello Teletalk' in b''.join(resp.iter_content(chunk_size=3))
    assert '-I' not in fake.calls[1][0] and '--head' not in fake.calls[1][0]
    assert resp.headers.get('content-type') == 'text/html; charset=UTF-8'


def test_robots_get_404_propagated():
    fake = FakeCurl(status=404, html=b'Not found')
    resp = checked('https://www.teletalk.com.bd/robots.txt', fake)
    assert resp.status_code == 404


def test_proxy_connection_status_does_not_mask_origin_denial():
    header = (b'HTTP/1.1 200 Connection established\r\n\r\n'
              b'HTTP/2 403\r\nContent-Type: text/html\r\n\r\n')
    resp = checked('https://www.teletalk.com.bd/internet-offers', FakeCurl(headers=header))
    assert resp.status_code == 403


@pytest.mark.parametrize('url', ['http://www.teletalk.com.bd/internet-offers',
  'https://teletalk.com.bd.evil.example/internet-offers',
  'https://www.teletalk.com.bd:8443/internet-offers',
  'https://user:pass@www.teletalk.com.bd/internet-offers',
  'https://www.teletalk.com.bd/internet-offers?token=123',
  'https://www.teletalk.com.bd/internet-offers#frag',
  'https://www.teletalk.com.bd/login',
  'https://www.teletalk.com.bd/api/private'])
def test_rejects_nonofficial_or_unsafe_urls_before_call(url):
    fake = FakeCurl()
    with pytest.raises(sch.SchannelTransportError): checked(url, fake)
    assert fake.calls == []


def test_rejects_curl_backend_without_schannel():
    def wrong(*args,**kwargs):
        return SimpleNamespace(returncode=0, stdout='curl libcurl OpenSSL',stderr='')
    with pytest.raises(sch.SchannelTransportError, match='Schannel'):
        sch.fetch_via_schannel('https://www.teletalk.com.bd/internet-offers', headers={},
             which=lambda _: 'C:/Windows/System32/curl.exe', run=wrong, platform='win32')


def test_curl_tls_failure_is_fail_closed():
    fake = FakeCurl(exit_code=60)
    with pytest.raises(sch.SchannelTransportError,match='exit 60'):
        checked('https://www.teletalk.com.bd/internet-offers',fake)


def test_response_over_max_size_is_rejected():
    fake = FakeCurl(html=b'a'*2000)
    with pytest.raises(sch.SchannelTransportError,match='maximum allowed bytes'):
        checked('https://www.teletalk.com.bd/internet-offers',fake,max_bytes=1024)


def test_rejects_nonwindows():
    with pytest.raises(sch.SchannelTransportError,match='Windows'):
        sch.fetch_via_schannel('https://www.teletalk.com.bd/robots.txt',headers={},platform='linux')


def test_requests_ssl_fallback_requires_opt_in(monkeypatch):
    class Failing:
        def get(self,*args,**kwargs):raise requests.exceptions.SSLError('handshake')
    monkeypatch.setattr(live,'official_session',lambda:Failing())
    monkeypatch.delenv('TELERETAIN_SCHANNEL_FALLBACK',raising=False)
    f=live.OfficialFetcher()
    with pytest.raises(live.LiveSourceError,match='not enabled'):
        f._raw('https://www.teletalk.com.bd/robots.txt')


def test_requests_ssl_fallback_enabled_and_no_custom_session(monkeypatch):
    class Failing:
        def get(self,*args,**kwargs):raise requests.exceptions.SSLError('handshake')
    monkeypatch.setattr(live,'official_session',lambda:Failing())
    monkeypatch.setenv('TELERETAIN_SCHANNEL_FALLBACK','1')
    observed=[]
    def transport(url,**kwargs):
        observed.append(url)
        return sch.CurlResponse(404,requests.structures.CaseInsensitiveDict(),b'')
    monkeypatch.setattr(live,'fetch_via_schannel',transport)
    assert live.OfficialFetcher()._raw('https://www.teletalk.com.bd/robots.txt').status_code == 404
    assert observed == ['https://www.teletalk.com.bd/robots.txt']
    with pytest.raises(live.LiveSourceError):
        live.OfficialFetcher(session=Failing())._raw('https://www.teletalk.com.bd/robots.txt')
    assert len(observed)==1


def test_robots_policy_restricts_using_fallback(monkeypatch):
    class Failing:
        def get(self,*args,**kwargs):raise requests.exceptions.SSLError('handshake')
    monkeypatch.setattr(live,'official_session',lambda:Failing())
    monkeypatch.setenv('TELERETAIN_SCHANNEL_FALLBACK','1')
    robots=b'User-agent: *\nDisallow: /internet-offers\n'
    calls=[]
    def transport(url,**kwargs):
        calls.append(url)
        if url.endswith('/robots.txt'):
            return sch.CurlResponse(200,requests.structures.CaseInsensitiveDict(),robots)
        raise AssertionError('Disallowed document must not be downloaded')
    monkeypatch.setattr(live,'fetch_via_schannel',transport)
    f=live.OfficialFetcher(min_interval=0.4,sleep=lambda _:None,clock=lambda:10)
    with pytest.raises(live.LiveSourceError,match='disallows'):
        f.fetch('https://www.teletalk.com.bd/internet-offers')
    assert calls==['https://www.teletalk.com.bd/robots.txt']


def test_robots_404_allows_normal_document_fetch_not_permissions_claim(monkeypatch):
    class Failing:
        def get(self,*args,**kwargs):raise requests.exceptions.SSLError('handshake')
    monkeypatch.setattr(live,'official_session',lambda:Failing())
    monkeypatch.setenv('TELERETAIN_SCHANNEL_FALLBACK','1')
    doc=b'<html><head><title>Teletalk</title></head><body><main>Official current Internet offer listing details here.</main></body></html>'
    calls=[]
    def transport(url,**kwargs):
        calls.append(url)
        if url.endswith('/robots.txt'):
            return sch.CurlResponse(404,requests.structures.CaseInsensitiveDict(),b'')
        return sch.CurlResponse(200,requests.structures.CaseInsensitiveDict({'Content-Type':'text/html'}),doc)
    monkeypatch.setattr(live,'fetch_via_schannel',transport)
    f=live.OfficialFetcher(min_interval=0.4,sleep=lambda _:None,clock=lambda:10)
    page=f.fetch('https://www.teletalk.com.bd/internet-offers')
    assert 'Official current Internet' in page.text
    assert len(calls)==2
