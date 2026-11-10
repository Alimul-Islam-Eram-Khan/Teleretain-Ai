from datetime import date
from pathlib import Path
import hashlib,json
import pytest
from src.teletalk_discovery import normalize_url,extract,approve_page,discover


def test_official_only_normalize():
    assert normalize_url('https://www.teletalk.com.bd/voice/prepaid/agami?x=1')=='https://www.teletalk.com.bd/voice/prepaid/agami'
    assert normalize_url('https://example.com/') is None
    assert normalize_url('https://www.teletalk.com.bd/account/login') is None


def test_html_to_documents():
    page=extract('<html><head><title>Offer</title></head><body><nav>PRIVATE NAV</nav><main><h1>Some data</h1><table><tr><th>Price</th><td>Tk 20</td></tr></table><a href="/en/sitemap">more</a></main></body></html>', 'https://www.teletalk.com.bd/a')
    assert page['title']=='Offer' and 'Tk 20' in page['text'] and 'PRIVATE NAV' not in page['text']
    assert page['tables'][0][0]==['Price','Tk 20'] and page['links']==['https://www.teletalk.com.bd/en/sitemap']


def test_approval_requires_ack(tmp_path,monkeypatch):
    import src.teletalk_discovery as mod
    monkeypatch.setattr(mod,'QUEUE',tmp_path)
    monkeypatch.setattr(mod,'REVIEWED',tmp_path.parent/'reviewed.json')
    p=tmp_path/'one.json'
    text='This is a manually reviewed official public content page with eligibility.'
    sha=hashlib.sha256(text.encode()).hexdigest()
    p.write_text(json.dumps(dict(url='https://www.teletalk.com.bd/en/sitemap',title='Reviewed',text=text,sha256=sha)),encoding='utf-8')
    with pytest.raises(ValueError,match='acknowledgement'):
        approve_page(p,expected_sha256=sha)
    with pytest.raises(ValueError,match='SHA'):
        approve_page(p,expected_sha256='bad',reviewer_acknowledgement=True)
    entry=approve_page(p,expected_sha256=sha,reviewer_acknowledgement=True)
    assert entry['title']=='Reviewed'


def test_discovery_never_serves_unreviewed(tmp_path,monkeypatch):
    import src.teletalk_discovery as mod
    monkeypatch.setattr(mod,'QUEUE',tmp_path/'queue')
    monkeypatch.setattr(mod,'REPORT',tmp_path/'report.json')
    monkeypatch.setattr(mod,'SEEDS',['https://www.teletalk.com.bd/en/sitemap'])
    monkeypatch.setattr(mod.time,'sleep',lambda x:None)
    class Allow:
        def can_fetch(self,*args):return True
    class Response:
        url='https://www.teletalk.com.bd/en/sitemap'; headers={'Content-Type':'text/html'};encoding='utf-8'
        def raise_for_status(self):pass
        def iter_content(self,n):yield b'<html><head><title>Offers</title></head><body><main><h1>Title</h1><p>Important public Teletalk information to inspect before adding.</p></main></body></html>'
    class FakeSession:
        def get(self,*args,**kwargs):return Response()
    result=discover(max_pages=1,delay=.4,session=FakeSession(),robots=Allow())
    assert result['captured']==1
    assert len(list((tmp_path/'queue').glob('*.json')))==1
    assert result['note'].startswith('Captured pages are UNVERIFIED')
