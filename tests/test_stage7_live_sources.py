"""Deterministic fixtures: no actual network, no customer data."""
import pytest
from src.teletalk_live import (
    OfficialFetcher, LivePage, LiveSourceError, parse_live_offers, answer_live,
    official, _int_amount, _valid_days, _filter_live, _snippet
)
from src.teletalk_answer import parse_question

HTML = '''<html><head><title>Teletalk - Internet Offers</title></head><body><main>
<h2>Regular Data Package</h2>
<table><tr><th>Volume</th><th>Validity</th><th>Price</th><th>USSD Code</th><th>Trigger Information</th></tr>
<tr><td>1 GB</td><td>7 Days</td><td>TK 22</td><td>*111*534#</td><td>Yes</td></tr>
<tr><td>2 GB</td><td>30 Days</td><td>TK 88</td><td>*111*93#</td><td>Yes</td></tr></table>
<h2>Regular Combo Package</h2>
<table><tr><th>Volume</th><th>Validity</th><th>Price</th><th>USSD Code</th><th>Trigger Information</th></tr>
<tr><td>1 GB+25 minute+10 SMS</td><td>7 days</td><td>Tk 39</td><td>*111*116#</td><td>Yes</td></tr>
<tr><td>30 days</td><td>Tk 47</td><td>*111*118#</td><td>Yes</td></tr></table>
<h2>Minute Bundle Package</h2><table>
<tr><td>Minute_Bundle_23 Min</td><td>3 Days</td><td>Tk 15</td><td>*111*7*1#</td><td>Yes</td></tr></table>
<h2>SMS Bundle Package</h2><table>
<tr><td>SMS_Bundle_Pack50</td><td>3 Days</td><td>Tk 7</td><td>*111*6#</td><td>Yes</td></tr></table>
</main></body></html>'''


def page(html=HTML,url='https://www.teletalk.com.bd/internet-offers'):
    return LivePage(url,html,'Official Teletalk Offers\n'+('Offers and activation details. '*10),
                    'Offers','2026-11-02T00:00:00+00:00','mock-sha256')


class Robot:
    def __init__(self,allowed=True): self.allowed=allowed
    def can_fetch(self,ua,url): return self.allowed


class FakeResponse:
    def __init__(self,status=200,html=HTML,url=None,headers=None):
        self.status_code=status
        self.raw=html.encode('utf-8')
        self.headers=headers or {'Content-Type':'text/html'}
        self.encoding='utf-8'
        self.url=url or 'https://www.teletalk.com.bd/internet-offers'
        self.closed=False
    def iter_content(self,chunk_size=32768):
        for i in range(0,len(self.raw),chunk_size): yield self.raw[i:i+chunk_size]
    def close(self): self.closed=True


class Session:
    def __init__(self,responses):self.responses=list(responses);self.urls=[]
    def get(self,url,**kwargs):
        self.urls.append(url)
        assert kwargs['allow_redirects'] is False
        return self.responses.pop(0)


def test_reject_nonofficial_and_private_paths():
    for bad in ('http://www.teletalk.com.bd/', 'https://evil.test/',
                'https://www.teletalk.com.bd.evil.test/',
                'https://www.teletalk.com.bd/login',
                'https://www.teletalk.com.bd/checkout',
                'https://www.teletalk.com.bd/internet-offers?phone=123',
                'https://a@www.teletalk.com.bd/a',
                'https://www.teletalk.com.bd:444/a'):
        with pytest.raises(LiveSourceError):official(bad)


def test_fetcher_uses_only_official_html_and_closes_response():
    response=FakeResponse()
    s=Session([response])
    f=OfficialFetcher(s,robots_override=Robot(),min_interval=.4,clock=lambda:0,sleep=lambda _:None)
    p=f.fetch('https://www.teletalk.com.bd/internet-offers')
    assert p.url=='https://www.teletalk.com.bd/internet-offers'
    assert 'Regular Combo' in p.text
    assert response.closed
    assert len(s.urls)==1


def test_disallowed_robots_does_not_fetch_page():
    s=Session([])
    f=OfficialFetcher(s,robots_override=Robot(False),min_interval=.4)
    with pytest.raises(LiveSourceError): f.fetch('https://www.teletalk.com.bd/internet-offers')
    assert s.urls==[]


def test_offsite_redirect_blocked_before_request():
    s=Session([FakeResponse(302,headers={'Location':'https://evil.test/offers'})])
    f=OfficialFetcher(s,robots_override=Robot(),min_interval=.4,clock=lambda:0,sleep=lambda _:None)
    with pytest.raises(LiveSourceError): f.fetch('https://www.teletalk.com.bd/internet-offers')
    assert len(s.urls)==1


def test_http_and_js_only_fail_closed():
    for response in (FakeResponse(429), FakeResponse(html='<html><body>Hi</body></html>')):
        f=OfficialFetcher(Session([response]),robots_override=Robot(),min_interval=.4)
        with pytest.raises(LiveSourceError): f.fetch('https://www.teletalk.com.bd/internet-offers')


def test_oversize_fail_closed():
    r=FakeResponse(headers={'Content-Type':'text/html','Content-Length':'9000000'})
    f=OfficialFetcher(Session([r]),robots_override=Robot(),min_interval=.4)
    with pytest.raises(LiveSourceError): f.fetch('https://www.teletalk.com.bd/internet-offers')


def test_extract_offers_and_combo_continuation():
    offers,skipped=parse_live_offers(page())
    assert len(offers)==6
    assert not skipped
    by_code={o.ussd:o for o in offers}
    assert by_code['*111*534#'].category=='internet'
    assert by_code['*111*118#'].category=='combo'
    assert by_code['*111*118#'].description=='1 GB+25 minute+10 SMS'
    assert by_code['*111*118#'].validity_days==30
    assert by_code['*111*6#'].category=='sms'


def test_prohibit_special_page_as_general_public_offers():
    offers,_=parse_live_offers(page(url='https://www.teletalk.com.bd/en/voice/prepaid/gen-z'))
    assert offers==[]


def test_conflict_same_code_quarantined():
    html=HTML.replace('<tr><td>2 GB</td>',
                      '<tr><td>1 GB</td><td>7 Days</td><td>TK 30</td><td>*111*534#</td><td>Yes</td></tr><tr><td>2 GB</td>')
    offers,flags=parse_live_offers(page(html=html))
    assert '*111*534#' not in {x.ussd for x in offers}
    assert any(x['reason']=='conflicting_same_page_offer' for x in flags)


def test_query_filters_banglish_bangla_and_no_7taka_confusion():
    offers,_=parse_live_offers(page())
    for q in ('100 takar moddhe 7 diner internet', '১০০ টাকার মধ্যে ৭ দিনের ইন্টারনেট'):
        p=parse_question(q)
        assert p['budget_bdt']==100
        assert p['exact_validity_days']==7
        matched=[o for o in offers if _filter_live(o,p)]
        assert [x.ussd for x in matched]==['*111*534#']


def test_info_excerpts_are_source_limited():
    body='Hello\nYou must have an eligible SIM before activating the offer.\nThe offer is listed on the official page.'
    assert _snippet(body,'eligible SIM')


def test_no_old_prices_when_fetch_fails():
    class Broken:
        def fetch(self,url): raise LiveSourceError('network down')
    a=answer_live('100 tk under internet 7 days',fetcher=Broken(),audit=False)
    assert a.mode=='unavailable'
    assert 'No cached offer price' in a.message


def test_realistic_answer_source_and_time():
    class Good:
        def fetch(self,url):return page()
    a=answer_live('100 takar moddhe 7 diner internet',fetcher=Good(),audit=False)
    assert a.mode=='live_public_listing'
    assert 'BDT 22' in a.message
    assert 'BDT 88' not in a.message
    assert '2026-11-02T' in a.message
    assert a.citations[0]=='https://www.teletalk.com.bd/internet-offers'


def test_malformed_question_rejects():
    assert answer_live('',audit=False).mode=='unavailable'
    assert answer_live('x'*601,audit=False).mode=='unavailable'


def test_invalid_price_not_extracted():
    assert _int_amount('5GB') is None
    assert _int_amount('Tk 55')==55
    assert _valid_days('30 Days')==30
    assert _valid_days('Unlimited') is None


def test_special_eligibility_is_excerpt_not_price_or_activation(monkeypatch):
    from src import teletalk_live as module
    special_html='<html><body><main><h2>Gen-Z eligibility conditions</h2><p>The special package is for verified eligible new subscribers only.</p><p>Data price: Tk 17 for 7 Days</p></main></body></html>'
    class Direct:
        def fetch(self,url):
            return LivePage(url,special_html,'Gen-Z eligibility conditions\nThe special package is for verified eligible new subscribers only.\nData price: Tk 17 for 7 Days',
                            'Official','2026-11-02T01:00:00Z','sha')
    r=answer_live('Gen-Z eligibility conditions',fetcher=Direct(),audit=False)
    assert r.mode=='live_source_information'
    assert 'verified eligible new subscribers only' in r.message
    assert 'Tk 17' not in r.message
    assert 'NOT asserted' in r.message
    assert 'subscriber status is not accessible' in r.message


def test_price_audit_does_not_store_customer_question(monkeypatch,tmp_path):
    from src import teletalk_live as module
    monkeypatch.setattr(module,'AUDIT_DIR',tmp_path)
    class Live:
        def fetch(self,url):return page()
    question='my_secret_customer_question 100 takar moddhe 7 diner internet'
    r=answer_live(question,fetcher=Live(),audit=True)
    assert r.mode=='live_public_listing'
    audit=(tmp_path/'live_reads.jsonl').read_text(encoding='utf8')
    assert 'my_secret_customer_question' not in audit
    assert 'observed_offers' in audit
    assert '*111*534#' in audit


def test_unknown_service_cannot_invent_official_page():
    class Down:
        def fetch(self,url):raise LiveSourceError('unavailable sitemap')
    r=answer_live('something unknown FAQ',fetcher=Down(),audit=False)
    assert r.mode=='unavailable'


def test_external_content_type_rejected():
    response=FakeResponse(headers={'Content-Type':'application/pdf'})
    fetcher=OfficialFetcher(Session([response]),robots_override=Robot(),min_interval=.4)
    with pytest.raises(LiveSourceError,match='Expected official HTML'):
        fetcher.fetch('https://www.teletalk.com.bd/internet-offers')
