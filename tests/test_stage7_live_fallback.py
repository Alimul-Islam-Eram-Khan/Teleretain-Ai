from src.teletalk_live import (
    LivePage, LiveSourceError, fetch_offers_fresh, answer_live,
    OFFERS_FALLBACK_URLS, _access_restricted,
)

HTML = """<html><head><title>Teletalk</title></head><body><main><h2>Regular Data Package</h2>
<table><tr><th>Volume</th><th>Validity</th><th>Price</th><th>USSD Code</th></tr>
<tr><td>1 GB</td><td>7 Days</td><td>Tk 22</td><td>*111*534#</td></tr>
</table></main></body></html>"""


def page(url):
    return LivePage(url, HTML, 'Official test offers from Teletalk public site',
                    'Internet Offers', '2026-11-02T00:00:00+00:00', 'test-sha')


def test_switch_official_host_after_connection_error():
    class Fetcher:
        def __init__(self): self.called = []
        def fetch(self, url):
            self.called.append(url)
            if len(self.called) == 1:
                raise LiveSourceError('Official source unavailable: ConnectionError')
            return page(url)
    fetcher=Fetcher()
    result=answer_live('100 takar moddhe 7 diner internet',fetcher=fetcher,audit=False)
    assert result.mode=='live_public_listing'
    assert len(fetcher.called)==2
    assert result.citations==('https://teletalk.com.bd/internet-offers',)
    assert 'BDT 22' in result.message


def test_no_rotating_hosts_to_bypass_403():
    class Denied:
        def __init__(self):self.called=[]
        def fetch(self,url):
            self.called.append(url)
            raise LiveSourceError('Official page returned HTTP 403')
    f=Denied()
    result=answer_live('internet under 100 taka', fetcher=f, audit=False)
    assert result.mode=='unavailable'
    assert len(f.called)==1
    assert 'BDT 22' not in result.message


def test_no_rotating_hosts_to_bypass_robots():
    class Denied:
        def __init__(self):self.called=[]
        def fetch(self,url):
            self.called.append(url)
            raise LiveSourceError('Official robots.txt disallows this page')
    f=Denied()
    result=answer_live('internet under 100 taka',fetcher=f,audit=False)
    assert result.mode=='unavailable'
    assert len(f.called)==1


def test_parse_failure_triggers_only_official_paths():
    class Fetcher:
        def __init__(self): self.called=[]
        def fetch(self,url):
            self.called.append(url)
            if len(self.called)==1:
                return LivePage(url,'<html><body>Valid text but no table.</body></html>',
                                'Valid text but no table.','Empty','2026-11-02T00:00:00Z','sha')
            return page(url)
    f=Fetcher()
    page_obj, offers, _=fetch_offers_fresh(f)
    assert offers and offers[0].price_bdt==22
    assert page_obj.url in OFFERS_FALLBACK_URLS
    assert all(u.startswith('https://') and 'teletalk.com.bd' in u for u in f.called)


def test_all_official_paths_fail_closed():
    class Down:
        def fetch(self,url): raise LiveSourceError('Official source unavailable: ConnectionError')
    result=answer_live('internet under 100 taka',fetcher=Down(),audit=False)
    assert result.mode=='unavailable'
    assert 'No cached offer price' in result.message


def test_critical_access_errors():
    assert _access_restricted(LiveSourceError('Cannot establish robots policy (HTTP 429)'))
    assert _access_restricted(LiveSourceError('Official page returned HTTP 403'))
    assert not _access_restricted(LiveSourceError('Official page returned HTTP 502'))
