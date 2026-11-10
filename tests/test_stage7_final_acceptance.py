"""Final repair regression tests: all responses use fake *official* pages."""
import json
from pathlib import Path
import pytest
from src.teletalk_live import LivePage, answer_live, _page_evidence, LiveSourceError
from src import teletalk_discovery as discovery

CLOSED = '''<html><head><title>Closed SIM Offer</title></head><body>
<header><nav>Corporate Information | Online recharge &amp; purchase</nav></header>
<main><h1>Closed SIM Offer</h1><h2>Other Information</h2>
<ol><li>Only selected customers will be eligible for the above offers.</li>
<li>To check whether a customer is eligible for the inactive SIM offer, dial *155* desired mobile number and then # from any Teletalk number.</li>
<li>During the campaign period, customers can enjoy each offer only once.</li></ol>
<p>Offer price Tk 20 for this bundle.</p></main>
<footer>Management Information | Download MyTeletalk App</footer></body></html>'''
CARE = '''<html><head><title>Call Center</title></head><body>
<header><nav>Management Information Corporate Information</nav></header>
<main><h1>Call Center</h1><ul><li>Help Line: 121</li>
<li>Other Operator: 01500121121-9</li><li>Help Line [Corporate]: 267</li></ul></main></body></html>'''
NAV = '''<html><body><main><h1>Closed SIM</h1><p>Corporate Information</p>
<p>Download MyTeletalk App</p><p>Management Information</p>
<p>Online recharge &amp; purchase</p></main></body></html>'''

def live(url,html):
 return LivePage(url,html,'Some page text', 'Document','2026-11-02T00:00:00+00:00','sha')

class Fake:
 def __init__(self,html):self.html=html;self.requests=[]
 def fetch(self,url):
  self.requests.append(url)
  return live(url,self.html)

def test_closed_sim_conditions_no_nav_and_no_price():
 f=Fake(CLOSED)
 r=answer_live('closed sim eligibility conditions',fetcher=f,audit=False)
 assert r.mode=='live_source_information'
 assert f.requests==['https://www.teletalk.com.bd/en/voice/prepaid/closed-sim-offer-promotion']
 assert 'Only selected customers' in r.message
 assert '*155*' in r.message
 assert 'once' in r.message
 assert 'Tk 20' not in r.message
 assert 'Corporate Information' not in r.message
 assert 'Management Information' not in r.message

def test_nav_only_fails_closed():
 r=answer_live('closed sim eligibility conditions',fetcher=Fake(NAV),audit=False)
 assert r.mode=='unavailable'
 assert 'No cached offer price' in r.message

def test_customer_care_uses_verified_direct_page():
 f=Fake(CARE)
 r=answer_live('customer care information',fetcher=f,audit=False)
 assert r.mode=='live_source_information'
 assert f.requests[0]=='https://www.teletalk.com.bd/en/customer-care-center/call-center'
 assert '121' in r.message
 assert 'Corporate Information' not in r.message

def test_customer_care_address_routes_to_care_directory():
 f=Fake(CARE)
 answer_live('customer care center address in Dhaka',fetcher=f,audit=False)
 assert f.requests[0]=='https://www.teletalk.com.bd/en/customer-care-center'

def test_direct_roaming_and_faq_official_pages():
 for q,url in [('roaming eligibility','https://www.teletalk.com.bd/en/voice/international-roaming'),
               ('FAQ information','https://www.teletalk.com.bd/en/others/faq')]:
  f=Fake(CARE)
  answer_live(q,fetcher=f,audit=False)
  assert f.requests[0]==url


def test_discovery_uses_verified_fetcher_and_queues_only(tmp_path,monkeypatch):
 monkeypatch.setattr(discovery,'QUEUE',tmp_path/'review_queue')
 monkeypatch.setattr(discovery,'REPORT',tmp_path/'report.json')
 monkeypatch.setattr(discovery,'SEEDS',['https://www.teletalk.com.bd/en/sitemap'])
 html='<html><head><title>Sitemap</title></head><body><main><p>Official Teletalk pages and customer information for public reference.</p><a href="/en/customer-care-center">Care</a></main></body></html>'
 f=Fake(html)
 result=discovery.discover(max_pages=1,delay=.4,fetcher=f)
 assert result['attempted']==1 and result['captured']==1
 assert result['frontier_left']==1
 assert f.requests==['https://www.teletalk.com.bd/en/sitemap']
 assert len(list((tmp_path/'review_queue').glob('*.json')))==1
 assert result['note'].startswith('Captured pages are UNVERIFIED')
 assert not (tmp_path/'approved_offers.json').exists()


def test_discovery_fetch_failure_reported_not_crash(tmp_path,monkeypatch):
 monkeypatch.setattr(discovery,'QUEUE',tmp_path/'review_queue')
 monkeypatch.setattr(discovery,'REPORT',tmp_path/'report.json')
 monkeypatch.setattr(discovery,'SEEDS',['https://www.teletalk.com.bd/en/sitemap'])
 class Bad:
  def fetch(self,url): raise LiveSourceError('Official HTTPS unavailable')
 r=discovery.discover(max_pages=1,delay=.4,fetcher=Bad())
 assert r['captured']==0 and len(r['failed'])==1 and not r['complete']
 assert json.loads((tmp_path/'report.json').read_text())['failed']
