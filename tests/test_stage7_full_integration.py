"""V3 integration must never call an LLM for a public Teletalk offer."""
import sys,types
from src.teletalk_full_copilot import ask_project
from src.teletalk_answer import KnowledgeAnswer
from src.teletalk_live import LivePage

HTML = '''<html><head><title>Teletalk</title></head><body><main><h2>Regular Data Package</h2>
<table><tr><td>1 GB</td><td>7 Days</td><td>TK 22</td><td>*111*534#</td></tr></table>
</main></body></html>'''

class Fetch:
    def fetch(self,url):
        return LivePage(url,HTML,'Regular Data Package 1 GB 7 Days TK 22 '*5,
                        'Teletalk','2026-11-02T00:00:00+00:00','dummysha')


def test_offer_never_calls_stage6(monkeypatch):
    fake=types.ModuleType('src.chatbot')
    fake.ask_copilot=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('No LLM for Teletalk public offer'))
    monkeypatch.setitem(sys.modules,'src.chatbot',fake)
    r=ask_project('internet 7 days under 100 taka',track='iranian',features={'x':1},live=True,fetcher=Fetch())
    assert r.mode=='live_public_listing' and r.citations


def test_churn_requires_validated_features():
    r=ask_project('Explain my churn prediction',track='iranian',features=None,live=True)
    assert r.mode=='needs_input'


def test_churn_only_calls_stage6(monkeypatch):
    fake=types.ModuleType('src.chatbot')
    def respond(question,**kwargs):
        assert kwargs['track']=='ibm' and kwargs['features']=={'some':'opaque'}
        assert kwargs['live'] is False
        return KnowledgeAnswer('offline','IBM','Known verified score')
    fake.ask_copilot=respond
    monkeypatch.setitem(sys.modules,'src.chatbot',fake)
    r=ask_project('Explain churn prediction',track='ibm',features={'some':'opaque'})
    assert r.message=='Known verified score'
