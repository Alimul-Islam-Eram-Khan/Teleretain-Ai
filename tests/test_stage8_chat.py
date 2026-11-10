from types import SimpleNamespace
from src import stage8_chat as chat
from src.retention_recommender import RecommendationResult

def test_combined_churn_plus_offer_rejected(monkeypatch):
    monkeypatch.setattr(chat,'ask_project',lambda *args,**kw: (_ for _ in ()).throw(AssertionError('No model call')))
    monkeypatch.setattr(chat,'from_question',lambda *args,**kw: (_ for _ in ()).throw(AssertionError('No price call')))
    status,msg,cites=chat.ask_research_copilot('churn prediction and internet package')
    assert status=='needs_clarification' and not cites

def test_offer_does_not_call_churn_or_llm(monkeypatch):
    monkeypatch.setattr(chat,'ask_project',lambda *args,**kw: (_ for _ in ()).throw(AssertionError('No model call')))
    calls=[]
    def offer(q,**kwargs):
        calls.append(q)
        return RecommendationResult('no_match','No publicly matched offer')
    monkeypatch.setattr(chat,'from_question',offer)
    status,txt,cites=chat.ask_research_copilot('100 takar moddhe 7 diner internet pack')
    assert status=='no_match' and calls

def test_churn_requires_validated_previous_route(monkeypatch):
    got=[]
    def fake(question,**kw):
        got.append(kw)
        return SimpleNamespace(mode='live',message='Non-Teletalk benchmark only',citations=())
    monkeypatch.setattr(chat,'ask_project',fake)
    status,text,cites=chat.ask_research_copilot('Explain churn prediction',track='iranian',features={'example':1})
    assert status=='live' and got[0]['track']=='iranian' and got[0]['features']=={'example':1}

def test_service_questions_use_stage7(monkeypatch):
    monkeypatch.setattr(chat,'answer_live',lambda *args,**kw:SimpleNamespace(mode='live_source_information',message='Official Care source',citations=('https://www.teletalk.com.bd/en/customer-care-center/call-center',)))
    status,text,cites=chat.ask_research_copilot('customer care information')
    assert status=='live_source_information' and cites
