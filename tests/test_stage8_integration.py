from __future__ import annotations
from types import SimpleNamespace
import pytest
from src.retention_recommender import RecommendationRequest, RecommendationResult
from src.stage8_integration import demo_independent_tracks
from src import routing
import src.stage8_integration as bridge


def test_prediction_route_is_explicit_and_separate(monkeypatch):
    requests=[]
    def predict(req):
        requests.append(req)
        return SimpleNamespace(model_dump=lambda: {'data_source_label':'Iranian Track','churn_probability':0.7})
    def fake_recommend(req,**kw):
        return RecommendationResult('no_match','No public offer')
    monkeypatch.setattr(routing,'predict_customer',predict)
    monkeypatch.setattr(bridge,'recommend',fake_recommend)
    req=RecommendationRequest(category='internet')
    result=demo_independent_tracks('iranian',{'dummy_test_only':1},req,fetcher=object())
    assert requests==[{'track':'iranian','features':{'dummy_test_only':1}}]
    assert result.public_dataset_prediction['churn_probability']==0.7
    assert result.separate_public_offer_matches.status=='no_match'
    assert 'NOT a personalized recommendation' in result.warning


def test_bad_track_rejected_before_model(monkeypatch):
    monkeypatch.setattr(routing,'predict_customer',lambda req: pytest.fail('Cannot predict'))
    with pytest.raises(ValueError):
        demo_independent_tracks('teletalk',{},RecommendationRequest(category='internet'))


def test_model_schema_error_prevents_offer_fetch(monkeypatch):
    monkeypatch.setattr(routing,'predict_customer',lambda req: (_ for _ in ()).throw(ValueError('bad schema')))
    monkeypatch.setattr(bridge,'recommend',lambda *args,**kw:pytest.fail('Should not retrieve offers'))
    with pytest.raises(ValueError,match='bad schema'):
        demo_independent_tracks('ibm',{'wrong':'columns'},RecommendationRequest(category='internet'))
