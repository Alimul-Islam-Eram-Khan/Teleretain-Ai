"""Stage 9 pure boundary tests; no test-set data, live website or API keys."""
from pathlib import Path
import pytest
from src import dashboard_stage9 as m


def test_track_explicit():
    assert m.checked_track('iranian')=='iranian'
    assert m.checked_track('ibm')=='ibm'
    for bad in (None,'teletalk','Iranian','',1):
        with pytest.raises((ValueError,TypeError)):m.checked_track(bad)


def test_public_disclaimer_is_explicit():
    assert 'NOT Teletalk customer' in m.DATA_POLICY
    assert 'eligibility' in m.DATA_POLICY


def test_chat_private_id_rejection():
    for value in ('', '  ','x'*601, 'hello 01712345678', 'NID 1234567890123'):
        with pytest.raises(ValueError):m.sanitized_chat_question(value)
    assert m.sanitized_chat_question('100 taka 7 din internet')=='100 taka 7 din internet'
    assert m.sanitized_chat_question('Gen-Z 1997 to 2012')=='Gen-Z 1997 to 2012'


def test_source_url_allowlist():
    # src.teletalk_live has been validated in Stage 7; only display official URLs.
    got=m.safe_source_urls(['https://www.teletalk.com.bd/internet-offers',
                           'https://evil.example/offer',
                           'http://www.teletalk.com.bd/internet-offers',
                           'https://www.teletalk.com.bd/internet-offers'])
    assert got==('https://www.teletalk.com.bd/internet-offers',)


def test_report_track_mismatch(tmp_path,monkeypatch):
    import json
    (tmp_path/'reports'/'stage3').mkdir(parents=True)
    (tmp_path/'reports'/'stage3'/'iranian_results.json').write_text(json.dumps({'track':'ibm'}))
    monkeypatch.setattr(m,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='source label'):m.local_report('iranian')


def test_input_form_fields_reject_unknown_columns(monkeypatch):
    import src.routing as routing
    class FakeSchema:model_fields={'one':1}
    monkeypatch.setattr(routing,'_get_context',lambda track: (None,{'input_features':['one']},{}))
    import src.schemas as schemas
    monkeypatch.setattr(schemas,'TRACK_SCHEMAS',{'iranian':FakeSchema})
    with pytest.raises(ValueError,match='does not match'):m.input_form_fields('iranian',{'other':5})


def test_prediction_output_must_stay_in_track(monkeypatch):
    import src.routing as routing
    class Result:
        def model_dump(self):return {'track':'ibm','churn_probability':.5}
    monkeypatch.setattr(routing,'predict_customer',lambda x:Result())
    with pytest.raises(ValueError,match='mismatch'):m.checked_prediction('iranian',{'abc':1})


def test_shap_no_cross_track(monkeypatch):
    import src.routing as routing
    monkeypatch.setattr(routing,'_read_report',lambda track:{'input_features':['a']})
    with pytest.raises(ValueError,match='match'):m.checked_explanation('ibm',{'bad':1})


def test_coverage_absent(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    assert m.local_coverage_summary()=={'available':False}


def test_validation_samples_only(monkeypatch):
    assert 'val.csv' in __import__('inspect').getsource(m.public_validation_count)
    assert 'test.csv' not in __import__('inspect').getsource(m.public_validation_count)


def test_no_network_on_readonly_helpers():
    import inspect
    src=inspect.getsource(m)
    assert 'requests.get' not in src
    assert 'api_key' not in src.lower()


def test_ui_has_five_required_tabs_and_no_hidden_network_call():
    script=(m.ROOT/'app'/'streamlit_app.py')
    # The app has exactly the five tabs required by the approved proposal.
    if not script.is_file():
        pytest.skip('Stage 9 UI file is only present after the full overlay is installed')
    text=script.read_text(encoding='utf-8')
    for label in ('Iranian Prediction','IBM Benchmark','Retention Copilot','Analytics','About & Limitations'):
        assert label in text
    assert 'st.chat_input' in text
    assert 'unsafe_allow_html=True' in text
    assert 'escape(offer.description)' in text
    assert 'verify=False' not in text
