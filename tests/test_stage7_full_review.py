import json
from pathlib import Path
import pytest
from scripts import revalidate_stage7_full as mod

def test_review_prepare(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,'CHECKLIST',tmp_path/'check.json')
    n=mod.prepare()
    assert n==91
    rows=json.loads(mod.CHECKLIST.read_text())['items']
    assert len(rows)==n and all(x['reviewed'] is False for x in rows)

def test_revalidation_not_auto_approved(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,'CHECKLIST',tmp_path/'check.json')
    mod.prepare()
    with pytest.raises(ValueError,match='acknowledgement'):
        mod.apply()
    with pytest.raises(ValueError,match='All entries'):
        mod.apply(explicit_ack=True,reviewer='test reviewer')
