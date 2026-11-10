"""Optional pytest integration tests; require existing saved models and public validation CSVs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import copy
import pytest
from pydantic import ValidationError
from src.routing import decide_track, missing_fields_for, predict_customer
from scripts.stage5_samples import validation_request


@pytest.mark.parametrize('track', ['iranian', 'ibm'])
def test_actual_validation_input_predicts(track):
    result = predict_customer(validation_request(track, row_index=0))
    assert result.track == track
    assert 0 <= result.churn_probability <= 1
    assert result.predicted_class in (0, 1)


@pytest.mark.parametrize('track', ['iranian', 'ibm'])
def test_rejects_missing_and_target_fields(track):
    request = validation_request(track)
    key = next(iter(request['features']))
    broken = copy.deepcopy(request)
    del broken['features'][key]
    assert key in missing_fields_for(track, broken['features'])
    with pytest.raises(ValidationError):
        predict_customer(broken)
    broken2 = copy.deepcopy(request)
    broken2['features']['Churn'] = 0
    with pytest.raises(ValidationError):
        predict_customer(broken2)


@pytest.mark.parametrize('correct,wrong', [('iranian','ibm'),('ibm','iranian')])
def test_rejects_cross_track(correct, wrong):
    request = validation_request(correct)
    request['track'] = wrong
    with pytest.raises(ValidationError):
        predict_customer(request)


def test_cannot_guess_track():
    with pytest.raises(ValueError):
        decide_track({'features': {}})
