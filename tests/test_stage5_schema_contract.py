"""Unit checks of strict schemas. No model artifacts or customer records needed."""
import pytest
from pydantic import ValidationError
from src.schemas import IBMInput, IranianInput
from src.routing import decide_track, missing_fields_for


def test_dataset_field_contracts():
    assert len(IranianInput.model_fields) == 13
    assert len(IBMInput.model_fields) == 19
    assert 'Call Failure' in missing_fields_for('iranian', {})
    assert 'TotalCharges' in missing_fields_for('ibm', {})
    assert 'Churn' not in missing_fields_for('iranian', {})
    assert 'customerID' not in missing_fields_for('ibm', {})


def test_reject_empty_input():
    for cls, number in ((IranianInput, 13), (IBMInput, 19)):
        with pytest.raises(ValidationError) as exc:
            cls.model_validate({})
        assert len(exc.value.errors()) == number


def test_requires_explicit_track():
    with pytest.raises(ValueError):
        decide_track({'features': {}})
    with pytest.raises(ValueError):
        decide_track({'track': 'teletalk', 'features': {}})
    with pytest.raises(ValueError):
        decide_track({'track': 'iranian', 'features': {}, 'Churn': 1})
    assert decide_track({'track': 'ibm', 'features': {}}) == 'ibm'
