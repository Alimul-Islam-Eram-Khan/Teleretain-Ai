"""Stage 9 regression: Streamlit dropdown category types match Pydantic's strict input schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import dashboard_stage9 as dashboard
from src.schemas import IranianInput, IBMInput


def _mock_context(monkeypatch, track, options):
    import src.routing as routing
    from src.schemas import TRACK_SCHEMAS
    schema = TRACK_SCHEMAS[track]
    columns = [f.alias or key for key, f in schema.model_fields.items()]
    monkeypatch.setattr(routing, '_get_context', lambda name: (
        None, {'input_features': columns}, options
    ))
    return columns


def test_iranian_numeric_dropdowns_are_native_ints(monkeypatch):
    """Protect Iranian strict=True fields from the old string-selectbox bug."""
    options = {'Complains': {'0', '1'}, 'Tariff Plan': {'1', '2'}, 'Status': {'1', '2'}}
    _mock_context(monkeypatch, 'iranian', options)
    sample = {
        'Call Failure': 12, 'Complains': 1, 'Subscription Length': 37,
        'Charge Amount': 1, 'Seconds of Use': 3050, 'Frequency of use': 44,
        'Frequency of SMS': 14, 'Distinct Called Numbers': 18,
        'Age Group': 3, 'Tariff Plan': 1, 'Status': 2, 'Age': 30,
        'Customer Value': 179.76,
    }
    descriptors = dashboard.input_form_fields('iranian', sample)
    fields = {f['name']: f for f in descriptors}
    for name in options:
        field = fields[name]
        assert field['kind'] == 'category'
        assert type(field['value']) is int
        assert all(type(choice) is int for choice in field['options'])
        assert field['value'] in field['options']
    # Exactly what the Streamlit form submits after using these options.
    submitted = {field['name']: field['value'] for field in descriptors}
    assert IranianInput.model_validate(submitted).model_dump(by_alias=True) == submitted
    # This test genuinely catches the original bug, rather than merely testing mocks.
    wrong = submitted | {'Complains': '1'}
    with pytest.raises(ValidationError):
        IranianInput.model_validate(wrong)


def test_iranian_numeric_selection_other_option_stays_int(monkeypatch):
    options = {'Complains': {'0', '1'}, 'Tariff Plan': {'1', '2'}, 'Status': {'1', '2'}}
    _mock_context(monkeypatch, 'iranian', options)
    sample = {
        'Call Failure': 0, 'Complains': 0, 'Subscription Length': 4,
        'Charge Amount': 1, 'Seconds of Use': 0, 'Frequency of use': 0,
        'Frequency of SMS': 0, 'Distinct Called Numbers': 0, 'Age Group': 1,
        'Tariff Plan': 2, 'Status': 1, 'Age': 19, 'Customer Value': 0.0,
    }
    descriptors = dashboard.input_form_fields('iranian', sample)
    for field in descriptors:
        if field['name'] in options:
            for selected in field['options']:
                assert type(selected) is int


def test_ibm_categorical_text_is_never_converted_to_int(monkeypatch):
    options = {'SeniorCitizen': {'0', '1'}, 'gender': {'Female', 'Male'},
               'Partner': {'No', 'Yes'}}
    _mock_context(monkeypatch, 'ibm', options)
    # No trained model/file dependency. A valid schema-shaped test fixture.
    sample = {
        'gender':'Female', 'SeniorCitizen':1, 'Partner':'Yes',
        'Dependents':'No', 'tenure':4, 'PhoneService':'Yes',
        'MultipleLines':'No', 'InternetService':'DSL', 'OnlineSecurity':'No',
        'OnlineBackup':'No', 'DeviceProtection':'No', 'TechSupport':'No',
        'StreamingTV':'No', 'StreamingMovies':'No', 'Contract':'Month-to-month',
        'PaperlessBilling':'Yes', 'PaymentMethod':'Electronic check',
        'MonthlyCharges':54.5, 'TotalCharges':89.5,
    }
    # Every IBM string feature is categorical in the saved encoder.
    for name, value in sample.items():
        if type(value) is str and name not in options:
            options[name] = {value}
    descriptors = dashboard.input_form_fields('ibm', sample)
    fields = {d['name']: d for d in descriptors}
    for name in options:
        assert isinstance(fields[name]['value'],str)
        assert all(isinstance(v,str) for v in fields[name]['options'])
    submitted = {field['name']: field['value'] for field in descriptors}
    assert IBMInput.model_validate(submitted).model_dump(by_alias=True) == submitted


def test_numeric_category_rejects_encoder_contract_anomalies(monkeypatch):
    _mock_context(monkeypatch, 'iranian', {'Status': {'1', '01'}})
    sample = {
        'Call Failure': 0, 'Complains': 0, 'Subscription Length': 1,
        'Charge Amount': 0, 'Seconds of Use': 0, 'Frequency of use': 0,
        'Frequency of SMS': 0, 'Distinct Called Numbers': 0,
        'Age Group': 1, 'Tariff Plan': 1, 'Status': 1, 'Age': 19,
        'Customer Value': 0.0,
    }
    with pytest.raises(ValueError,match='Numeric category'):
        dashboard.input_form_fields('iranian', sample)
