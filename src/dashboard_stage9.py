"""Stage 9 Streamlit dashboard boundary: read-only local reports + isolated UI adapters.

NO real Teletalk customer data. Dataset rows are PUBLIC validation samples ONLY.
NO test-set browsing, retraining, fake prices, local model mutation, or implicit API calls.
All network/LLM calls are performed ONLY when the user explicitly presses a button.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACK_NAMES = {
    "iranian": "Iranian Track · UCI Churn",
    "ibm": "IBM Benchmark · Fictional Sample",
}
DATA_POLICY = (
    "Research proof of concept. Churn predictions use public Iranian/IBM datasets, "
    "NOT Teletalk customer data. Teletalk offers are independent public listings; "
    "availability and subscriber eligibility are NOT verified."
)


def checked_track(track: str) -> str:
    if track not in TRACK_NAMES:
        raise ValueError("Only explicitly selected Iranian or IBM track is allowed")
    return track


def local_report(track: str) -> dict:
    track = checked_track(track)
    p = ROOT / 'reports' / 'stage3' / f'{track}_results.json'
    data = json.loads(p.read_text(encoding='utf-8'))
    if data.get('track') != track:
        raise ValueError('Local metrics file source label mismatch')
    return data


def public_validation_count(track: str) -> int:
    checked_track(track)
    path = ROOT / 'data' / 'processed' / track / 'val.csv'
    # Only validation samples are exposed to the UI; no test split is loaded.
    return len(pd.read_csv(path, usecols=['Churn']))


def public_validation_sample(track: str, index: int) -> dict:
    checked_track(track)
    if type(index) is not int or index < 0:
        raise ValueError('Invalid validation sample index')
    from scripts.stage5_samples import validation_request
    request = validation_request(track, index)
    if request['track'] != track or not isinstance(request['features'],dict):
        raise ValueError('Validation sample has an unexpected feature contract')
    return request['features']


def input_form_fields(track: str, sample: dict) -> list[dict]:
    """Description of ONLY the strict saved-model contract; no default guesses.

    The allowed categorical values come from the actual train-fitted encoder.
    """
    checked_track(track)
    from src.routing import _get_context
    from src.schemas import TRACK_SCHEMAS
    _, report, allowed = _get_context(track)
    schema = TRACK_SCHEMAS[track]
    if set(sample) != set(report['input_features']):
        raise ValueError('Sample does not match model input schema')
    # Encoder categories are stored as strings for membership checks, but the
    # strict Iranian input schema requires *integer* Complains/Status/Tariff Plan.
    # Streamlit selectbox returns the same Python type as its option values.
    # Using string options here used to create a Pydantic ValidationError on submit.
    annotations = {field.alias or key: field.annotation
                   for key, field in schema.model_fields.items()}
    fields = []
    for name in report['input_features']:
        raw = sample[name]
        if name in allowed:
            choices = sorted(allowed[name])
            actual = str(raw)
            if actual not in choices:
                raise ValueError('Validation categorical value outside fitted encoder')
            if annotations[name] is int:
                # Keep strict Pydantic validation; supply native integers to it.
                # Reject unexpected encoder labels rather than quietly coercing.
                try:
                    numeric_choices = sorted(int(c) for c in choices)
                except ValueError as exc:
                    raise ValueError('Invalid numeric category in fitted encoder') from exc
                if (type(raw) is not int or len(set(numeric_choices)) != len(choices)
                        or any(str(c) not in allowed[name] for c in numeric_choices)):
                    raise ValueError('Numeric category does not match strict input schema')
                fields.append({'name':name,'kind':'category',
                               'value':raw,'options':numeric_choices})
            else:
                # IBM categorical text (including SeniorCitizen) remains text.
                fields.append({'name':name,'kind':'category','value':actual,'options':choices})
        elif raw is None:
            if name != 'TotalCharges' or track != 'ibm':
                raise ValueError('Unexpected null numeric sample feature')
            fields.append({'name':name,'kind':'nullable_float','value':None})
        elif type(raw) is int:
            fields.append({'name':name,'kind':'int','value':raw})
        elif type(raw) is float:
            fields.append({'name':name,'kind':'nullable_float' if name == 'TotalCharges' else 'float','value':raw})
        else:
            raise ValueError(f'Unknown feature type for {name}')
    # Unit tests check the exact ordered features, never silently add/delete columns.
    if len(fields) != len(schema.model_fields):
        raise ValueError('Model feature count mismatch')
    return fields


def checked_prediction(track: str, features: dict) -> dict:
    """Always call Stage 5; raw input never goes to the LLM."""
    checked_track(track)
    from src.routing import predict_customer
    predicted = predict_customer({'track': track, 'features': features}).model_dump()
    if predicted['track'] != track or not 0 <= predicted['churn_probability'] <= 1:
        raise ValueError('Prediction source mismatch')
    return predicted


def checked_explanation(track: str, features: dict) -> dict:
    """Stage 4 SHAP; only named raw features from the matching dataset."""
    checked_track(track)
    from src.routing import _read_report
    names = _read_report(track)['input_features']
    if set(features) != set(names):
        raise ValueError('Explanation features do not match the selected track')
    from src.explain import explain_one
    outcome = explain_one(track, pd.DataFrame([{k:features[k] for k in names}]))
    if outcome['track'] != track:
        raise ValueError('SHAP track mismatch')
    return outcome


def model_scorecard(track: str) -> dict:
    report = local_report(track)
    final = report['best_model_final_test_metrics']
    return {'track':track, 'model':report['best_model_name'],
            'threshold':float(report['selected_threshold']),
            'f1':float(final['f1']), 'precision':float(final['precision']),
            'recall':float(final['recall']), 'roc_auc':float(final['roc_auc']),
            'average_precision':float(final['pr_auc_average_precision']),
            'brier_score':float(final['brier_score']),
            'confusion_matrix':final['confusion_matrix'],
            'validation_rows':report['validation_rows'],
            'test_rows':report['test_rows']}


def safe_source_urls(urls) -> tuple[str,...]:
    """Never render untrusted external citations as clickable links."""
    from src.teletalk_live import official
    checked = []
    for url in urls:
        try: value = official(url)
        except (ValueError,TypeError,RuntimeError): continue
        if value not in checked: checked.append(value)
    return tuple(checked)


def sanitized_chat_question(question: str) -> str:
    """UI guard against accidentally submitting private phone/NID/account data."""
    if not isinstance(question,str) or not question.strip() or len(question)>600:
        raise ValueError('Write a short question (up to 600 characters).')
    import re
    digits = re.sub(r'\D','',question)
    if re.search(r'(?<!\d)(?:\+?8801|01)\d{9}(?!\d)', question) or re.search(r'(?<!\d)\d{10,17}(?!\d)',digits):
        raise ValueError('Remove phone, NID, or account numbers before asking.')
    return question.strip()


def local_coverage_summary() -> dict:
    p = ROOT / 'knowledge' / 'stage7_full' / 'coverage_audit.json'
    if not p.is_file(): return {'available':False}
    audit = json.loads(p.read_text(encoding='utf-8'))
    return {
        'available':True,
        'approved_offers':int(audit.get('approved_offers',0)),
        'reviewed_entries':int(audit.get('reviewed_information_entries',0)),
        'review_queue':int(audit.get('review_queue_count',0)),
        'conflicts':len(audit.get('quarantined_conflicts', [])),
        'discovery':audit.get('discovery'),
    }
