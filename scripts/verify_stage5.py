"""Read-only, repeatable routing and prediction tests using actual public validation rows.

No synthetic customer records; intentionally invalid mutations are used ONLY to
check that unsafe input is rejected, never stored/trained or sent to a model.
"""
from pathlib import Path
import copy
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from pydantic import ValidationError
from src.routing import decide_track, predict_customer, missing_fields_for, _get_context
from scripts.stage5_samples import validation_request


def rejects(fn, expected_type=(ValueError, ValidationError), reason=''):
    try:
        fn()
    except expected_type:
        return
    raise AssertionError(f'FAIL: invalid input was accepted: {reason}')


def verify_track(track: str):
    request = validation_request(track, row_index=0)
    output = predict_customer(request)
    pipeline, report, _ = _get_context(track)
    ordered = pd.DataFrame([{k: request['features'][k] for k in report['input_features']}])
    raw = float(pipeline.predict_proba(ordered)[0, 1])
    assert np.isclose(output.churn_probability, raw, atol=1e-10), 'Predictions differ from Stage 3 pipeline'
    assert output.predicted_class == int(raw >= report['selected_threshold'])
    assert output.decision_threshold == report['selected_threshold']
    assert output.track == track and output.data_source_label
    assert 'Teletalk' in output.disclaimer
    # Validate another actual public row to exercise the path more than once.
    output2 = predict_customer(validation_request(track, row_index=1))
    assert output2.track == track
    missing_key = next(iter(request['features']))
    reduced = copy.deepcopy(request)
    del reduced['features'][missing_key]
    assert missing_key in missing_fields_for(track, reduced['features'])
    rejects(lambda: predict_customer(reduced), reason='missing required feature')
    extra = copy.deepcopy(request)
    extra['features']['Churn'] = 1
    rejects(lambda: predict_customer(extra), reason='target feature must not be used')
    bad_category = copy.deepcopy(request)
    catname = 'Status' if track == 'iranian' else 'Contract'
    bad_category['features'][catname] = 99999 if track == 'iranian' else 'NOT_A_VALID_CONTRACT'
    rejects(lambda: predict_customer(bad_category), reason='unseen categorical value')
    not_number = copy.deepcopy(request)
    nname = 'Call Failure' if track == 'iranian' else 'tenure'
    not_number['features'][nname] = 'not-a-number'
    rejects(lambda: predict_customer(not_number), reason='invalid numeric feature')
    if track == 'ibm':
        forbidden_id = copy.deepcopy(request)
        forbidden_id['features']['customerID'] = 'not-a-real-customer-id'
        rejects(lambda: predict_customer(forbidden_id), reason='customer ID must not be an input')
    print(f'PASS: {track} | saved-model probability, threshold, output label, schemas, bad-input rejection')
    return request


def main():
    iranian = verify_track('iranian')
    ibm = verify_track('ibm')
    bad_track = copy.deepcopy(iranian)
    bad_track['track'] = 'ibm'
    rejects(lambda: predict_customer(bad_track), reason='Iranian features routed as IBM')
    bad_track2 = copy.deepcopy(ibm)
    bad_track2['track'] = 'iranian'
    rejects(lambda: predict_customer(bad_track2), reason='IBM features routed as Iranian')
    rejects(lambda: predict_customer({'features': iranian['features']}), reason='track omitted')
    rejects(lambda: decide_track({'track': 'unknown','features': {}}), reason='invalid track')
    rejects(lambda: decide_track({'track':'iranian','features':{},'source':'teletalk'}), reason='unexpected envelope field')
    print('PASS: Cross-track routing, ambiguous request, missing track, invalid envelope rejected')
    print('SUCCESS: Stage 5 strict Pydantic schemas and safe routing verified on local saved models.')


if __name__ == '__main__':
    main()
