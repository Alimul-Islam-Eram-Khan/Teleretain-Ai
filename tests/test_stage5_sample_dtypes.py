"""Regression test for mixed numeric column dtypes in Stage 5 validation samples.

Toy numeric software data only; no customer records or model training.
"""

from pathlib import Path
import pandas as pd
from scripts import stage5_samples


def test_validation_request_preserves_integer_and_float_types(tmp_path, monkeypatch):
    path = tmp_path / 'data' / 'processed' / 'iranian' / 'val.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'Count': pd.Series([0, 1], dtype='int64'),
        'Rate': pd.Series([0.25, 0.5], dtype='float64'),
        'Optional': pd.Series([float('nan'), 1.25], dtype='float64'),
    }).to_csv(path, index=False)

    monkeypatch.setattr(stage5_samples, 'ROOT', tmp_path)
    monkeypatch.setattr(stage5_samples, '_read_report', lambda track: {
        'input_features': ['Count', 'Rate', 'Optional']
    })

    first = stage5_samples.validation_request('iranian', row_index=0)
    assert first['track'] == 'iranian'
    assert type(first['features']['Count']) is int
    assert first['features']['Count'] == 0
    assert type(first['features']['Rate']) is float
    assert first['features']['Rate'] == 0.25
    assert first['features']['Optional'] is None

    second = stage5_samples.validation_request('iranian', row_index=1)
    assert type(second['features']['Count']) is int
    assert second['features']['Count'] == 1
    assert type(second['features']['Optional']) is float
    assert second['features']['Optional'] == 1.25
