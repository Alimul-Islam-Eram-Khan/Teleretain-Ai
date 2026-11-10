"""Stage 5 verification inputs copied from actual Stage 2 public VALIDATION rows.

Never creates fictional/synthetic customer records, never touches the test split.
"""
from pathlib import Path
import math
import pandas as pd
from src.data_prep_common import ROOT
from src.routing import _read_report


def validation_request(track: str, row_index: int = 0) -> dict:
    report = _read_report(track)
    path = ROOT / 'data' / 'processed' / track / 'val.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Stage 2 validation split missing: {path}')
    frame = pd.read_csv(path)
    if not (0 <= row_index < len(frame)):
        raise IndexError(f'row_index must be 0..{len(frame)-1}')
    features = {}
    for name in report['input_features']:
        value = frame[name].iloc[row_index]  # Column-first access preserves integer dtype.
        if pd.isna(value):
            features[name] = None
        elif hasattr(value, 'item'):
            features[name] = value.item()
        else:
            features[name] = value
    return {'track': track, 'features': features}
