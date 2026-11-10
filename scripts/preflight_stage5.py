"""Read-only Stage 5 compatibility check for actual locally trained pipelines."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pydantic
from src.routing import _get_context, predict_customer
from scripts.stage5_samples import validation_request


def main():
    if int(pydantic.__version__.split('.')[0]) < 2:
        raise RuntimeError('Stage 5 needs pydantic v2. Upgrade: python -m pip install "pydantic>=2,<3"')
    print(f'Pydantic version: {pydantic.__version__}')
    for track in ('iranian','ibm'):
        pipeline, report, categories = _get_context(track)
        result = predict_customer(validation_request(track))
        features = len(report['input_features'])
        print(f'PASS: {track} | model={result.selected_model} | '
              f'raw_features={features} | threshold={result.decision_threshold:.2f} | '
              f'validation-row score={result.churn_probability:.6f}')
    print('SUCCESS: Stage 5 strict routing preflight passed.')


if __name__ == '__main__':
    main()
