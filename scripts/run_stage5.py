"""Demonstrate Stage 5 predictions on actual PUBLIC-DATA validation rows only."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.routing import predict_customer
from scripts.stage5_samples import validation_request


def main():
    parser = argparse.ArgumentParser(description='Stage 5 demo: public dataset rows only')
    parser.add_argument('--track', choices=['iranian','ibm','both'], default='both')
    parser.add_argument('--row-index', type=int, default=0)
    args = parser.parse_args()
    tracks = ('iranian', 'ibm') if args.track == 'both' else (args.track,)
    for track in tracks:
        request = validation_request(track, args.row_index)
        output = predict_customer(request)
        print('\n' + '=' * 55)
        print(f'VALIDATION DATA DEMO: {output.data_source_label}')
        print(output.model_dump_json(indent=2))
    print('\nStage 5 demo finished. Predictions were NOT made for Teletalk customers.')


if __name__ == '__main__':
    main()
