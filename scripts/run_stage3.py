"""Execute from project root: python scripts/run_stage3.py"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_iranian, train_ibm


def main():
    parser = argparse.ArgumentParser(description="TeleRetain AI Stage 3 ML training")
    parser.add_argument("--track", choices=["iranian", "ibm", "both"], default="both")
    args = parser.parse_args()
    targets = (["iranian", "ibm"] if args.track == "both" else [args.track])
    print("TeleRetain AI Stage 3 — real public data; independent prediction tracks", flush=True)
    for track in targets:
        print(f"\n===== {track.upper()} =====", flush=True)
        result = (train_iranian.run() if track == "iranian" else train_ibm.run())
        print("Selected model:", result["best_model_name"])
        print("Locked validation threshold:", result["selected_threshold"])
        print("Validation metrics:", json.dumps(result["best_model_validation_metrics"], indent=2))
        print("FINAL TEST metrics:", json.dumps(result["best_model_final_test_metrics"], indent=2))
    print("\nStage 3 training completed. Run: python scripts/verify_stage3.py", flush=True)


if __name__ == "__main__":
    main()
