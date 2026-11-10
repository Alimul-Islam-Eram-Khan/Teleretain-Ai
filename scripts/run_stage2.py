"""Run Stage 2 from project root: python scripts/run_stage2.py"""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import preprocessing_iranian, preprocessing_ibm


def main():
    print("TeleRetain AI | Stage 2 | Two independent public-data tracks")
    print("Raw CSVs are never overwritten. No models trained.")
    outputs = {}
    for name, module in [("Iranian", preprocessing_iranian), ("IBM", preprocessing_ibm)]:
        print(f"\n--- {name} track ---")
        result = module.run()
        outputs[name.lower()] = result
        print("Cleaned rows:", result["cleaned_rows"])
        print("Class distribution:", result["class_distribution_cleaned"])
        print("Train / Validation / Test:",
              *(result["splits"][x]["rows"] for x in ["train", "val", "test"]))
        print("Transformed features:", result["transformed_feature_count"])
        print("Split method:", result["split_method"])
        print("Train/val/test churn rates:",
              *[round(result["splits"][k]["class_distribution"].get("1", 0)
                      / result["splits"][k]["rows"], 4)
                for k in ("train", "val", "test")])
        print("Checks:", json.dumps(result["checks"], indent=2))
    report_file = ROOT / "reports" / "stage2" / "README.md"
    report_file.write_text(
        "# TeleRetain AI — Stage 2 artifacts\n\n"
        "Two strictly independent tracks. No Teletalk customer records; no synthetic records.\n"
        "Raw files are untouched, exact Iranian duplicates removed before splitting, "
        "approximately 70/15/15 split (Iranian feature-vector groups kept intact), "
        "and separate train-fitted preprocessors.\n\n"
        "Read `iranian_data_quality.json` and `ibm_data_quality.json` for exact "
        "counts, input SHA-256 digests, columns, and data checks.\n"
        "No train/validation/test data was combined to fit preprocessors.\n"
        "The IBM `customerID` is excluded from features. The Iranian `Status` field "
        "needs sensitivity analysis for target proxy risk.\n",
        encoding="utf-8",
    )
    print("\nStage 2 preparation completed. Next: python scripts/verify_stage2.py")


if __name__ == "__main__":
    main()
