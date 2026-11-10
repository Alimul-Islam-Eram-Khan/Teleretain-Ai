from pathlib import Path
import pandas as pd

datasets = {
    "Iranian Track": Path("data/raw/iranian"),
    "IBM Benchmark": Path("data/raw/ibm"),
}

print("=" * 50)
print("TELERETAIN AI - DATASET INSPECTION")
print("=" * 50)

for name, folder in datasets.items():
    print(f"\n===== {name} =====")

    files = sorted(folder.glob("*.csv"))

    if not files:
        print(f"No CSV found in {folder}")
        continue

    for file in files:
        print(f"\nFile: {file.name}")

        try:
            df = pd.read_csv(file)

            print("Rows:", df.shape[0])
            print("Columns:", df.shape[1])
            print("Column names:", df.columns.tolist())
            print("Duplicates:", df.duplicated().sum())

            print("\nMissing values:")
            print(df.isnull().sum())

            print("\nChurn distribution:")
            if "Churn" in df.columns:
                print(df["Churn"].value_counts(dropna=False))
            else:
                print("Churn column not found")

            print("\nFirst 5 rows:")
            print(df.head())

        except Exception as e:
            print("Error:", e)

print("\nInspection completed!")
