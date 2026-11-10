# TeleRetain AI — Stage 2 artifacts

Two strictly independent tracks. No Teletalk customer records; no synthetic records.
Raw files are untouched, exact Iranian duplicates removed before splitting, approximately 70/15/15 split (Iranian feature-vector groups kept intact), and separate train-fitted preprocessors.

Read `iranian_data_quality.json` and `ibm_data_quality.json` for exact counts, input SHA-256 digests, columns, and data checks.
No train/validation/test data was combined to fit preprocessors.
The IBM `customerID` is excluded from features. The Iranian `Status` field needs sensitivity analysis for target proxy risk.
