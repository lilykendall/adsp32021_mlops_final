"""
Drift monitoring — Stage 4, "Set up monitoring framework/dashboard (EvidentlyAI, ...)"

Compares a REFERENCE dataset (what "normal" looks like) against a CURRENT
dataset (new/live data) and reports on distribution drift per feature.
Outputs an HTML report you can open directly, plus a plain console summary.

FEATURE SET: v3 -- wide, multi-station (~40+ columns: Midway's full 12
NOAA datatypes plus scoped columns from Rockford/Minneapolis/Milwaukee/
Dubuque/O'Hare/Indianapolis, wind direction as sin/cos, zero-filled event
flags, lags/rolling windows, day-of-year seasonality). Defined in
data_pipelines/11_features_nightly.ipynb and consumed by 05_automl.ipynb,
which is the model actually being registered.

WHY THIS SCRIPT DOESN'T RE-DERIVE v3 ITSELF (unlike the earlier v2 version,
which reproduced 02_feature_store.ipynb's logic in pandas): v3 needs raw
data from six OTHER stations plus circular wind-direction encoding and
multi-station lag logic that currently only exists in Spark. Re-implementing
~40 columns by hand from reading code (that can't be run here to verify
against) risks a silent mismatch -- a wrong lag window or fill value that
shows up as a wrong drift number nobody catches. Instead, this script
consumes the ACTUAL joined training set Databricks already produces --
same dataframe 05_automl.ipynb trains on, exported once, nothing re-derived.

REQUIRED SETUP (one-time, from Databricks, after `pdf = ts.load_df().toPandas()`
in 05_automl.ipynb):
    pdf.to_csv("/dbfs/tmp/weather_v3_reference.csv", index=False)
Then download it (Databricks UI DBFS browser, or dbutils.fs.cp to a volume)
and save it locally as weather_daily_v3_reference.csv, in the same folder
as this script.

Feature columns are auto-detected from that CSV (everything except
date/station/Bad) rather than hardcoded here, since the exact v3 column
list depends on live station data availability and could shift as the
nightly pipeline runs -- auto-detection stays correct without this script
needing to be updated every time a column is added or dropped upstream.

Status: reference data is real (the actual v3 training set, once exported).
Current data here is SYNTHETIC by design (see get_current_data() below) --
this script runs standalone, without Databricks credentials, so it samples
and perturbs the reference set rather than calling a live endpoint. The
live version -- real predictions from the actual deployed model
(Databricks Model Serving; see 04_drift_simulation.ipynb Section 4) --
already exists there, reusing run_drift_report()/print_summary() from this
module. It isn't duplicated here since it needs a live endpoint + token
this local script deliberately doesn't require.

Usage:
    pip install evidently
    python drift_monitoring.py                 # synthetic drifted sample
    python drift_monitoring.py --no-drift       # synthetic UNDRIFTED sample, as a sanity check
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

REFERENCE_CSV = "weather_daily_v3_reference.csv"
DRIFT_REPORT_PATH = "drift_report.html"

TARGET_COLUMN = "Bad"  # represents TOMORROW's label (Bad_t+1)
NON_FEATURE_COLUMNS = {"date", "station", TARGET_COLUMN}


def get_reference_data():
    """The actual v3 training set exported from Databricks -- see module
    docstring for the export step. Not re-derived locally."""
    if not os.path.exists(REFERENCE_CSV):
        sys.exit(
            f"{REFERENCE_CSV} not found.\n\n"
            f"This script needs the real v3 feature+label set exported from "
            f"Databricks -- it can't be re-derived locally (v3 needs six "
            f"other stations' raw data plus Spark-side lag/rolling logic).\n\n"
            f"In Databricks, after `pdf = ts.load_df().toPandas()` in "
            f"05_automl.ipynb, run:\n"
            f'    pdf.to_csv("/dbfs/tmp/{REFERENCE_CSV}", index=False)\n'
            f"Download it and save it here as {REFERENCE_CSV}."
        )
    df = pd.read_csv(REFERENCE_CSV)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    missing_target = TARGET_COLUMN not in df.columns
    if missing_target:
        sys.exit(
            f"'{TARGET_COLUMN}' column not found in {REFERENCE_CSV}. Make sure "
            f"you exported `pdf` from `ts.load_df().toPandas()` (which joins "
            f"features to the label), not the raw feature table by itself."
        )

    df = df.dropna(subset=[TARGET_COLUMN])
    return df.reset_index(drop=True)


def get_feature_columns(df):
    """Auto-detected, not hardcoded -- see module docstring for why."""
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    dropped = set(cols) - set(numeric_cols)
    if dropped:
        print(f"Note: excluding non-numeric columns from drift monitoring: {sorted(dropped)}")
    return numeric_cols


def get_current_data(reference_df, feature_columns, inject_drift=True):
    """
    Synthetic by design, not a placeholder waiting on deployment -- this
    script runs standalone without Databricks credentials, so it samples
    from the reference data and optionally injects drift into a few
    representative columns instead of calling a live endpoint. See
    04_drift_simulation.ipynb Section 4 for the live version, which POSTs
    real records to the deployed model and reuses run_drift_report()/
    print_summary() from this module.

    Columns are matched by substring rather than exact name -- v3's column
    list can shift as the nightly pipeline runs, so this stays correct
    without needing hardcoded column names.
    """
    sample = reference_df.sample(n=min(200, len(reference_df)), random_state=1).copy()
    if not inject_drift:
        return sample

    def perturb_matching(substring, fn, max_cols=2):
        matches = [c for c in feature_columns if substring in c][:max_cols]
        for c in matches:
            sample[c] = fn(sample[c])
        return matches

    touched = []
    touched += perturb_matching("TMAX", lambda s: s + 12, max_cols=1)          # a raw temp column
    touched += perturb_matching("AWND", lambda s: s * 1.5, max_cols=1)         # a raw wind-speed column
    touched += perturb_matching("roll3", lambda s: s + 2.0, max_cols=1)        # a smoothed/rolling column
    touched += perturb_matching("PRCP_lag", lambda s: s + 1.5, max_cols=1)     # an upstream-lag column

    if not touched:
        print("Warning: no columns matched the expected drift-injection substrings "
              "(TMAX/AWND/roll3/PRCP_lag) -- check the actual v3 column names in "
              f"{REFERENCE_CSV} and adjust get_current_data() if naming has changed.")
    else:
        print(f"Injected synthetic drift into: {touched}")

    return sample


def run_drift_report(reference_df, current_df, feature_columns):
    definition = DataDefinition(
        numerical_columns=feature_columns,
        categorical_columns=[TARGET_COLUMN],
    )
    reference_dataset = Dataset.from_pandas(reference_df[feature_columns + [TARGET_COLUMN]], data_definition=definition)
    current_dataset = Dataset.from_pandas(current_df[feature_columns + [TARGET_COLUMN]], data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference_dataset, current_data=current_dataset)
    result.save_html(DRIFT_REPORT_PATH)
    return result


def print_summary(result):
    payload = result.dict()
    print("\n" + "=" * 70)
    print("Drift summary")
    print("=" * 70)
    for metric in payload.get("metrics", []):
        name = metric.get("metric_name", "unknown")
        value = metric.get("value")
        print(f"{name}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-drift", action="store_true",
        help="Use an undrifted synthetic sample instead -- sanity check that the report doesn't flag drift when there isn't any.",
    )
    args = parser.parse_args()

    reference_df = get_reference_data()
    feature_columns = get_feature_columns(reference_df)
    print(f"Reference data: {len(reference_df)} rows from {REFERENCE_CSV}")
    print(f"Auto-detected {len(feature_columns)} feature columns: {feature_columns}")

    current_df = get_current_data(reference_df, feature_columns, inject_drift=not args.no_drift)
    print(f"Current data: {len(current_df)} rows "
          f"({'synthetic, drift injected' if not args.no_drift else 'synthetic, no drift injected'})")

    result = run_drift_report(reference_df, current_df, feature_columns)
    print_summary(result)

    print(f"\nFull HTML report -> {DRIFT_REPORT_PATH}")
    print("Done. For live predictions against the deployed model instead of a synthetic")
    print("sample, see 04_drift_simulation.ipynb Section 4.")


if __name__ == "__main__":
    main()
