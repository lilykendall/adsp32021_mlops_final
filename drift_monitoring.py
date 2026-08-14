"""
Drift monitoring — Stage 4, "Set up monitoring framework/dashboard (EvidentlyAI, ...)"

Compares a REFERENCE dataset (the baseline model's training data — i.e. what
"normal" looks like) against a CURRENT dataset (new/live data) and reports
on distribution drift per feature, using the same AWND/TMAX/TMIN/Bad columns
train_baseline.py already defines. Outputs an HTML report you can open
directly, plus a plain console summary.

Status: reference data is real (reuses the baseline's cached NOAA pull).
Current data is SYNTHETIC for now (see get_current_data() below) since
there's no live API to pull from yet (blocked on Stage 3 / Noah's
deployment). Swap that one function out once the API is live — everything
else (report generation, thresholds, output) doesn't need to change.

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

# Reuse the exact same feature/target/cache definitions as the baseline
# model, instead of redefining them here and risking drift (no pun intended)
# between what the model was trained on and what this script monitors.
try:
    from train_baseline import CACHE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, PRCP_BAD_THRESHOLD_MM
except ImportError:
    sys.exit(
        "Couldn't import from train_baseline.py -- make sure this script "
        "sits in the same folder as train_baseline.py and that its cached "
        "data file exists (run train_baseline.py at least once first)."
    )

DRIFT_REPORT_PATH = "drift_report.html"


def get_reference_data():
    """The baseline model's training data -- what 'normal' looks like."""
    if not os.path.exists(CACHE_PATH):
        sys.exit(
            f"{CACHE_PATH} not found. Run train_baseline.py first so there's "
            f"a cached dataset to use as the drift-detection reference."
        )
    df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
    df[TARGET_COLUMN] = (df["PRCP"] > PRCP_BAD_THRESHOLD_MM).astype(int)
    return df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])


def get_current_data(reference_df, inject_drift=True):
    """
    PLACEHOLDER. Once Noah's FastAPI service is live, replace this function's
    body with an actual pull of recent live data (e.g. a rolling window of
    NWS observations passed through the same feature/label logic as
    train_baseline.py) instead of sampling+perturbing the reference set.

    For now: samples from the reference data and optionally injects an
    artificial shift, so the reporting logic itself can be built and tested
    before real live data is available.
    """
    sample = reference_df.sample(n=min(200, len(reference_df)), random_state=1).copy()
    if inject_drift:
        # Simulate e.g. a seasonal shift or sensor miscalibration -- pick
        # something a real drift-simulation script (Payton's task) might
        # also produce, so this is a reasonable stand-in until then.
        sample["TMAX"] = sample["TMAX"] + 12
        sample["AWND"] = sample["AWND"] * 1.5
    return sample


def run_drift_report(reference_df, current_df):
    definition = DataDefinition(
        numerical_columns=FEATURE_COLUMNS,
        categorical_columns=[TARGET_COLUMN],
    )
    reference_dataset = Dataset.from_pandas(reference_df[FEATURE_COLUMNS + [TARGET_COLUMN]], data_definition=definition)
    current_dataset = Dataset.from_pandas(current_df[FEATURE_COLUMNS + [TARGET_COLUMN]], data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference_dataset, current_data=current_dataset)
    result.save_html(DRIFT_REPORT_PATH)
    return result


def print_summary(result):
    """Plain-console summary, independent of the HTML report -- useful for
    logging/alerting later (Payton's automated-evaluation task) without
    needing to parse HTML."""
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
    print(f"Reference data: {len(reference_df)} rows from {CACHE_PATH}")

    current_df = get_current_data(reference_df, inject_drift=not args.no_drift)
    print(f"Current data: {len(current_df)} rows "
          f"({'synthetic, drift injected' if not args.no_drift else 'synthetic, no drift injected'})")

    result = run_drift_report(reference_df, current_df)
    print_summary(result)

    print(f"\nFull HTML report -> {DRIFT_REPORT_PATH}")
    print("Done. Swap get_current_data() for a real live-data pull once the API is deployed.")


if __name__ == "__main__":
    main()
