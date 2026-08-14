"""
Drift monitoring — Stage 4, "Set up monitoring framework/dashboard (EvidentlyAI, ...)"

Compares a REFERENCE dataset (what the candidate model was trained on) against
a CURRENT dataset (new/live data) and reports on distribution drift per
feature, using EvidentlyAI. Outputs an HTML report you can open directly,
plus a plain console summary.

UPDATED (Payton): originally sourced its feature/target
definitions from train_baseline.py, which is the earlier 3-feature local
prototype (AWND/TMAX/TMIN, same-day PRCP label, local CSV cache). The model
actually registered as the `candidate` alias in Unity Catalog
(mlo.models.weather_quality) uses the improved v2 feature set instead --
7 features including lagged/rolling PRCP + TMAX, predicting TOMORROW's
weather, sourced from mlo.features.weather_daily_v2 / weather_labels (see
02_feature_store.ipynb / 03_baseline_model.ipynb). This version points at
that real feature set so drift results are meaningful for whatever's
actually deployed.

Requires a Databricks runtime (or Databricks Connect) with `spark` in scope,
since the reference data now comes from Unity Catalog rather than a local
CSV.

Status: reference data is real (pulled from the v2 feature store tables).
Current data is SYNTHETIC for now (see get_current_data() below) since
there's no live API to pull from yet (blocked on Stage 3 / Noah's
deployment) -- Payton's drift_corruption / 04_drift_simulation notebook
generates more varied synthetic scenarios than the single shift below; swap
get_current_data() out for a real live-data pull once the API is live.

Usage (run inside a Databricks notebook/job with spark available):
    pip install evidently
    python drift_monitoring.py                 # synthetic drifted sample
    python drift_monitoring.py --no-drift       # synthetic UNDRIFTED sample, as a sanity check
"""

import argparse
import sys

import numpy as np
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

# Real candidate-model feature/target definitions -- keep these in sync with
# FEATURES in 03_baseline_model.ipynb, not train_baseline.py.
CATALOG = "mlo"
FEATURE_TABLE = f"{CATALOG}.features.weather_daily_v2"
LABEL_TABLE = f"{CATALOG}.features.weather_labels"
MODEL_NAME = f"{CATALOG}.models.weather_quality"
MODEL_ALIAS = "candidate"

FEATURE_COLUMNS = ["AWND", "TMAX", "TMIN", "PRCP_today", "PRCP_lag1", "PRCP_roll3", "TMAX_lag1"]
TARGET_COLUMN = "Bad"

DRIFT_REPORT_PATH = "drift_report.html"


def get_reference_data():
    """What 'normal' looks like -- the same v2 feature store tables the
    candidate model was actually trained on (02_feature_store.ipynb /
    03_baseline_model.ipynb), not train_baseline.py's local CSV cache."""
    try:
        feat_df = spark.table(FEATURE_TABLE)
        label_df = spark.table(LABEL_TABLE).select("station", "date", TARGET_COLUMN)
    except NameError:
        sys.exit(
            "`spark` isn't in scope -- this script now reads from Unity Catalog "
            "feature tables, so it needs to run inside a Databricks notebook/job "
            "(or via Databricks Connect), not as a plain local script."
        )

    joined = feat_df.join(label_df, on=["station", "date"], how="inner")
    df = joined.toPandas().dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if df.empty:
        sys.exit(
            f"No rows after joining {FEATURE_TABLE} + {LABEL_TABLE} and dropping "
            f"nulls -- check that 02_feature_store.ipynb has been run."
        )
    return df.sort_values("date").reset_index(drop=True)


def get_current_data(reference_df, inject_drift=True):
    """
    PLACEHOLDER. Once Noah's FastAPI service is live, replace this function's
    body with an actual pull of recent live data (e.g. a rolling window of
    NWS observations passed through the same feature/label logic as
    02_feature_store.ipynb) instead of sampling+perturbing the reference set.

    For broader coverage than this single shift (missing data, schema
    changes, unit errors, stuck sensors, outliers), see
    04_drift_simulation.ipynb, which runs several corruption scenarios
    against this same v2 feature set.
    """
    sample = reference_df.sample(n=min(200, len(reference_df)), random_state=1).copy()
    if inject_drift:
        # Simulate e.g. a seasonal shift or sensor miscalibration.
        sample["TMAX"] = sample["TMAX"] + 12
        sample["TMAX_lag1"] = sample["TMAX_lag1"] + 12
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
    print(f"Reference data: {len(reference_df)} rows from {FEATURE_TABLE} + {LABEL_TABLE}")

    current_df = get_current_data(reference_df, inject_drift=not args.no_drift)
    print(f"Current data: {len(current_df)} rows "
          f"({'synthetic, drift injected' if not args.no_drift else 'synthetic, no drift injected'})")

    result = run_drift_report(reference_df, current_df)
    print_summary(result)

    print(f"\nFull HTML report -> {DRIFT_REPORT_PATH}")
    print("Done. Swap get_current_data() for a real live-data pull once the API is deployed.")


if __name__ == "__main__":
    main()
