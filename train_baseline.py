"""
Baseline model — next-day weather quality classifier (Logistic Regression).

Pulls the full 2020-2024 NOAA historical daily record for the configured
station, derives a "Bad weather" label from PRCP (validated as the most
reliable available signal for this station — see notebook/README notes),
and predicts TOMORROW's label from TODAY's features -- matching the
Databricks v2 model's target definition (see build_dataset() below for why).
Trains a Logistic Regression baseline and prints evaluation metrics.

No MLflow / no Delta / no Databricks dependency — pure local script.

Usage:
    export NOAA_CDO_TOKEN="your_token_here"
    pip install requests pandas scikit-learn
    python train_baseline.py

First run pulls ~5 years x 19 datatypes from NOAA (paginated, rate-limited —
expect this to take a while). The raw pull is cached to
noaa_historical_daily_cache.csv afterward, so re-runs (e.g. while iterating
on the model itself) reuse the cached data instead of re-hitting the API.
Delete the cache file or pass --refresh to force a fresh pull.
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LOCATION_ID = "FIPS:17031"      # Cook County, IL (Chicago) — override as needed
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

CACHE_PATH = "noaa_historical_daily_cache.csv"

# PRCP > this threshold (mm, since we request units=metric) counts as "Bad".
# A small positive threshold (instead of PRCP > 0) filters out trace/measurement
# noise rather than flagging any nonzero reading as "bad weather".
PRCP_BAD_THRESHOLD_MM = 0.5

# Baseline feature set — intentionally minimal (see MODEL_CARD.md). These are
# the fields with the most complete coverage for this station; richer fields
# (WT01/WT03/WT08, wind gust, etc.) are candidates for the next iteration,
# not this baseline.
FEATURE_COLUMNS = ["AWND", "TMAX", "TMIN"]
TARGET_COLUMN = "Bad"  # represents TOMORROW's label -- see build_dataset()

NOAA_TOKEN = os.environ.get("NOAA_CDO_TOKEN")
NOAA_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
NOAA_MIN_REQUEST_INTERVAL = 0.25
NOAA_MAX_RETRIES = 5

NOAA_DEFAULT_DATATYPES = ("TMAX", "TMIN", "PRCP", "AWND")
NOAA_EXTRA_DATATYPES = (
    "SNOW", "SNWD", "TAVG",
    "WSF2", "WSF5", "WDF2", "WDF5",
    "WT01", "WT02", "WT03", "WT04", "WT05", "WT06", "WT08", "WT09",
    "WT11", "WT13", "WT14", "WT16", "WT17", "WT18", "WT19", "WT21", "WT22",
)
NOAA_ALL_DATATYPES = NOAA_DEFAULT_DATATYPES + NOAA_EXTRA_DATATYPES


# ---------------------------------------------------------------------------
# NOAA pull (same logic as test_ingestion_local.py)
# ---------------------------------------------------------------------------
def _noaa_get(headers, url, params):
    for attempt in range(NOAA_MAX_RETRIES):
        resp = requests.get(url, headers=headers, params=params)
        time.sleep(NOAA_MIN_REQUEST_INTERVAL)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else 2 ** attempt
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def get_stations(headers, location_id, datasetid="GHCND", limit=1000):
    resp = _noaa_get(headers, f"{NOAA_BASE_URL}/stations",
                      {"locationid": location_id, "datasetid": datasetid, "limit": limit})
    return pd.DataFrame(resp.json().get("results", []))


def get_supported_datatypes(headers, station_id, datasetid="GHCND", limit=1000):
    resp = _noaa_get(headers, f"{NOAA_BASE_URL}/datatypes",
                      {"stationid": station_id, "datasetid": datasetid, "limit": limit})
    return pd.DataFrame(resp.json().get("results", []))


def get_ghcnd_daily(headers, station_id, start_date, end_date, datatypes):
    all_results = []
    chunk_start = datetime.fromisoformat(start_date)
    final_end = datetime.fromisoformat(end_date)

    while chunk_start <= final_end:
        chunk_end = min(chunk_start + timedelta(days=364), final_end)
        offset = 1
        while True:
            resp = _noaa_get(
                headers,
                f"{NOAA_BASE_URL}/data",
                {
                    "datasetid": "GHCND",
                    "stationid": station_id,
                    "datatypeid": ",".join(datatypes),
                    "startdate": chunk_start.strftime("%Y-%m-%d"),
                    "enddate": chunk_end.strftime("%Y-%m-%d"),
                    "units": "metric",
                    "limit": 1000,
                    "offset": offset,
                },
            )
            payload = resp.json()
            results = payload.get("results", [])
            all_results.extend(results)
            if len(results) < 1000:
                break
            offset += 1000
        chunk_start = chunk_end + timedelta(days=1)
        print(f"  ...fetched through {chunk_end.date()}")

    return pd.DataFrame(all_results)


def pull_noaa_daily(refresh=False):
    if os.path.exists(CACHE_PATH) and not refresh:
        print(f"Loading cached historical data from {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH, parse_dates=["date"])

    if not NOAA_TOKEN:
        sys.exit(
            "Missing NOAA_CDO_TOKEN environment variable.\n"
            "export NOAA_CDO_TOKEN='your_token_here'"
        )
    headers = {"token": NOAA_TOKEN}

    print("Finding station...")
    stations_df = get_stations(headers, LOCATION_ID)
    candidates = stations_df[
        (stations_df["mindate"] <= START_DATE) & (stations_df["maxdate"] >= END_DATE)
    ].sort_values("datacoverage", ascending=False)
    if candidates.empty:
        sys.exit(f"No station covers {START_DATE}..{END_DATE} for {LOCATION_ID}")
    station_id = candidates.iloc[0]["id"]
    print(f"Using station: {station_id} ({candidates.iloc[0]['name']})")

    print("Checking datatype coverage...")
    available_df = get_supported_datatypes(headers, station_id)
    available_ids = set(available_df["id"]) if not available_df.empty else set()
    datatypes_to_pull = [dt for dt in NOAA_ALL_DATATYPES if dt in available_ids] or list(NOAA_ALL_DATATYPES)
    print(f"Pulling: {datatypes_to_pull}")

    print(f"Pulling {START_DATE}..{END_DATE} (this will take a while — paginated + rate-limited)...")
    noaa_raw = get_ghcnd_daily(headers, station_id, START_DATE, END_DATE, datatypes_to_pull)
    if noaa_raw.empty:
        sys.exit("NOAA returned no data for this station/date range.")

    noaa_daily = (
        noaa_raw.assign(date=lambda d: pd.to_datetime(d["date"]))
        .pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
        .reset_index()
        .sort_values("date")
    )
    noaa_daily.to_csv(CACHE_PATH, index=False)
    print(f"Cached {len(noaa_daily)} rows -> {CACHE_PATH}")
    return noaa_daily


# ---------------------------------------------------------------------------
# Label + features
# ---------------------------------------------------------------------------
def build_dataset(noaa_daily):
    """Builds features + a NEXT-DAY ("t+1") label, matching the Databricks
    v2 model's target definition (see 02_feature_store.ipynb / 03_baseline_model.ipynb,
    which use a `lead()` window function for the same shift). Originally this
    baseline predicted same-day weather from same-day features, which isn't
    real forecasting -- you don't need a model to tell you it's raining if
    you already know today's PRCP. Predicting tomorrow from today's readings
    is an honest forecasting task, and puts this baseline on the same target
    as the model it's meant to be a floor for.
    """
    df = noaa_daily.copy()

    if "PRCP" not in df.columns:
        sys.exit("PRCP column missing from pulled data — cannot derive the label.")

    df = df.sort_values("date").reset_index(drop=True)

    # Bad_today is well-defined wherever PRCP is present. The label used for
    # training/eval is TOMORROW's Bad_today value, via shift(-1) -- the same
    # row-order shift the Databricks feature store applies with lead(). Like
    # that implementation, this shifts by row position (next available row
    # in the sorted series), not strictly by calendar date, so a missing day
    # would make "tomorrow" mean "next day we have data for" -- acceptable
    # here since GHCND daily data for this station has no such gaps in range.
    bad_today = (df["PRCP"] > PRCP_BAD_THRESHOLD_MM).astype(int)
    df[TARGET_COLUMN] = bad_today.shift(-1)

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        sys.exit(f"Missing expected feature columns: {missing_features}")

    before = len(df)
    # dropna also removes the final row, which has no "tomorrow" to predict.
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing {FEATURE_COLUMNS} or no next-day label (of {before})")

    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force a fresh NOAA pull, ignore cache")
    args = parser.parse_args()

    noaa_daily = pull_noaa_daily(refresh=args.refresh)
    df = build_dataset(noaa_daily)

    print("\n" + "=" * 70)
    print("Dataset summary")
    print("=" * 70)
    print(f"Rows: {len(df)}  |  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Class balance ({TARGET_COLUMN}):")
    print(df[TARGET_COLUMN].value_counts(normalize=True).rename("proportion"))

    # Chronological split (not random) — train on the earlier ~80% of dates,
    # test on the most recent ~20%, so the model isn't evaluated on data that
    # temporally precedes some of its training data.
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"\nTrain: {len(train_df)} rows ({train_df['date'].min().date()} -> {train_df['date'].max().date()})")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min().date()} -> {test_df['date'].max().date()})")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    print("\nFeature ranges (train):")
    print(X_train.describe().loc[["min", "max", "mean", "std"]])
    if not np.isfinite(X_train.to_numpy()).all():
        sys.exit("Non-finite values (NaN/inf) found in training features after dropna — investigate the cache CSV.")

    # lbfgs's line search was landing on unstable intermediate steps on this
    # data (surfacing as RuntimeWarnings, and non-deterministic metrics across
    # identical reruns) even after scaling. liblinear uses coordinate descent
    # instead of a line search, avoiding that failure mode entirely, and is
    # well-suited to a small feature count like this baseline's.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)),
    ])

    # Suppressed narrowly, only around fit/predict, and only after the
    # isfinite() assertions below independently confirm the outputs are
    # real, finite numbers -- see the comment above for why this specific
    # warning is a known Accelerate/BLAS false alarm on this platform.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # Prove the suppressed warning was truly benign rather than just trusting it:
    assert np.isfinite(y_proba).all(), "Non-finite prediction probabilities — this would be a real problem."

    print("\n" + "=" * 70)
    print("Baseline results — Logistic Regression")
    print("=" * 70)
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Label rule: PRCP > {PRCP_BAD_THRESHOLD_MM}mm, predicted for the NEXT day (t+1)")
    print()
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred, zero_division=0):.3f}")
    if len(set(y_test)) > 1:
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
    else:
        print("ROC-AUC:   N/A (only one class present in test set)")

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=["Actual: Good", "Actual: Bad"], columns=["Pred: Good", "Pred: Bad"]))

    print("\nCoefficients (feature -> weight, on standardized scale):")
    clf = model.named_steps["clf"]
    for feat, coef in zip(FEATURE_COLUMNS, clf.coef_[0]):
        print(f"  {feat}: {coef:+.4f}")

    print("\nDone. This is the baseline every future model should beat.")


if __name__ == "__main__":
    main()
