"""
Local test script for the NOAA + NWS data ingestion logic.

Mirrors the API-pulling logic in 00_data_ingestion.ipynb, but with every
Databricks-specific piece removed so it runs in a plain local Python
environment. Pulls Chicago Midway (GHCND:USW00014819), pinned via
data_pipelines.noaa_client.PRIMARY_STATION -- 00 pulls the full pinned
DEFAULT_STATIONS list, but a single station is enough to validate that the
pull/pagination/reshape mechanics work, and pinning it avoids ever silently
testing a different station than the rest of the pipeline is built on:

  - dbutils.widgets.*      -> plain variables in the CONFIG block below
  - dbutils.secrets.get()  -> NOAA_CDO_TOKEN environment variable
  - spark.createDataFrame(...).saveAsTable(...) -> NOT included here.
    This script only tests the API pulls + shape/field checks. Storage
    (Delta / local CSV / whatever you land on) is a separate concern —
    add it once you've confirmed the pulls themselves work.

Usage:
    export NOAA_CDO_TOKEN="your_token_here"   # https://www.ncdc.noaa.gov/cdo-web/token
    pip install requests pandas
    python test_ingestion_local.py
"""

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from data_pipelines.noaa_client import PRIMARY_STATION

pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# CONFIG (equivalent to the notebook widgets in Section 0)
# ---------------------------------------------------------------------------
START_DATE = "2025-07-01"           # short window for a quick local test
END_DATE = "2025-07-31"
LAT, LON = 41.85, -87.65            # NWS lookup point (Chicago-area)
NWS_CONTACT_EMAIL = "your_email@example.com"   # NWS requires a contact email in User-Agent

NOAA_TOKEN = os.environ.get("NOAA_CDO_TOKEN")
if not NOAA_TOKEN:
    sys.exit(
        "Missing NOAA_CDO_TOKEN environment variable.\n"
        "Get a free token at https://www.ncdc.noaa.gov/cdo-web/token, then:\n"
        "  export NOAA_CDO_TOKEN='your_token_here'"
    )

NOAA_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
HEADERS = {"token": NOAA_TOKEN}

NWS_HEADERS = {
    "User-Agent": f"(mlops-weather-project-local-test, contact: {NWS_CONTACT_EMAIL})",
    "Accept": "application/geo+json",
}

# Same extended datatype list as the notebook.
NOAA_DEFAULT_DATATYPES = ("TMAX", "TMIN", "PRCP", "AWND")
NOAA_EXTRA_DATATYPES = (
    "SNOW", "SNWD", "TAVG",
    "WSF2", "WSF5", "WDF2", "WDF5",
    "WT01", "WT02", "WT03", "WT04", "WT05", "WT06", "WT08", "WT09",
    "WT11", "WT13", "WT14", "WT16", "WT17", "WT18", "WT19", "WT21", "WT22",
)
NOAA_ALL_DATATYPES = NOAA_DEFAULT_DATATYPES + NOAA_EXTRA_DATATYPES

NOAA_MIN_REQUEST_INTERVAL = 0.25
NOAA_MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# NOAA CDO — historical data
# ---------------------------------------------------------------------------
def _noaa_get(url, params):
    """GET with NOAA-friendly pacing and retry/backoff on 429s."""
    for attempt in range(NOAA_MAX_RETRIES):
        resp = requests.get(url, headers=HEADERS, params=params)
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


def get_supported_datatypes(station_id, datasetid="GHCND", limit=1000):
    resp = _noaa_get(
        f"{NOAA_BASE_URL}/datatypes",
        params={"stationid": station_id, "datasetid": datasetid, "limit": limit},
    )
    return pd.DataFrame(resp.json().get("results", []))


def get_ghcnd_daily(station_id, start_date, end_date, datatypes=NOAA_ALL_DATATYPES):
    all_results = []
    chunk_start = datetime.fromisoformat(start_date)
    final_end = datetime.fromisoformat(end_date)

    while chunk_start <= final_end:
        chunk_end = min(chunk_start + timedelta(days=364), final_end)
        offset = 1
        while True:
            resp = _noaa_get(
                f"{NOAA_BASE_URL}/data",
                params={
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

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# NWS — live data
# ---------------------------------------------------------------------------
def get_nws_point_metadata(lat, lon):
    resp = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=NWS_HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_nws_forecast(forecast_url):
    resp = requests.get(forecast_url, headers=NWS_HEADERS)
    resp.raise_for_status()
    periods = resp.json()["properties"]["periods"]
    return pd.json_normalize(periods)


def get_nws_latest_observation(stations_url):
    stations_resp = requests.get(stations_url, headers=NWS_HEADERS)
    stations_resp.raise_for_status()
    nearest_station_id = stations_resp.json()["features"][0]["properties"]["stationIdentifier"]

    obs_resp = requests.get(
        f"https://api.weather.gov/stations/{nearest_station_id}/observations/latest",
        headers=NWS_HEADERS,
    )
    obs_resp.raise_for_status()
    return nearest_station_id, obs_resp.json()["properties"]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("1. NOAA CDO — station")
    print("=" * 70)
    station_id = PRIMARY_STATION
    print(f"Using station: {station_id} (Chicago Midway, pinned -- matches 00_data_ingestion.ipynb)\n")

    print("=" * 70)
    print("2. NOAA CDO — checking datatype coverage for this station")
    print("=" * 70)
    available_datatypes_df = get_supported_datatypes(station_id)
    available_ids = set(available_datatypes_df["id"]) if not available_datatypes_df.empty else set()
    missing = [dt for dt in NOAA_ALL_DATATYPES if dt not in available_ids]
    datatypes_to_pull = [dt for dt in NOAA_ALL_DATATYPES if dt in available_ids] or list(NOAA_ALL_DATATYPES)

    if missing:
        print(f"Station does NOT report: {missing}")
    print(f"Will pull: {datatypes_to_pull}\n")

    print("=" * 70)
    print("3. NOAA CDO — pulling historical daily data")
    print("=" * 70)
    noaa_raw = get_ghcnd_daily(station_id, START_DATE, END_DATE, datatypes=datatypes_to_pull)
    print(f"Raw rows: {len(noaa_raw)}")
    if noaa_raw.empty:
        print("WARNING: no historical data returned for this window/station.\n")
    else:
        noaa_daily = (
            noaa_raw.assign(date=lambda d: pd.to_datetime(d["date"]))
            .pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
            .reset_index()
            .sort_values("date")
        )
        print(f"Pivoted shape: {noaa_daily.shape}")
        print(f"Columns actually populated: {list(noaa_daily.columns)}")
        print(noaa_daily.head(), "\n")

    print("=" * 70)
    print("4. NWS — live forecast + current observation")
    print("=" * 70)
    point_meta = get_nws_point_metadata(LAT, LON)
    forecast_url = point_meta["properties"]["forecast"]
    stations_url = point_meta["properties"]["observationStations"]

    nws_forecast = get_nws_forecast(forecast_url)
    print(f"Forecast periods shape: {nws_forecast.shape}")
    print(nws_forecast[["name", "startTime", "temperature", "shortForecast"]].head(3), "\n")

    nearest_station_id, latest_obs = get_nws_latest_observation(stations_url)
    nws_observation_latest = pd.json_normalize(latest_obs)
    nws_observation_latest["nearest_station_id"] = nearest_station_id

    print(f"Nearest NWS station: {nearest_station_id}")
    print(f"Live observation shape: {nws_observation_latest.shape}")
    print(f"Live observation columns: {list(nws_observation_latest.columns)}\n")

    preview_cols = [c for c in [
        "timestamp", "temperature.value", "dewpoint.value", "windDirection.value",
        "windSpeed.value", "windGust.value", "barometricPressure.value",
        "seaLevelPressure.value", "relativeHumidity.value", "visibility.value",
        "textDescription", "presentWeather",
    ] if c in nws_observation_latest.columns]
    print(nws_observation_latest[preview_cols])

    print("\n" + "=" * 70)
    print("Done. No storage step run — this script only validates the pulls.")
    print("=" * 70)


if __name__ == "__main__":
    main()
