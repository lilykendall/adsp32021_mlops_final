# Data Pipelines

Scheduled, incremental counterparts to the root-level notebooks. The split is:

| | Root notebooks | This folder |
|---|---|---|
| Run frequency | Once, by hand | Nightly, as a Databricks Job |
| Bronze write | Full rebuild (`DROP` + `overwrite`) | `MERGE` on `(station, date)` |
| Date range | Fixed 2020-01-01..2024-12-31 | Rolling lookback window |
| Purpose | Establish the starting point | Keep it current |

Root notebooks stay the reference implementation of *how the data was first built*.
Anything that runs on a schedule lives here.

## Contents

- `noaa_client.py` — NOAA CDO GHCND fetch, pagination, retry/backoff, the catalogue
  lookups (`get_stations`, `get_datatypes`), and the long→wide reshape. Imported by both
  `01_data_ingestion.ipynb` (backfill) and `10_ingest_nightly.ipynb` so the two paths
  cannot drift apart.
- `00_explore_noaa_catalog.ipynb` — read-only scratch notebook for choosing the station
  and the datatype contract. Measures how densely each datatype is actually populated
  over a sample window and emits the exact `datatypes` string to adopt.
- `10_ingest_nightly.ipynb` — nightly bronze refresh.

Run `00_explore` before freezing anything; `10_ingest_nightly` depends on the contract it
produces.

## Design notes

**Why a lookback window rather than "yesterday".** GHCND publishes on a lag of
roughly five days, and NOAA revises already-published days after the fact. Asking
only for yesterday returns nothing most nights. The job re-requests the last
`lookback_days` (default 10) every run, which covers the lag and picks up
corrections for free.

**Why MERGE rather than append.** Because of the above, the same `(station, date)`
legitimately arrives more than once with different values. `MERGE` makes the job
idempotent — re-running it, or running it twice in a night, converges to the same
table instead of duplicating rows.

**NOAA wins on conflict, including nulls.** The merge uses `UPDATE SET *`, so a
re-requested day overwrites what's in bronze. That's what makes revisions land, but
it does mean if NOAA starts returning null for a datatype it previously reported,
the null overwrites the old value. NOAA is the system of record here, so this is the
intended behaviour — worth knowing if a column ever goes unexpectedly sparse.

**The station is pinned.** `01` picks a station dynamically (highest `datacoverage`
among those spanning the date range). That's fine for a one-time backfill but unsafe
on a schedule: if NOAA's station metadata shifts, a run could silently switch stations
and splice a different location's readings into the same table. The nightly job takes
`station_id` as an explicit parameter instead.

**NWS goes to its own table.** `01` writes `nws_forecast_snapshot` with
`mode("overwrite")` — a point-in-time snapshot. The nightly job appends to
`nws_forecast_history` with an `ingested_at` column instead, so forecast snapshots
accumulate into the time series that stage-4 drift monitoring needs. Two tables, two
purposes; `01`'s behaviour is unchanged.

## Prerequisites

`01_data_ingestion.ipynb` must have run at least once — the nightly job only does
incremental updates and will fail fast with a clear message if the bronze table
doesn't exist yet.

The NOAA token is read from a Databricks secret scope. Note the notebook currently
uses scope `mlo` / key `WEATHER_API_KEY`, while the root `README.md` documents scope
`weather-mlops` / key `noaa_cdo_token`. Both are parameters here; confirm which one
actually exists before scheduling, since a wrong value fails at run time rather than
at edit time.

## Scheduling

Run `10_ingest_nightly.ipynb` as a Databricks Job task with the widgets supplied as
task parameters. Suggested settings:

- **Schedule:** daily, early morning local time.
- **Retries:** 2, with a 15-minute backoff — the NOAA and NWS APIs both have
  transient outages.
- **Timeout:** generous enough for a 10-day window (single-digit minutes) but low
  enough to catch a hung request.
- **Notifications:** on failure, so a silent nightly break doesn't go unnoticed.

If the project needs Airflow in the picture for the deliverable, point a
`DatabricksRunNowOperator` at this job rather than reimplementing the pull — the
execution stays on Databricks and the DAG still exists.
