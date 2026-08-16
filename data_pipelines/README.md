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
  over a sample window, compares candidate stations on a station × datatype matrix, and
  emits the exact `datatypes` string to adopt.
- `10_ingest_nightly.ipynb` — nightly bronze refresh.
- `11_features_nightly.ipynb` — builds `weather_daily_v3`. Runs after `10`.

## v3: wide, and why

v1/v2 are long — one row per station-day, Midway only. v3 is wide: one row per *date*, with
other stations pivoted into IATA-prefixed columns (`RFD_PRCP_lag1`). That shape is what makes
upstream signal usable, since Midwest systems track west to east and a long table can't
express "Rockford yesterday" without a self-join.

`station` changes meaning in v3. In v1/v2 it identifies who measured the row; in v3 every row
is a prediction *for* Midway, so it's a constant. It's kept only so v3 shares a primary-key
shape with v1/v2 and `03`'s `FeatureLookup` needs no structural change.

The feature scope is an explicit `FEATURE_SPEC` dict at the top of the notebook rather than
logic buried in the pivot — 11 stations × 12 datatypes is 132 columns before a single lag,
against roughly 475 minority-class events, so what gets included is a decision that belongs in
a diff. Current scope yields 44 features, about 11 events each.

v3 is also the first table written **without** a `DROP` — created once, overwritten after — so
its Delta history is real and `DESCRIBE HISTORY` / `VERSION AS OF` actually mean something.

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

**Stations are pinned, and bronze is long in the station dimension.** Both the backfill
and the nightly job pull `noaa_client.DEFAULT_STATIONS` — ten first-order airport stations
with Midway as the prediction target and the rest weighted toward the west/north-west
approach corridor, since Midwest systems track roughly west to east. More stations means
more *rows*, not more columns, so the `(station, date)` merge key is unaffected.

The list lives in the module rather than in job parameters on purpose: the nightly `MERGE`
requires bronze to match both the station list and the column contract, so changing either
should be a reviewable code change, not a silently-edited job parameter. Both notebooks
expose `station_ids` / `datatypes` widgets that fall back to the module when left blank.

**Anything reading bronze must now filter by station.** `02_feature_store.ipynb` had no
station filter — correct when bronze held one station, silently tenfold wrong once it holds
ten. It now filters to `PRIMARY_STATION` so `weather_daily_v2`, `weather_labels`,
`03_baseline_model.ipynb` and `drift_monitoring.py` keep exactly the shape they had. The
other nine stations get used in v3, pivoted into wide columns.

**NWS goes to its own table.** `01` writes `nws_forecast_snapshot` with
`mode("overwrite")` — a point-in-time snapshot. The nightly job appends to
`nws_forecast_history` with an `ingested_at` column instead, so forecast snapshots
accumulate into the time series that stage-4 drift monitoring needs. Two tables, two
purposes; `01`'s behaviour is unchanged.

## Prerequisites

`01_data_ingestion.ipynb` must have run at least once — the nightly job only does
incremental updates and will fail fast with a clear message if the bronze table
doesn't exist yet.

The NOAA token is read from Databricks secret scope `mlo`, key `WEATHER_API_KEY`. Both
are widgets here, so a Job task can override them, but that pair is the one that exists.

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
