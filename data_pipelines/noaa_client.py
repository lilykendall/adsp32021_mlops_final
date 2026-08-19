"""NOAA CDO (Climate Data Online) GHCND client.

Shared by the one-time historical backfill (``00_data_ingestion.ipynb``) and the
nightly incremental ingest (``data_pipelines/10_ingest_nightly.ipynb``). Both paths
write into the same bronze table, so they have to paginate, retry and reshape
identically — if these helpers were duplicated and drifted, the two paths could
disagree about what a given station-day looks like.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import requests

NOAA_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"

# The frozen bronze column contract. Kept here so the backfill and the nightly job
# can't disagree about it.
#
# WT* are weather-type event flags, not measurements: NOAA writes 1 on days the
# phenomenon occurred and writes nothing at all otherwise. A null means "did not happen",
# not "missing". They therefore read as low-coverage columns, and downstream they must be
# filled with 0 rather than dropped -- dropping nulls on WT03 would delete ~86% of rows.
DEFAULT_DATATYPES = (
    "AWND", "PRCP", "TMAX", "TMIN",
    "WDF2", "WDF5", "WSF2", "WSF5",
    "SNWD",
    "WT01", "WT02", "WT03",
)

# Every station whose readings land in bronze. Midway is the prediction target; the rest
# are context, weighted toward the west/north-west approach corridor since Midwest systems
# track roughly west to east.
DEFAULT_STATIONS = (
    "GHCND:USW00014819",  # Chicago Midway                -- target
    "GHCND:USW00094846",  # Chicago O'Hare
    "GHCND:USW00004838",  # Chicago Executive (Palwaukee)
    "GHCND:USW00014834",  # Joliet
    "GHCND:USW00094822",  # Rockford
    "GHCND:USW00014839",  # Milwaukee Mitchell            -- N
    "GHCND:USW00094908",  # Dubuque                       -- W
    "GHCND:USW00014922",  # Minneapolis-St Paul           -- NW
    "GHCND:USW00093819",  # Indianapolis International    -- SSE
    "GHCND:USW00094815",  # Kalamazoo                     -- E
    "GHCND:USW00014840",  # Muskegon                      -- ENE
)

# The station the model predicts for. v1/v2 features, the labels, and the model card are
# all built on Midway, so anything comparing against those has to filter to it.
PRIMARY_STATION = "GHCND:USW00014819"

_MIN_REQUEST_INTERVAL = 0.25  # NOAA CDO caps requests at ~5/sec; stay comfortably under
_MAX_RETRIES = 5
_PAGE_SIZE = 1000             # API hard cap on records per response
_MAX_CHUNK_DAYS = 364         # API rejects date ranges longer than a year


def make_headers(token):
    """Build the auth header dict NOAA CDO expects."""
    return {"token": token}


def _get(url, params, headers):
    """GET with NOAA-friendly pacing and retry/backoff on 429s."""
    for attempt in range(_MAX_RETRIES):
        resp = requests.get(url, headers=headers, params=params)
        time.sleep(_MIN_REQUEST_INTERVAL)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2 ** attempt)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def _get_all(url, params, headers):
    """Follow NOAA's offset pagination until a short page comes back."""
    results = []
    offset = 1  # NOAA offsets are 1-based
    while True:
        page = _get(url, {**params, "limit": _PAGE_SIZE, "offset": offset}, headers)
        page_results = page.json().get("results", [])
        results.extend(page_results)
        if len(page_results) < _PAGE_SIZE:
            return results
        offset += _PAGE_SIZE


def get_stations(location_id=None, headers=None, datasetid="GHCND", extent=None):
    """List GHCND stations, scoped by NOAA location id and/or a lat/lon bounding box.

    ``location_id`` takes a NOAA id such as 'FIPS:17031' (Cook County). ``extent`` takes
    'minlat,minlon,maxlat,maxlon' and is the easier way to sweep several metros at once,
    since it avoids looking up a FIPS code per county.
    """
    if headers is None:
        raise ValueError("headers is required — build it with make_headers(token)")
    if location_id is None and extent is None:
        raise ValueError("pass location_id, extent, or both")

    params = {"datasetid": datasetid}
    if location_id:
        params["locationid"] = location_id
    if extent:
        params["extent"] = extent
    return pd.DataFrame(_get_all(f"{NOAA_BASE_URL}/stations", params, headers))


def get_station_names(station_ids, headers):
    """Map GHCND station ids (e.g. DEFAULT_STATIONS) to NOAA's official station names.

    ``get_stations`` above only filters by location/extent, not by a specific id list, and
    NOAA CDO has no batch-by-id lookup -- so this hits GET /stations/{id} once per id via
    the same retry/backoff ``_get`` uses elsewhere. Fine for a handful of ids (DEFAULT_STATIONS
    is 11); for a large list, page through ``get_stations`` and filter instead.

    Returns {station_id: name}.
    """
    return {
        station_id: _get(f"{NOAA_BASE_URL}/stations/{station_id}", {}, headers).json()["name"]
        for station_id in station_ids
    }


def get_datatypes(headers, station_id=None, location_id=None, datasetid="GHCND"):
    """List the datatypes NOAA publishes, optionally narrowed to a station or location.

    Scoping to ``station_id`` is the useful form: it returns only what that station
    actually reports, each with its own ``mindate``/``maxdate``/``datacoverage`` — a far
    shorter and more honest list than the full GHCND catalogue, which advertises hundreds
    of soil-temperature and weather-type variants almost no station carries.
    """
    params = {"datasetid": datasetid}
    if station_id:
        params["stationid"] = station_id
    if location_id:
        params["locationid"] = location_id
    return pd.DataFrame(_get_all(f"{NOAA_BASE_URL}/datatypes", params, headers))


def get_ghcnd_daily(station_id, start_date, end_date, headers, datatypes=DEFAULT_DATATYPES):
    """Fetch daily GHCND records for one station between start_date and end_date
    (YYYY-MM-DD), paginated in <=1-year chunks per NOAA CDO API limits.

    Returns the API's long format: one row per (station, date, datatype).
    """
    all_results = []
    chunk_start = datetime.fromisoformat(start_date)
    final_end = datetime.fromisoformat(end_date)

    while chunk_start <= final_end:
        chunk_end = min(chunk_start + timedelta(days=_MAX_CHUNK_DAYS), final_end)
        offset = 1  # NOAA offsets are 1-based
        while True:
            resp = _get(
                f"{NOAA_BASE_URL}/data",
                {
                    "datasetid": "GHCND",
                    "stationid": station_id,
                    "datatypeid": ",".join(datatypes),
                    "startdate": chunk_start.strftime("%Y-%m-%d"),
                    "enddate": chunk_end.strftime("%Y-%m-%d"),
                    "units": "metric",
                    "limit": _PAGE_SIZE,
                    "offset": offset,
                },
                headers,
            )
            results = resp.json().get("results", [])
            all_results.extend(results)
            if len(results) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        chunk_start = chunk_end + timedelta(days=1)

    return pd.DataFrame(all_results)


def to_wide(noaa_raw, datatypes=DEFAULT_DATATYPES):
    """Long (one row per date/datatype) -> wide (one row per station/date).

    Every column in ``datatypes`` is guaranteed present, filled with NaN if the
    window contained no rows for it. This matters for the nightly job: a short
    window legitimately contains no SNOW rows in July, and without the fill the
    pivot would silently drop the column and break the MERGE against a table that
    has it.
    """
    columns = ["station", "date", *datatypes]

    if noaa_raw.empty:
        return pd.DataFrame(columns=columns)

    wide = (
        noaa_raw
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .pivot_table(index=["station", "date"], columns="datatype", values="value", aggfunc="first")
        .reset_index()
    )
    wide.columns.name = None  # pivot_table leaves the columns index named 'datatype'

    for datatype in datatypes:
        if datatype not in wide.columns:
            wide[datatype] = float("nan")
    wide[list(datatypes)] = wide[list(datatypes)].astype("float64")

    return wide[columns].sort_values("date").reset_index(drop=True)


def get_ghcnd_daily_multi(station_ids, start_date, end_date, headers,
                          datatypes=DEFAULT_DATATYPES, skip_errors=False, progress=None):
    """Fetch several stations and return one wide frame keyed (station, date).

    Bronze is long in the station dimension -- more stations means more rows, not more
    columns -- so the merge key ``(station, date)`` handles this unchanged.

    ``skip_errors`` is for the scheduled job: one station being unavailable overnight
    shouldn't cost the other nine, and MERGE simply leaves that station's rows alone
    until the next run picks them up. The backfill leaves it False so a partial history
    can't be written silently. ``progress`` is called as ``progress(station_id, n_rows)``
    after each station.

    Returns ``(frame, failures)`` where failures is a list of ``(station_id, message)``.
    """
    frames, failures = [], []

    for station_id in station_ids:
        try:
            raw = get_ghcnd_daily(station_id, start_date, end_date, headers, datatypes)
        except Exception as exc:
            if not skip_errors:
                raise
            failures.append((station_id, f"{type(exc).__name__}: {exc}"))
            continue

        wide = to_wide(raw, datatypes=datatypes)
        frames.append(wide)
        if progress:
            progress(station_id, len(wide))

    combined = (pd.concat(frames, ignore_index=True) if frames
                else to_wide(pd.DataFrame(), datatypes=datatypes))
    return combined, failures
