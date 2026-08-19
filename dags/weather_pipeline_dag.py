"""
Airflow DAG — Stage 2 "automated orchestrator pipeline" requirement.

Orchestrates the nightly Databricks pipeline described in
data_pipelines/README.md: `10_ingest_nightly.ipynb` (bronze refresh) must
finish before `11_features_nightly.ipynb` (builds weather_daily_v3) runs.
Both notebooks already run correctly as Databricks Job tasks on their own —
this DAG doesn't reimplement that logic, it triggers the two existing jobs
in the right order and on a schedule, using DatabricksRunNowOperator so
execution stays on Databricks (per the suggestion already in
data_pipelines/README.md's Scheduling section).

Setup (one-time, on the Airflow instance that will run this DAG):
    pip install apache-airflow-providers-databricks

    1. Create an Airflow connection `databricks_default`
       (Admin -> Connections -> Conn Type: Databricks), with:
         Host:     https://<your-workspace>.cloud.databricks.com
         Password: a Databricks personal access token (or use OAuth --
                   see the provider docs) with permission to run these jobs.
       Never put the token in this file or in Airflow Variables in plaintext.

    2. In Databricks, create two Jobs (Workflows -> Jobs -> Create Job),
       each with a single notebook task pointing at:
         - data_pipelines/10_ingest_nightly.ipynb
         - data_pipelines/11_features_nightly.ipynb
       Note each Job's numeric ID (shown in the Jobs UI / URL).

    3. Set two Airflow Variables (Admin -> Variables) with those IDs:
         weather_ingest_job_id
         weather_features_job_id
       (Defaults below are placeholders and will fail fast if left unset.)

    4. Copy this file into the Airflow instance's dags/ folder.

This DAG only triggers pre-existing Databricks Jobs; it does not define the
ingestion/feature logic itself (see data_pipelines/10_ingest_nightly.ipynb
and 11_features_nightly.ipynb for that).
"""

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator

DATABRICKS_CONN_ID = "databricks_default"

default_args = {
    "owner": "mlops-weather",
    "retries": 2,
    # NOAA and NWS both have transient outages; a 15-minute backoff (rather
    # than immediate retry) gives those a chance to clear, matching the
    # scheduling guidance in data_pipelines/README.md.
    "retry_delay": timedelta(minutes=15),
    "email_on_failure": True,
}

with DAG(
    dag_id="weather_nightly_pipeline",
    description="Nightly bronze refresh + v3 feature build (Stage 2 orchestration)",
    default_args=default_args,
    schedule="0 6 * * *",  # daily, early morning local time
    start_date=pendulum.datetime(2025, 1, 1, tz="America/Chicago"),
    catchup=False,
    tags=["weather", "mlops-final"],
) as dag:

    ingest_nightly = DatabricksRunNowOperator(
        task_id="ingest_nightly",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_id=Variable.get("weather_ingest_job_id"),
        # 10_ingest_nightly.ipynb's own widgets already default sensibly
        # (station_ids/datatypes fall back to noaa_client.DEFAULT_STATIONS);
        # override here only if a run needs different bounds.
        notebook_params={},
    )

    features_nightly = DatabricksRunNowOperator(
        task_id="features_nightly",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_id=Variable.get("weather_features_job_id"),
        notebook_params={},
    )

    # 11_features_nightly.ipynb reads the bronze table 10_ingest_nightly.ipynb
    # writes -- it fails fast with a clear message if bronze doesn't exist yet
    # (see data_pipelines/README.md Prerequisites), but the ordering still has
    # to be enforced here so a slow/retrying ingest run can't let features
    # start against stale or half-written bronze data.
    ingest_nightly >> features_nightly
