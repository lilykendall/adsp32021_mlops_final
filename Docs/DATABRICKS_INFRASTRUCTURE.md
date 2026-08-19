# Databricks Infrastructure

This repo's notebooks and scripts are only half the system. The rest —
Unity Catalog objects, registered models, serving endpoints, secrets, and
compute — lives entirely in the Databricks workspace and isn't visible from
the git history. This doc is a snapshot of that state, captured via the
`databricks` CLI (workspace `dbc-6e95f6fb-0d49.cloud.databricks.com`) on
**2026-08-19**. It will drift out of date; re-run the commands below to
refresh it rather than trusting these numbers indefinitely.

## Unity Catalog

Catalog: **`mlo`**

| Schema | Contents |
|---|---|
| `weather_mlops` | Bronze tables: `noaa_historical_daily`, `nws_forecast_history`, `nws_forecast_snapshot`. Plus 6 auto-generated OpenTelemetry tables (`random_forest_endpoint_otel_{logs,metrics,spans}`, `weather_daily_v3_otel_{logs,metrics,spans}`) — these are created automatically by Databricks Model Serving's inference-logging/monitoring feature once enabled on an endpoint, not by any code in this repo. |
| `features` | `weather_daily`, `weather_daily_v2` (+ `_online`), `weather_daily_v3` (+ `_online`), `weather_labels` |
| `models` | 7 registered models — see below |
| `default` | Auto-created, unused |

```bash
databricks catalogs list --profile <profile>
databricks schemas list mlo --profile <profile>
databricks tables list mlo <schema> --profile <profile>
```

## Registered Models

| Model | Created by | Traces to a notebook in this repo? |
|---|---|---|
| `mlo.models.weather_quality` (3 versions) | dbegin@uchicago.edu | Yes — `03_baseline_model.ipynb` |
| `mlo.models.weather_quality_flaml_best` | nsahmad@uchicago.edu | Yes — `05_automl.ipynb` |
| `mlo.models.weather_quality_hist_gbm` | nsahmad@uchicago.edu | No — built directly in the workspace |
| `mlo.models.weather_quality_logreg` | nsahmad@uchicago.edu | No — built directly in the workspace |
| `mlo.models.weather_quality_logreg_direct` | dbegin@uchicago.edu | No — built directly in the workspace |
| `mlo.models.weather_quality_loreg_l1` | nsahmad@uchicago.edu | No — built directly in the workspace |
| `mlo.models.weather_quality_prod` (random forest) | begin.diego@gmail.com | No — built directly in the workspace |

Only `weather_quality` and `weather_quality_flaml_best` have corresponding
training code committed anywhere. The other five are real, live models with
no reproducible source in this repo — if they matter for grading or
production, the training code that produced them needs to be added, or they
need to be documented as manual experiments.

```bash
databricks registered-models list --catalog-name mlo --schema-name models --profile <profile>
databricks model-versions list mlo.models.<name> --profile <profile>
```

## Serving Endpoints

All 7 are live (`READY`) and were created manually in the Databricks UI —
no code in this repo provisions any of them.

| Endpoint | Serves | Created | By |
|---|---|---|---|
| `endpoint_weather` | `weather_quality` v2 | 2026-08-12 | nsahmad@uchicago.edu |
| `flaml_endpoint` | `weather_quality_flaml_best` v1 | 2026-08-16 | nsahmad@uchicago.edu |
| `hist_gbm_endpoint` | `weather_quality_hist_gbm` v1 | 2026-08-16 | nsahmad@uchicago.edu |
| `l1_logreg_endpoint` | `weather_quality_loreg_l1` v1 | 2026-08-16 | nsahmad@uchicago.edu |
| `logreg_endpoint` | `weather_quality_logreg` v1 | 2026-08-16 | nsahmad@uchicago.edu |
| `logreg_direct_endpoint` | `weather_quality_logreg_direct` v1 | 2026-08-19 | paytonrosestewart@gmail.com |
| `random_forest_endpoint` | `weather_quality_prod` v1 | 2026-08-16 | nsahmad@uchicago.edu |

`04_drift_simulation.ipynb` (Section 4) calls `logreg_direct_endpoint` —
note it serves a plain Logistic Regression, not the FLAML AutoML champion
(`flaml_endpoint`) that the rest of the pipeline treats as the actual
registered model. `endpoint_weather` shows `config_update: UPDATE_FAILED`
from its last edit, though it's still `READY` and serving traffic.

```bash
databricks serving-endpoints list --profile <profile>
databricks serving-endpoints get <name> --profile <profile>
```

## Secrets

Scope `mlo` (backend: DATABRICKS) has one secret: **`WEATHER_API_KEY`**
(the NOAA CDO token, read by `00_data_ingestion.ipynb` and
`10_ingest_nightly.ipynb`).

**`DATABRICKS_TOKEN` does not exist yet.** `04_drift_simulation.ipynb`
was updated to read it via `dbutils.secrets.get("mlo", "DATABRICKS_TOKEN")`
after a hardcoded token was found and removed from that notebook — that
cell will fail with a `SecretDoesNotExist` error until someone creates it:

```bash
databricks secrets put-secret mlo DATABRICKS_TOKEN
```

```bash
databricks secrets list-scopes --profile <profile>
databricks secrets list-secrets mlo --profile <profile>
```

## Compute

No always-on clusters — everything runs on serverless compute. One SQL
warehouse (`Serverless Starter Warehouse`, Small), currently stopped.

```bash
databricks clusters list --profile <profile>
databricks warehouses list --profile <profile>
```

## Known gap: no Databricks Job exists

`data_pipelines/README.md`'s Scheduling section documents how the nightly
pipeline *should* be set up — `10_ingest_nightly.ipynb` as a Databricks Job
task, chained to `11_features_nightly.ipynb`, with retries/timeout/failure
notifications. As of this snapshot, **no Job has actually been created** —
`databricks jobs list` returns empty. The orchestration described in the
docs is a real, specific plan; it just isn't deployed yet.

```bash
databricks jobs list --profile <profile>
```
