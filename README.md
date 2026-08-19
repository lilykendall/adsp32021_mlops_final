# Weather Prediction MLOps Pipeline

An end-to-end MLOps pipeline that predicts weather conditions (e.g. bad weather / rain) from historical and live weather data, built for the ADSP 32021 MLOps final project.

**Team:** Gabe Horas, Noah Ahmad, Lily Kendall, Payton Stewart, Diego Begin

## Project Overview

The system pulls historical weather data for model training, tracks experiments, deploys a champion model as a live inference API, and monitors that model in production for drift. It's organized into four stages:

1. **Data Ingestion & Baselines** — pull historical + live weather data, define the target variable and evaluation metric, and establish a baseline model.
2. **Pipeline Automation & Experiment Tracking** — automate preprocessing/training (Databricks Jobs), version data/features, track experiments (MLflow), and register the winning model to a model registry.
3. **Containerization & Deployment** — deploy the registered model via Databricks Model Serving for real-time predictions.
4. **Production Monitoring & Drift Simulation** — monitor the deployed model, simulate data drift, verify the monitoring dashboard catches it, and set up automated evaluation/alerting.

## Architecture

- **Experimentation system** with promotion steps (champion vs. candidate) and sanity checks / secret holdout before a model reaches production.
- **CI/CD/CT** for automated build, test, and (re)training.
- **A live inference API** that also serves as evaluation/monitoring input via real-time data.

## Data Sources

| Source | Purpose | Auth |
|---|---|---|
| [NOAA CDO (Climate Data Online)](https://www.ncdc.noaa.gov/cdo-web/token) — GHCND daily station data | Historical training/test data | Free API token |
| [api.weather.gov](https://www.weather.gov/documentation/services-web-api) (NWS) | Live observations & forecasts for production/monitoring and drift simulation | None (descriptive `User-Agent` required) |
| [Open-Meteo](https://open-meteo.com/) archive & forecast APIs | Early proof-of-concept data pulls (see `FinalProjectProposal.html`) | None |

## Tooling

- **MLflow** for experiment tracking (params, metrics, artifacts) and the model registry, run on Databricks where available; otherwise prototyped with custom champion/candidate grading code.
- **DVC + Git** for data versioning and feature storage.
- **Databricks Jobs** for pipeline automation (preprocessing → training).
- **Databricks Model Serving** for serving the registered model.
- **EvidentlyAI / Prometheus + Grafana** (or custom code) for production monitoring and drift detection.

## Repository Structure

```
.
├── 00_data_ingestion.ipynb                 # Stage 1: pulls NOAA CDO historical data and NWS live data,
│                                           # writes bronze Delta tables (Databricks)
├── 01_eda.ipynb                            # Stage 1: exploratory data analysis on the bronze tables
├── 02_feature_store.ipynb                  # Stage 2: builds the v2 (Midway-only) feature table
├── 03_baseline_model.ipynb                 # Stage 2: Databricks baseline model, MLflow tracking,
│                                           # Unity Catalog model registration + semantic-version tags
├── 04_drift_simulation.ipynb               # Stage 4: corruption scenarios, Evidently drift reports,
│                                           # live endpoint stress test, alerting/decision logic
├── 05_automl.ipynb                         # Stage 2: FLAML AutoML search over the v3 feature set,
│                                           # logged to MLflow and registered to the model registry
├── drift_monitoring.py                     # Stage 4: reusable Evidently drift-report/summary functions,
│                                           # runnable locally against an exported reference CSV
├── test_ingestion_local.py                 # Local, non-Databricks smoke test for the NOAA/NWS API
│                                           # pulls — validates field coverage without dbutils/spark
├── train_baseline.py                       # Stage 1: local baseline weather-quality classifier
│                                           # (Logistic Regression), runs without Databricks
├── data_pipelines/                         # Scheduled, incremental counterparts to the root notebooks
│                                           # (nightly ingestion + v3 wide feature build) — see its own README
├── MODEL_CARD.md                           # Baseline model card — target definition, features,
│                                           # metrics, caveats
├── FinalProjectProposal.html               # Original project proposal + data-pulling proof of concept
├── MLOps_Weather_Project_Task_Tracker.xlsx # Task tracker across all four project phases
├── environment.yaml                        # Conda environment spec (local scripts + notebook tooling)
├── requirements.txt                        # Pip dependency list, with Databricks-only deps documented
└── README.md
```

Deployment is **Databricks Model Serving**, hosting the AutoML champion
trained in `05_automl.ipynb` on the full v3 feature set.
`04_drift_simulation.ipynb` (Section 4) calls that endpoint directly for
baseline validation and drift stress-testing — no code in this repo
provisions that endpoint, it's created manually in the Databricks UI.

## Orchestration (Databricks Jobs)

The nightly pipeline is orchestrated as a Databricks Job chaining
`data_pipelines/10_ingest_nightly.ipynb` → `11_features_nightly.ipynb`, on a
daily schedule — see the Scheduling section of `data_pipelines/README.md`
for the suggested settings (retries, timeout, failure notifications). This
satisfies the "automated orchestrator" requirement as an equivalent workflow
platform to Airflow/Prefect, without adding a second orchestrator on top of
infrastructure the pipeline doesn't otherwise need.

## Baseline Model

The Stage 1 baseline (`train_baseline.py`) is a Logistic Regression classifier
predicting whether **tomorrow** will be "Bad" weather (`PRCP > 0.5mm`) from
**today's** `AWND`, `TMAX`, `TMIN` — matching the target definition used by
the Databricks v2 model (`03_baseline_model.ipynb`), so the two are
comparable.

Trained on Chicago Midway Airport (`GHCND:USW00014819`), 2020–2024,
chronological 80/20 split:

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.503 |
| Precision | 0.284 |
| Recall    | 0.540 |
| F1        | 0.372 |
| ROC-AUC   | 0.527 |

ROC-AUC near 0.5 shows today's temperature/wind alone barely predict
tomorrow's rain — this baseline was originally same-day (ROC-AUC 0.691),
but that was mostly detecting current rain, not forecasting it. The weak
next-day signal here is likely why the Databricks v2 model uses lagged
precipitation features instead. Full rationale in `MODEL_CARD.md`.

To reproduce:
```bash
export NOAA_CDO_TOKEN="your_token_here"   # https://www.ncdc.noaa.gov/cdo-web/token
pip install requests pandas scikit-learn
python train_baseline.py
```
First run pulls and caches ~5 years of NOAA data (slow, rate-limited);
subsequent runs reuse the cache. Use `--refresh` to force a fresh pull.

## Getting Started

`00_data_ingestion.ipynb` is a Databricks notebook. To run it:

1. Get a free NOAA CDO token: https://www.ncdc.noaa.gov/cdo-web/token
2. Store it in a Databricks secret scope (never hardcode it in the notebook):
   ```bash
   databricks secrets create-scope mlo
   databricks secrets put-secret mlo WEATHER_API_KEY
   ```
3. Attach the notebook to a cluster/SQL warehouse with Unity Catalog access (needed for the Delta table writes in section 4), and update the `catalog` widget if you're not using `main`.
4. Adjust the remaining notebook widgets as needed: `location_id`, `start_date`/`end_date`, `lat`/`lon`, `nws_contact_email`, and `schema`.

The notebook pulls NOAA historical daily data (GHCND) and a live NWS forecast/observation snapshot, runs basic sanity checks, and persists both as bronze Delta tables.

