# Weather Prediction MLOps Pipeline

An end-to-end MLOps pipeline that predicts weather conditions (e.g. bad weather / rain) from historical and live weather data, built for the ADSP 32021 MLOps final project.

**Team:** Gabe Horas, Noah Ahmad, Lily Kendall, Payton Stewart, Diego

## Project Overview

The system pulls historical weather data for model training, tracks experiments, deploys a champion model as a live inference API, and monitors that model in production for drift. It's organized into four stages:

1. **Data Ingestion & Baselines** — pull historical + live weather data, define the target variable and evaluation metric, and establish a baseline model.
2. **Pipeline Automation & Experiment Tracking** — automate preprocessing/training (Airflow), version data/features, track experiments (MLflow), and register the winning model to a model registry.
3. **Containerization & Deployment** — wrap the registered model in a FastAPI service, containerize it with Docker, and deploy it for real-time predictions.
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
- **Airflow** for pipeline automation (preprocessing → training).
- **FastAPI + Docker** for serving the registered model.
- **EvidentlyAI / Prometheus + Grafana** (or custom code) for production monitoring and drift detection.

## Repository Structure

```
.
├── 01_data_ingestion.ipynb       # Stage 1: pulls NOAA CDO historical data and NWS live data,
│                                  # writes bronze Delta tables (Databricks)
├── FinalProjectProposal.html     # Original project proposal + data-pulling proof of concept
├── MLOps_Weather_Project_Task_Tracker.xlsx  # Task tracker across all four project phases
└── README.md
```

## Getting Started

`01_data_ingestion.ipynb` is a Databricks notebook. To run it:

1. Get a free NOAA CDO token: https://www.ncdc.noaa.gov/cdo-web/token
2. Store it in a Databricks secret scope (never hardcode it in the notebook):
   ```bash
   databricks secrets create-scope weather-mlops
   databricks secrets put-secret weather-mlops noaa_cdo_token
   ```
3. Attach the notebook to a cluster/SQL warehouse with Unity Catalog access (needed for the Delta table writes in section 4), and update the `catalog` widget if you're not using `main`.
4. Adjust the remaining notebook widgets as needed: `location_id`, `start_date`/`end_date`, `lat`/`lon`, `nws_contact_email`, and `schema`.

The notebook pulls NOAA historical daily data (GHCND) and a live NWS forecast/observation snapshot, runs basic sanity checks, and persists both as bronze Delta tables.

## Project Status

Currently in **Stage 1: Data Ingestion & Baselines**. Preprocessing, automated pipelines, MLflow tracking, deployment, and monitoring are planned for subsequent stages — see `MLOps_Weather_Project_Task_Tracker.xlsx` for the full task breakdown and status.
