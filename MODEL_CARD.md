# Model Card: Baseline Weather Quality Classifier

## Model Details

- **Model name**: `weather-quality-baseline`
- **Model type**: Logistic Regression, `solver="liblinear"`,
  `class_weight="balanced"`, standardized features (`StandardScaler`)
- **Version**: v0.1 (baseline, trained locally)
- **Owner**: Gabe Horas — baseline model owner, MLOps final project
- **Framework**: scikit-learn (`sklearn.linear_model.LogisticRegression`)
- **Tracking**: Not yet tracked — trained and evaluated locally. MLflow
  integration (Databricks-hosted) planned for Stage 2.

## Intended Use

- **Primary use**: Serve as the benchmark model within the team's MLOps
  pipeline. Every subsequent model (Random Forest, XGBoost, or other
  candidates explored in the experimentation sandbox) is evaluated against
  this baseline's metrics before being considered for promotion.
- **Out of scope**: This model is not intended for production weather
  forecasting. We are using as benchmark.

## Training Data

- **Source**: NOAA GHCND daily records, station `GHCND:USW00014819`
  (Chicago Midway Airport, IL), 2020-01-01 through 2024-12-31 (1,827 days).
- **Features**: `AWND` (avg wind speed), `TMAX`, `TMIN`. This baseline is intentionally minimal.
- **Label**: `Bad = 1 if PRCP > 0.5mm else 0`. Empirically validated as the
  most reliable available signal for this station —
  `WT**` coded weather-type flags were tried first and found too sparse.
- **Split**: chronological 80/20 — train on 2020-01-01 through 2023-12-31
  (1,461 rows), test on all of 2024 (366 rows). Chronological rather than
  random, so the model isn't evaluated on dates that precede training data.
- **Class balance**: ~26% "Bad" / ~74% "Good" (imbalanced).
- **Live-data equivalent label** (not yet used in training): `Bad = 1 if
  precipitationLast3Hours.value > 0` from the NWS live feed — proposed for
  future work once the historical (daily) and live (sub-hourly) schemas are
  reconciled; they don't currently share a common feature set, so this
  baseline trains on historical data only.

## Evaluation Data

366 days of 2024 (Jan 1 – Dec 31), held out chronologically from the 2024
training set — see Training Data above.

## Metrics

| Metric    | Value | Notes |
|-----------|-------|-------|
| Accuracy  | 0.637 | Not the primary metric given class imbalance |
| Precision | 0.402 | Of predicted "Bad" days, 40% were actually bad |
| Recall    | 0.680 | Of actual "Bad" days, 68% were caught |
| F1        | 0.506 | Primary metric for imbalanced comparison |
| ROC-AUC   | 0.691 | |

Confusion matrix (n=366): 165 true negatives, 101 false positives, 32 false
negatives, 68 true positives.

`class_weight="balanced"` was used during training, which explains the
recall > precision skew — the model is tuned to catch bad-weather days at
the cost of more false alarms. 

## Practical Considerations

- Weather condition labels derived via keyword matching on
  `textDescription` may not generalize well if the upstream API changes its
  vocabulary 


