# Model Card: Baseline Weather Quality Classifier

## Model Details

- **Model name**: `weather-quality-baseline`
- **Model type**: Logistic Regression, `solver="liblinear"`,
  `class_weight="balanced"`, standardized features (`StandardScaler`)
- **Version**: v0.2 (baseline, trained locally — target changed from
  same-day to next-day prediction; see Training Data below)
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
  (Chicago Midway Airport, IL), 2020-01-01 through 2024-12-30 (1,826 days
  after the last day is dropped — it has no "tomorrow" to label).
- **Features**: `AWND` (avg wind speed), `TMAX`, `TMIN`, all from the
  *current* day. This baseline is intentionally minimal.
- **Label**: `Bad_t+1 = 1 if next day's PRCP > 0.5mm else 0` — predicts
  **tomorrow's** weather from **today's** features, matching the target
  definition used by the Databricks v2 model (`03_baseline_model.ipynb`,
  which computes the same shift via a `lead()` window function). Originally
  this baseline predicted same-day weather from same-day features, which
  isn't real forecasting; shifted to next-day prediction so this baseline
  and the model it's meant to be a floor for are answering the same
  question. `PRCP > 0.5mm` itself was empirically validated as the most
  reliable available signal for this station — `WT**` coded weather-type
  flags were tried first and found too sparse.
- **Split**: chronological 80/20 — train on 2020-01-01 through 2023-12-30
  (1,460 rows), test on 2023-12-31 through 2024-12-30 (366 rows).
  Chronological rather than random, so the model isn't evaluated on dates
  that precede training data.
- **Class balance**: ~26% "Bad" / ~74% "Good" (imbalanced).
- **Live-data equivalent label** (not yet used in training): `Bad = 1 if
  precipitationLast3Hours.value > 0` from the NWS live feed — proposed for
  future work once the historical (daily) and live (sub-hourly) schemas are
  reconciled; they don't currently share a common feature set, so this
  baseline trains on historical data only.

## Evaluation Data

366 days (2023-12-31 through 2024-12-30), held out chronologically — see
Training Data above.

## Metrics

| Metric    | Value | Notes |
|-----------|-------|-------|
| Accuracy  | 0.503 | Barely above the ~74% majority-class floor's inverse — see interpretation below |
| Precision | 0.284 | Of predicted "Bad" days, 28% were actually bad |
| Recall    | 0.540 | Of actual "Bad" days, 54% were caught |
| F1        | 0.372 | Primary metric for imbalanced comparison |
| ROC-AUC   | 0.527 | Barely above 0.5 (random) — see interpretation below |

Confusion matrix (n=366): 130 true negatives, 136 false positives, 46 false
negatives, 54 true positives.

**Interpretation**: ROC-AUC of 0.527 means today's `AWND`/`TMAX`/`TMIN`
carry almost no predictive signal about tomorrow's rain — confirmed by the
fitted coefficients, which are all near zero (`AWND: +0.03`, `TMAX: -0.16`,
`TMIN: +0.36`, on standardized features). This is a real, if weak, result:
the *same-day* version of this baseline scored much higher (ROC-AUC 0.691),
but that was mostly detecting current rain rather than forecasting it —
an easier and less useful task than what's being measured here. The
weakness of this next-day signal is likely why the Databricks v2 model adds
lagged/rolling precipitation features (`PRCP_lag1`, `PRCP_roll3`) rather
than relying on same-day readings — plain temperature and wind don't appear
to be enough on their own.

`class_weight="balanced"` was used during training; despite that, recall
(0.540) is only modestly above precision (0.284) here, unlike the same-day
version's larger recall/precision gap — with almost no real signal in the
features, there's less for the class weighting to lean on.

## Practical Considerations

- Weather condition labels derived via keyword matching on
  `textDescription` may not generalize well if the upstream API changes its
  vocabulary 


