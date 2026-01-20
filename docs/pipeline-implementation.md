# Phase 1: Temperature Data Infrastructure (2–3 weeks)

## Temperature-focused data pipelines

- Build pipelines **only for temperature variables**:
    - 2m air temperature
    - Daily max (Tmax) / min (Tmin)
    - Hourly temperature profiles
- API connectors scoped to:
    - Open-Meteo (hourly + daily temperature)
    - NOAA (station-level historical temps)
    - ECMWF (ensemble temperature forecasts)
- Normalize units (°F/°C), timestamps, and station metadata
- Handle station drift, missing hours, and sensor bias

## Temperature-Specific Feature Engineering

### Core Forecast Features

- **Predicted daily Tmax (`tmax_pred_f`)**

    Raw daily maximum temperature forecast from Open-Meteo

- **Lead time (`lead_hours`)**

    Hours between forecast issue time and target local calendar day

- **Forecast source**

    Identifier for forecast provider (e.g., `openmeteo`)


---

### Seasonal & Calendar Features

- **Day-of-year encoding (`sin_doy`, `cos_doy`)**

    Captures seasonal patterns in forecast bias

- **Month**

    Coarse seasonal regime indicator


---

### Forecast Bias & Error History

Rolling statistics computed **only from past forecast residuals** (no future leakage):

- **Rolling bias:** 7-day, 14-day, 30-day mean forecast error
- **Rolling error:** 14-day and 30-day RMSE
- **Lead-time uncertainty (`sigma_lead`)**

    Historical residual standard deviation for the given lead time


---

### Static Station Metadata

Leakage-free location corrections derived from `stations.csv`:

- **Elevation**
- **Distance to coast**
- **Urban heat proxy**

(Optional interaction features with seasonality or lead time.)

## Temperature evaluation framework

- Historical temperature loader (station-aligned)
- Metrics focused on temperature accuracy:
    - MAE (°F)
    - RMSE
    - % within ±1°F / ±2°F
    - Calibration coverage (interval accuracy)
- Sliced analysis:
    - By month/season
    - By lead time bucket
    - By temperature regime
- Visualization:
    - Error vs lead-time
    - Bias plots (systematic warm/cool error)
    - Calibration diagrams

---

# Phase 2: Temperature ML Model Development (4–6 weeks)

## Baseline temperature models

- Persistence model (last observed temp)
- Linear / Ridge regression on engineered temp features
- Simple kNN on recent temperature trajectories
- Establish **temperature-only benchmarks**

## Advanced temperature models

- Sequence models trained on **hourly temperature series**:
    - LSTM / GRU (24–120h input windows)
- Tree-based models for Tmax/Tmin:
    - XGBoost / Random Forest
- Separate models for:
    - Hourly temp
    - Daily Tmax
    - Daily Tmin
- Hyperparameter tuning targeting **MAE minimization**
- Model versioning with temperature-specific metadata

## Temperature ensemble architecture

- Stacked ensemble with meta-learner predicting:
    - Final temperature estimate
    - Prediction uncertainty (σ)
- Dynamic weighting based on:
    - Recent MAE
    - Bias drift (warm/cool skew)
- Fallback logic:
    - If volatility spikes → favor ECMWF ensemble mean
    - If station data drops → forecast-only mode

## Validation (temperature-specific)

- Walk-forward validation by season
- Hold-out extreme temp periods (heat waves, cold snaps)
- Performance by regime:
    - Normal days vs tail events
- Ensure no leakage across forecast horizons

---

# Phase 3: Real-Time Temperature Prediction System (2–3 weeks)

## Inference optimization

- Optimize models for **fast temperature inference**
- ONNX runtime for tree + neural models
- Cache repeated feature windows
- Target: <500ms per temperature update

## Real-time temperature ingestion

- Live hourly temperature updates
- Nowcasting layer:
    - Interpolate sub-hour temps
    - Adjust short-term forecasts using latest observations
- Multi-source temperature fusion:
    - Station-weighted blending
    - Forecast bias correction in real time

## Temperature monitoring

- Drift detection on:
    - Mean error
    - Bias direction (warm vs cool)
- Dashboards:
    - Live MAE
    - Error distribution over time
- Alerts:
    - Sudden error spikes
    - Model disagreement beyond threshold

---

# Phase 4: Testing & Optimization (2–3 weeks)

## Temperature evaluation runs

- Run full system on historical + live temps
- Track:
    - Forecast error analysis
- Diagnose:
    - Overconfidence in narrow ranges
    - Systematic warm/cool bias

## Gradual deployment

- Start with **single-station**
- A/B test:
    - Different uncertainty thresholds
    - Different ensemble weighting schemes
- Continuous retraining triggered by MAE drift

## Documentation & observability

- Temperature-specific docs:
    - Feature definitions
    - Error interpretations
- Logs:
    - Raw temp → features → forecast
- Performance dashboards tied directly to °F error

---

# Key Decision Points (Temperature-Specific)

- **After Phase 2:**

    Ensemble must achieve:

    - MAE ≤ 1.5°F (hourly) or ≤ 2°F (daily Tmax)
    - Calibrated 90% prediction intervals

- **After Phase 3:**

    End-to-end temperature inference latency <500ms

- **Phase 4 guardrails:**

    Halt if:

    - MAE worsens by >25% vs baseline
    - Bias exceeds ±2°F for 3 consecutive days
