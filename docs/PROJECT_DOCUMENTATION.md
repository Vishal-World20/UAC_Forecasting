# Project Documentation
## Predictive Forecasting of Care Load & Placement Demand — HHS UAC Program

---

### 1. Background & Context

The UAC Program operates in a high-uncertainty environment, where sudden changes
in border activity, policy enforcement, or humanitarian crises can rapidly increase
the number of children entering federal care. This project moves the program from
**descriptive** analytics (what happened) to **predictive** analytics (what will
happen), enabling proactive healthcare and child-welfare planning.

### 2. Problem Statement

Despite high-quality daily time-series data, the program lacked:
- Short-term forecasts of children in HHS care
- Predictive estimates of discharge (placement) demand
- Early-warning indicators of upcoming capacity stress

### 3. Objectives

**Primary**
- Forecast the number of children in HHS care
- Estimate future imbalance between intake and exits
- Predict short-term discharge demand

**Secondary**
- Provide early warnings for healthcare planners
- Quantify forecast uncertainty
- Compare statistical vs. machine-learning forecasting approaches

### 4. Dataset

| Column | Renamed | Description |
|---|---|---|
| Date | `date` | Reporting date |
| Children apprehended and placed in CBP custody | `cbp_intake` | Daily intake volume |
| Children in CBP custody | `cbp_active` | Active CBP care load |
| Children transferred out of CBP custody | `cbp_transferred_out` | Flow into HHS system |
| Children in HHS Care | `hhs_care` | Active HHS care load (**primary target**) |
| Children discharged from HHS Care | `hhs_discharged` | Successful sponsor placements (**secondary target**) |

Source rows span **2023-01-12 to 2025-12-21** (720 reported rows before continuous
reindexing; the source does not report every calendar day).

### 5. Methodology

#### 5.1 Time-Series Preparation
- Parsed `Date` into a proper datetime index
- Converted comma-formatted numeric strings (e.g. `"2,484"`) to numeric
- Reindexed onto a continuous daily calendar; gaps filled via linear interpolation
- Decomposed the `hhs_care` series into trend / seasonality / residual components
  (see notebook, Section 4) using additive decomposition with a 7-day seasonal period
- Verified stationarity via the Augmented Dickey-Fuller test

#### 5.2 Feature Engineering
- **Lag features**: t-1, t-7, t-14 for `hhs_care`, `hhs_discharged`, `cbp_intake`,
  `cbp_active`, `cbp_transferred_out`
- **Rolling statistics**: 7-day and 14-day rolling mean/std (computed on shifted
  series to prevent look-ahead leakage)
- **Flow-based signal**: `net_pressure` = `cbp_transferred_out` − `hhs_discharged`
  (positive = system load increasing)
- **Calendar effects**: day of week, day of month, month, quarter, year,
  weekend flag, day of year

**Leakage control**: Same-day (contemporaneous) raw exogenous columns
(`cbp_intake`, `cbp_active`, `cbp_transferred_out`, `net_pressure`) are excluded
from the ML feature matrix. Only their lagged/rolling derivatives are used, since
same-day values would not be known in advance in a real forecasting deployment.

#### 5.3 Train/Test Strategy
- Strict **time-based split** — no random sampling
- Last **30 days** held out as the test set
- ML models use **recursive multi-step forecasting**: predict one day, feed the
  prediction back into the feature history, repeat for the full horizon

#### 5.4 Models Trained

| Category | Model | Notes |
|---|---|---|
| Baseline | Naive Persistence | Last observed value repeated |
| Baseline | Moving Average (7d) | Simple rolling mean benchmark |
| Statistical | ARIMA(2,1,2) | Captures trend/autocorrelation |
| Statistical | SARIMA(1,1,1)x(1,0,1,7) | Adds weekly seasonality |
| Statistical | Holt-Winters Exponential Smoothing | Additive trend + weekly seasonality |
| ML | Linear Regression | Simple ML baseline |
| ML | Random Forest Regressor | 300 trees, max depth 10 |
| ML | Gradient Boosting Regressor | 300 estimators, depth 3, lr 0.05 |

#### 5.5 Evaluation Metrics
- **MAE** (Mean Absolute Error) — average absolute forecast miss
- **RMSE** (Root Mean Squared Error) — penalizes large errors more heavily
- **MAPE** (Mean Absolute Percentage Error) — relative error, easy to communicate

The model with the **lowest RMSE** on the holdout set is selected as the
production model, then **retrained on the full dataset** before being saved to
`models/best_model.pkl`.

### 6. Key Performance Indicators (KPIs)

| KPI | Definition |
|---|---|
| Forecast Accuracy (%) | 100 − MAPE |
| Surge Lead Time | Days of advance warning before a capacity threshold is crossed |
| Capacity Breach Probability | % of forecast horizon days exceeding a user-defined threshold |
| Forecast Stability Index | Inverse of forecast variance (robustness proxy) |

### 7. Streamlit Application

Six pages: Overview, Exploratory Data Analysis, Model Comparison, Future Forecast
(adjustable horizon + confidence intervals + capacity breach probability + CSV
export), Scenario / What-If (manual input prediction), and About & Documentation.

### 8. Deliverables

- `notebooks/UAC_Forecasting_Analysis.ipynb` — full research notebook (EDA,
  decomposition, feature engineering, all models, evaluation, KPIs)
- `app/streamlit_app.py` — live analytics dashboard
- `docs/EXECUTIVE_SUMMARY.md` — non-technical summary for government stakeholders
- `models/`, `reports/` — saved production model and evaluation artifacts

### 9. Limitations & Future Work

- Forecasts assume the historical relationship between intake, transfer, and
  discharge processes continues to hold; abrupt policy or enforcement shifts are
  not predictable from historical data alone.
- Recursive multi-step ML forecasting accumulates error over long horizons;
  forecasts beyond ~30 days should be treated as directional.
- Future work could incorporate external variables (policy change dates,
  regional border-crossing statistics, shelter-capacity data) as additional
  features, and explore probabilistic/Bayesian forecasting for sharper
  uncertainty quantification.
