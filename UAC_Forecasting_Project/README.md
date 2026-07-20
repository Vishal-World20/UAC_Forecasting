# HHS UAC Program — Predictive Forecasting of Care Load & Placement Demand

A complete, ready-to-run data science project that forecasts the number of children
in HHS care and future discharge (placement) demand for the Unaccompanied Alien
Children (UAC) Program, using daily CBP-to-HHS transfer data.

Built for: **U.S. Department of Health and Human Services / Unified Mentor Program**

---

## Problem Type

**Time-Series Forecasting (Regression)** — the dataset is a daily time series of
five operational metrics. The project forecasts:
1. **Primary target**: `Children in HHS Care` (care load)
2. **Secondary target**: `Children discharged from HHS Care` (placement/discharge demand)

---

## Quick Start (Windows)

1. Extract this ZIP anywhere on your machine.
2. Make sure Python 3.9+ is installed and on your PATH.
3. Double-click **`run.bat`** (or run it from Command Prompt).

That's it — `run.bat` will:
- Install all dependencies from `requirements.txt`
- Run the data cleaning / preprocessing pipeline
- Train and compare all forecasting models
- Launch the Streamlit dashboard in your browser

## Quick Start (Mac / Linux)

```bash
chmod +x run.sh
./run.sh
```

## Manual Setup (any OS)

```bash
pip install -r requirements.txt
python src/data_preprocessing.py     # clean data + engineer features
python src/train_models.py           # train, compare, and save the best model
streamlit run app/streamlit_app.py   # launch the dashboard
```

The Jupyter notebook (full EDA + modeling walkthrough) can be opened with:

```bash
jupyter notebook notebooks/UAC_Forecasting_Analysis.ipynb
```

---

## Folder Structure

```
UAC_Forecasting_Project/
├── app/
│   └── streamlit_app.py          # Interactive dashboard (6 pages)
├── data/
│   ├── raw_data.csv              # Original uploaded dataset
│   ├── cleaned_data.csv          # Cleaned + feature-engineered output
│   └── featured_data.csv
├── models/
│   ├── best_model.pkl            # Production forecasting model
│   ├── discharge_model.pkl       # Secondary model (discharge demand)
│   ├── feature_columns.pkl       # Feature list used at inference time
│   └── metadata.json             # Metrics, model choice, data summary
├── notebooks/
│   └── UAC_Forecasting_Analysis.ipynb   # Full EDA + modeling notebook
├── reports/
│   ├── model_comparison.csv      # All model metrics side by side
│   └── test_forecasts.csv        # Holdout actual vs. predicted values
├── src/
│   ├── data_preprocessing.py     # Cleaning + feature engineering (shared module)
│   └── train_models.py           # Model training & selection pipeline
├── docs/
│   ├── PROJECT_DOCUMENTATION.md  # Full technical documentation
│   └── EXECUTIVE_SUMMARY.md      # Non-technical summary for stakeholders
├── requirements.txt
├── run.bat                       # Windows one-click setup + launch
├── run.sh                        # Mac/Linux one-click setup + launch
└── README.md                     # This file
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | Key metrics, latest care load, recent trend charts |
| **Exploratory Data Analysis** | Distributions, correlations, seasonality, missing-data summary |
| **Model Comparison** | MAE / RMSE / MAPE across all models, actual-vs-forecast holdout chart |
| **Future Forecast** | Adjustable horizon (7–60 days), confidence interval, capacity breach probability, CSV download |
| **Scenario / What-If** | Manually set recent conditions to predict next-day care load and discharges |
| **About & Documentation** | Methodology summary and current production model metadata |

---

## Models Trained & Compared

| Category | Models |
|---|---|
| Baseline | Naive Persistence, Moving Average (7-day) |
| Statistical | ARIMA, SARIMA (weekly seasonality), Holt-Winters Exponential Smoothing |
| Machine Learning | Linear Regression, Random Forest Regressor, Gradient Boosting Regressor |

The best model is automatically selected by **lowest RMSE** on a strict 30-day
time-based holdout set, then retrained on the full dataset and saved for
production use in the dashboard.

## Evaluation Metrics
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error (penalizes large misses)
- **MAPE** — Mean Absolute Percentage Error

## KPIs Surfaced in the Dashboard
- Forecast Accuracy (%) = 100 − MAPE
- Capacity Breach Probability (% of forecast days exceeding a user-set threshold)
- 95% confidence interval bands around every forecast (based on holdout RMSE)

---

## Notes on Data Handling

- The source CSV stores large numbers with thousands-separator commas (e.g. `"2,484"`)
  inside quoted strings — these are parsed and converted to numeric automatically.
- HHS does not publish data every calendar day (weekends/holidays are frequently
  skipped). The pipeline reindexes onto a continuous daily calendar and linearly
  interpolates gaps so that lag/rolling features and time-series models are valid.
- Same-day (contemporaneous) exogenous columns are **excluded** from the ML feature
  set to avoid data leakage — only their lagged/rolling versions are used, since
  that mirrors what would actually be known at real forecast time.

---

## Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `scikit-learn`,
`statsmodels`, `matplotlib`, `seaborn`, `joblib`, `streamlit`, `jupyter`.

## Retraining

To retrain on updated data, replace `data/raw_data.csv` with a new export in the
same column format, then re-run `python src/train_models.py`. The dashboard will
automatically pick up the newly saved model on next launch.
