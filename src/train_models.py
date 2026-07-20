"""
train_models.py
-----------------
Trains and compares multiple forecasting models for HHS UAC "Children in
HHS Care" (primary target) using a strict time-based train/test split and
walk-forward-style multi-horizon evaluation. Saves the best model + metadata
to the /models directory for use by the Streamlit app.

Models trained:
  Baseline   : Naive Persistence, Moving Average
  Statistical: ARIMA / SARIMA, Exponential Smoothing (Holt-Winters)
  ML         : Random Forest Regressor, Gradient Boosting Regressor,
               Linear Regression (extra baseline)

Author: Data Science Project - UAC Predictive Forecasting
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(__file__))
from data_preprocessing import (
    load_raw_data, clean_data, build_continuous_daily_index,
    engineer_features, get_model_feature_columns, TARGET_COL
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TEST_HORIZON_DAYS = 30   # last 30 days held out as test set
RANDOM_STATE = 42


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Avoid division by zero in MAPE
    denom = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return {"MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3), "MAPE": round(float(mape), 3)}


# ------------------------------------------------------------------
# Baseline models
# ------------------------------------------------------------------
def naive_persistence_forecast(train_series, horizon):
    last_val = train_series.iloc[-1]
    return np.repeat(last_val, horizon)


def moving_average_forecast(train_series, horizon, window=7):
    avg = train_series.iloc[-window:].mean()
    return np.repeat(avg, horizon)


# ------------------------------------------------------------------
# Statistical models (ARIMA / SARIMA / Exponential Smoothing)
# ------------------------------------------------------------------
def arima_forecast(train_series, horizon):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train_series, order=(2, 1, 2))
    fit = model.fit()
    forecast = fit.forecast(steps=horizon)
    return np.array(forecast), fit


def sarima_forecast(train_series, horizon):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(
        train_series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False
    )
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=horizon)
    return np.array(forecast), fit


def exp_smoothing_forecast(train_series, horizon):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(
        train_series, trend="add", seasonal="add", seasonal_periods=7,
        initialization_method="estimated"
    )
    fit = model.fit()
    forecast = fit.forecast(horizon)
    return np.array(forecast), fit


# ------------------------------------------------------------------
# ML models (supervised tabular approach using lag/rolling features)
# ------------------------------------------------------------------
def train_ml_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def recursive_ml_forecast(model, df, feature_cols, target_col, start_idx, horizon):
    """
    Recursive multi-step forecasting for tabular ML models: predict one step,
    append it to history, recompute lag/rolling features, predict next step.
    This mirrors real deployment where future exogenous values are unknown.
    """
    history = df.iloc[:start_idx].copy().reset_index(drop=True)
    preds = []

    for step in range(horizon):
        last_row = history.iloc[[-1]].copy()
        # Build next day's feature row from current history
        next_date = last_row["date"].values[0] + np.timedelta64(1, "D")

        # Recompute lag features from history's target column values
        new_row = {"date": next_date}
        for col in ["hhs_care", "hhs_discharged", "cbp_intake", "cbp_active", "cbp_transferred_out"]:
            series = history[col]
            for lag in (1, 7, 14):
                new_row[f"{col}_lag_{lag}"] = series.iloc[-lag] if len(series) >= lag else series.iloc[-1]
            for w in (7, 14):
                window_vals = series.iloc[-w:]
                new_row[f"{col}_rollmean_{w}"] = window_vals.mean()
                new_row[f"{col}_rollstd_{w}"] = window_vals.std() if len(window_vals) > 1 else 0.0

        new_row["net_pressure_lag_1"] = history["net_pressure"].iloc[-1]
        new_row["net_pressure_roll7"] = history["net_pressure"].iloc[-7:].mean()

        dt = pd.Timestamp(next_date)
        new_row["day_of_week"] = dt.dayofweek
        new_row["day_of_month"] = dt.day
        new_row["month"] = dt.month
        new_row["quarter"] = dt.quarter
        new_row["year"] = dt.year
        new_row["is_weekend"] = int(dt.dayofweek >= 5)
        new_row["day_of_year"] = dt.dayofyear

        X_next = pd.DataFrame([new_row])[feature_cols]
        X_next = X_next.ffill(axis=1).fillna(0)
        pred = model.predict(X_next)[0]
        preds.append(pred)

        # Append predicted target back into history to allow next-step lag computation
        approx_row = history.iloc[-1].copy()
        approx_row["date"] = next_date
        approx_row["hhs_care"] = pred
        # carry-forward assumption for exogenous flow columns (unknown at forecast time)
        approx_row["hhs_discharged"] = history["hhs_discharged"].iloc[-7:].mean()
        approx_row["cbp_intake"] = history["cbp_intake"].iloc[-7:].mean()
        approx_row["cbp_active"] = history["cbp_active"].iloc[-7:].mean()
        approx_row["cbp_transferred_out"] = history["cbp_transferred_out"].iloc[-7:].mean()
        approx_row["net_pressure"] = approx_row["cbp_transferred_out"] - approx_row["hhs_discharged"]
        history = pd.concat([history, pd.DataFrame([approx_row])], ignore_index=True)

    return np.clip(np.array(preds), a_min=0, a_max=None)


# ------------------------------------------------------------------
# Main training routine
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("UAC PREDICTIVE FORECASTING - MODEL TRAINING PIPELINE")
    print("=" * 70)

    raw_path = os.path.join(DATA_DIR, "raw_data.csv")
    raw = load_raw_data(raw_path)
    cleaned = clean_data(raw)
    continuous = build_continuous_daily_index(cleaned)
    featured = engineer_features(continuous)

    cleaned.to_csv(os.path.join(DATA_DIR, "cleaned_data.csv"), index=False)
    featured.to_csv(os.path.join(DATA_DIR, "featured_data.csv"), index=False)
    print(f"Data ready: {featured.shape[0]} rows, {featured.shape[1]} columns")

    target_series_full = continuous.set_index("date")[TARGET_COL]

    split_idx = len(featured) - TEST_HORIZON_DAYS
    train_df = featured.iloc[:split_idx].reset_index(drop=True)
    test_df = featured.iloc[split_idx:].reset_index(drop=True)

    y_test_true = test_df[TARGET_COL].values
    horizon = len(test_df)
    print(f"Train size: {len(train_df)} | Test (holdout) horizon: {horizon} days")

    feature_cols = get_model_feature_columns(featured)

    results = {}
    forecasts = {}

    # --- Baseline: Naive persistence ---
    train_series_target = train_df.set_index("date")[TARGET_COL]
    preds = naive_persistence_forecast(train_series_target, horizon)
    results["Naive Persistence"] = compute_metrics(y_test_true, preds)
    forecasts["Naive Persistence"] = preds.tolist()
    print("Trained: Naive Persistence")

    # --- Baseline: Moving Average ---
    preds = moving_average_forecast(train_series_target, horizon, window=7)
    results["Moving Average (7d)"] = compute_metrics(y_test_true, preds)
    forecasts["Moving Average (7d)"] = preds.tolist()
    print("Trained: Moving Average (7d)")

    # --- Statistical: ARIMA ---
    try:
        preds, arima_fit = arima_forecast(train_series_target, horizon)
        results["ARIMA"] = compute_metrics(y_test_true, preds)
        forecasts["ARIMA"] = preds.tolist()
        print("Trained: ARIMA(2,1,2)")
    except Exception as e:
        print(f"ARIMA failed: {e}")

    # --- Statistical: SARIMA ---
    try:
        preds, sarima_fit = sarima_forecast(train_series_target, horizon)
        results["SARIMA"] = compute_metrics(y_test_true, preds)
        forecasts["SARIMA"] = preds.tolist()
        print("Trained: SARIMA(1,1,1)x(1,0,1,7)")
    except Exception as e:
        print(f"SARIMA failed: {e}")

    # --- Statistical: Exponential Smoothing ---
    try:
        preds, es_fit = exp_smoothing_forecast(train_series_target, horizon)
        results["Exponential Smoothing"] = compute_metrics(y_test_true, preds)
        forecasts["Exponential Smoothing"] = preds.tolist()
        print("Trained: Holt-Winters Exponential Smoothing")
    except Exception as e:
        print(f"Exponential Smoothing failed: {e}")

    # --- ML: Linear Regression ---
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    lr = train_ml_model(LinearRegression(), X_train, y_train)
    preds = recursive_ml_forecast(lr, featured, feature_cols, TARGET_COL, split_idx, horizon)
    results["Linear Regression"] = compute_metrics(y_test_true, preds)
    forecasts["Linear Regression"] = preds.tolist()
    print("Trained: Linear Regression")

    # --- ML: Random Forest ---
    rf = train_ml_model(
        RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        X_train, y_train
    )
    preds = recursive_ml_forecast(rf, featured, feature_cols, TARGET_COL, split_idx, horizon)
    results["Random Forest"] = compute_metrics(y_test_true, preds)
    forecasts["Random Forest"] = preds.tolist()
    print("Trained: Random Forest Regressor")

    # --- ML: Gradient Boosting ---
    gb = train_ml_model(
        GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE),
        X_train, y_train
    )
    preds = recursive_ml_forecast(gb, featured, feature_cols, TARGET_COL, split_idx, horizon)
    results["Gradient Boosting"] = compute_metrics(y_test_true, preds)
    forecasts["Gradient Boosting"] = preds.tolist()
    print("Trained: Gradient Boosting Regressor")

    # ------------------------------------------------------------------
    # Select best model by RMSE
    # ------------------------------------------------------------------
    best_model_name = min(results, key=lambda k: results[k]["RMSE"])
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (lower is better)")
    print("=" * 70)
    comp_df = pd.DataFrame(results).T.sort_values("RMSE")
    print(comp_df.to_string())
    print(f"\nBEST MODEL: {best_model_name}")

    # Save comparison table
    comp_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))

    # Save test-set forecasts for dashboard visualization
    forecast_compare_df = pd.DataFrame({"date": test_df["date"].values, "actual": y_test_true})
    for name, vals in forecasts.items():
        forecast_compare_df[name] = vals
    forecast_compare_df.to_csv(os.path.join(REPORTS_DIR, "test_forecasts.csv"), index=False)

    # ------------------------------------------------------------------
    # Retrain best model (if ML) on FULL data for production use
    # ------------------------------------------------------------------
    model_registry = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE),
    }

    is_ml_best = best_model_name in model_registry
    if is_ml_best:
        final_model = model_registry[best_model_name]
        final_model.fit(featured[feature_cols], featured[TARGET_COL])
        joblib.dump(final_model, os.path.join(MODELS_DIR, "best_model.pkl"))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
        print(f"\nSaved production ML model -> models/best_model.pkl")
    else:
        # Save a fallback Gradient Boosting model too, since Streamlit app needs
        # an interactive "what-if" ML model regardless of which stat model won on RMSE.
        final_model = model_registry["Gradient Boosting"]
        final_model.fit(featured[feature_cols], featured[TARGET_COL])
        joblib.dump(final_model, os.path.join(MODELS_DIR, "best_model.pkl"))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
        print(f"\nBest model was statistical ({best_model_name}); Gradient Boosting "
              f"saved as the interactive ML model for the dashboard.")

    # Also train a secondary model to forecast discharge demand (secondary objective)
    from data_preprocessing import SECONDARY_TARGET_COL
    disc_features = [c for c in feature_cols]  # reuse same features
    gb_discharge = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE)
    gb_discharge.fit(featured[disc_features], featured[SECONDARY_TARGET_COL])
    joblib.dump(gb_discharge, os.path.join(MODELS_DIR, "discharge_model.pkl"))
    print("Saved discharge-demand model -> models/discharge_model.pkl")

    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "best_model_is_ml": is_ml_best,
        "production_model_file": "best_model.pkl",
        "target_column": TARGET_COL,
        "test_horizon_days": TEST_HORIZON_DAYS,
        "feature_columns": feature_cols,
        "metrics": results,
        "data_shape": list(featured.shape),
        "date_range": [str(featured["date"].min().date()), str(featured["date"].max().date())],
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata -> models/metadata.json")

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()
