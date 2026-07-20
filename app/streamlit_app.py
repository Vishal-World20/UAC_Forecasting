"""
streamlit_app.py
------------------
Professional Streamlit dashboard for the HHS Unaccompanied Alien Children
(UAC) Program - Predictive Forecasting of Care Load & Placement Demand.

Run with:  streamlit run app/streamlit_app.py

Author: Data Science Project - UAC Predictive Forecasting
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ------------------------------------------------------------------
# Path setup so this app can be launched from any working directory
# ------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

sys.path.append(SRC_DIR)
from data_preprocessing import (
    load_raw_data, clean_data, build_continuous_daily_index,
    engineer_features, get_model_feature_columns, TARGET_COL, SECONDARY_TARGET_COL
)

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="HHS UAC Predictive Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Caching helpers
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and processing dataset...")
def get_data():
    raw = load_raw_data(os.path.join(DATA_DIR, "raw_data.csv"))
    cleaned = clean_data(raw)
    continuous = build_continuous_daily_index(cleaned)
    featured = engineer_features(continuous)
    return cleaned, continuous, featured


@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    discharge_path = os.path.join(MODELS_DIR, "discharge_model.pkl")
    feat_cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")

    if not os.path.exists(model_path):
        return None, None, None, None

    model = joblib.load(model_path)
    discharge_model = joblib.load(discharge_path) if os.path.exists(discharge_path) else None
    feature_cols = joblib.load(feat_cols_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, discharge_model, feature_cols, metadata


def recursive_forecast(model, featured_df, feature_cols, target_col, horizon,
                        discharge_model=None, secondary_col=SECONDARY_TARGET_COL):
    """Recursively forecast `horizon` future days beyond the end of featured_df."""
    history = featured_df.copy().reset_index(drop=True)
    preds_target = []
    preds_secondary = []
    future_dates = []

    for step in range(horizon):
        last_row = history.iloc[[-1]]
        next_date = pd.Timestamp(last_row["date"].values[0]) + pd.Timedelta(days=1)
        future_dates.append(next_date)

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

        dt = next_date
        new_row["day_of_week"] = dt.dayofweek
        new_row["day_of_month"] = dt.day
        new_row["month"] = dt.month
        new_row["quarter"] = dt.quarter
        new_row["year"] = dt.year
        new_row["is_weekend"] = int(dt.dayofweek >= 5)
        new_row["day_of_year"] = dt.dayofyear

        X_next = pd.DataFrame([new_row])[feature_cols]
        X_next = X_next.ffill(axis=1).fillna(0)

        pred_target = model.predict(X_next)[0]
        preds_target.append(pred_target)

        pred_secondary = None
        if discharge_model is not None:
            pred_secondary = discharge_model.predict(X_next)[0]
            preds_secondary.append(pred_secondary)

        approx_row = history.iloc[-1].copy()
        approx_row["date"] = next_date
        approx_row["hhs_care"] = pred_target
        approx_row["hhs_discharged"] = pred_secondary if pred_secondary is not None else history["hhs_discharged"].iloc[-7:].mean()
        approx_row["cbp_intake"] = history["cbp_intake"].iloc[-7:].mean()
        approx_row["cbp_active"] = history["cbp_active"].iloc[-7:].mean()
        approx_row["cbp_transferred_out"] = history["cbp_transferred_out"].iloc[-7:].mean()
        approx_row["net_pressure"] = approx_row["cbp_transferred_out"] - approx_row["hhs_discharged"]
        history = pd.concat([history, pd.DataFrame([approx_row])], ignore_index=True)

    result = pd.DataFrame({
        "date": future_dates,
        "forecast_hhs_care": np.clip(preds_target, a_min=0, a_max=None),
    })
    if preds_secondary:
        result["forecast_discharged"] = np.clip(preds_secondary, a_min=0, a_max=None)
    return result


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("UAC Forecasting Dashboard")
st.sidebar.markdown("**HHS Unaccompanied Alien Children Program**")
st.sidebar.markdown("Predictive Forecasting of Care Load & Placement Demand")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Exploratory Data Analysis", "Model Comparison",
     "Future Forecast", "Scenario / What-If", "About & Documentation"]
)

cleaned, continuous, featured = get_data()
model, discharge_model, feature_cols, metadata = get_models()

st.sidebar.markdown("---")
st.sidebar.caption(f"Data range: {continuous['date'].min().date()} → {continuous['date'].max().date()}")
st.sidebar.caption(f"Total daily records (post-cleaning): {len(continuous)}")
if metadata:
    st.sidebar.caption(f"Production model: **{metadata['best_model_name']}**")

# ==================================================================
# PAGE 1: OVERVIEW
# ==================================================================
if page == "Overview":
    st.title("HHS UAC Program — Predictive Forecasting Dashboard")
    st.markdown(
        "Forward-looking intelligence for **care load** and **discharge/placement demand**, "
        "built on daily CBP → HHS transfer and care data."
    )

    latest = continuous.iloc[-1]
    prev7 = continuous.iloc[-8]
    delta_care = latest["hhs_care"] - prev7["hhs_care"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Children Currently in HHS Care", f"{int(latest['hhs_care']):,}", f"{delta_care:+.0f} vs 7d ago")
    col2.metric("Children in CBP Custody", f"{int(latest['cbp_active']):,}")
    col3.metric("Daily CBP Intake (latest)", f"{int(latest['cbp_intake']):,}")
    col4.metric("Daily Discharges (latest)", f"{int(latest['hhs_discharged']):,}")

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(continuous["date"], continuous["hhs_care"], color="#1f4e79", linewidth=1.3)
    ax.fill_between(continuous["date"], continuous["hhs_care"], alpha=0.1, color="#1f4e79")
    ax.set_title("Children in HHS Care — Full History")
    ax.set_ylabel("Children in Care")
    ax.set_xlabel("Date")
    st.pyplot(fig)

    st.markdown("### Recent Trend (Last 90 Days)")
    recent = continuous.tail(90)
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 4))
    ax2[0].plot(recent["date"], recent["hhs_care"], color="#c0392b")
    ax2[0].set_title("HHS Care Load (90d)")
    ax2[0].tick_params(axis="x", rotation=30)

    ax2[1].plot(recent["date"], recent["cbp_intake"], label="Intake", color="#27ae60")
    ax2[1].plot(recent["date"], recent["hhs_discharged"], label="Discharged", color="#8e44ad")
    ax2[1].set_title("Daily Intake vs Discharge (90d)")
    ax2[1].legend()
    ax2[1].tick_params(axis="x", rotation=30)
    st.pyplot(fig2)

    st.info(
        "Use the sidebar to explore full EDA, compare forecasting models, generate "
        "future forecasts, and run custom what-if scenarios."
    )

# ==================================================================
# PAGE 2: EDA
# ==================================================================
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    st.markdown("### Dataset Summary")
    st.dataframe(cleaned.describe().T.style.format(precision=2), use_container_width=True)

    st.markdown("### Missing Data Before Cleaning (calendar gaps filled via interpolation)")
    full_range_len = (cleaned["date"].max() - cleaned["date"].min()).days + 1
    reported_days = len(cleaned)
    missing_days = full_range_len - reported_days
    c1, c2, c3 = st.columns(3)
    c1.metric("Calendar Days in Range", f"{full_range_len:,}")
    c2.metric("Days with Reported Data", f"{reported_days:,}")
    c3.metric("Gap Days (interpolated)", f"{missing_days:,}", f"{missing_days/full_range_len*100:.1f}%")

    st.markdown("### Time Series — All Metrics")
    metric_choice = st.multiselect(
        "Select metrics to plot",
        ["hhs_care", "cbp_active", "cbp_intake", "cbp_transferred_out", "hhs_discharged"],
        default=["hhs_care"]
    )
    if metric_choice:
        fig, ax = plt.subplots(figsize=(12, 5))
        for m in metric_choice:
            ax.plot(continuous["date"], continuous[m], label=m, linewidth=1.2)
        ax.legend()
        ax.set_title("Selected Metrics Over Time")
        st.pyplot(fig)

    st.markdown("### Distribution Analysis")
    dist_col = st.selectbox("Choose a column for distribution",
                             ["hhs_care", "cbp_active", "cbp_intake", "cbp_transferred_out", "hhs_discharged"])
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(continuous[dist_col], kde=True, ax=ax3[0], color="#2980b9")
    ax3[0].set_title(f"Distribution of {dist_col}")
    sns.boxplot(x=continuous[dist_col], ax=ax3[1], color="#e67e22")
    ax3[1].set_title(f"Boxplot of {dist_col}")
    st.pyplot(fig3)

    st.markdown("### Correlation Heatmap")
    corr_cols = ["hhs_care", "cbp_active", "cbp_intake", "cbp_transferred_out", "hhs_discharged"]
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    sns.heatmap(continuous[corr_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax4, fmt=".2f")
    st.pyplot(fig4)

    st.markdown("### Seasonality — Day of Week Pattern")
    dow_df = continuous.copy()
    dow_df["day_of_week"] = dow_df["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=dow_df, x="day_of_week", y="cbp_intake", order=order, ax=ax5, color="#16a085")
    ax5.set_title("CBP Intake by Day of Week")
    ax5.tick_params(axis="x", rotation=30)
    st.pyplot(fig5)

    st.markdown("### Net Pressure Indicator (Transfers In − Discharges Out)")
    net_pressure = continuous["cbp_transferred_out"] - continuous["hhs_discharged"]
    fig6, ax6 = plt.subplots(figsize=(12, 4))
    ax6.plot(continuous["date"], net_pressure.rolling(14).mean(), color="#d35400")
    ax6.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax6.set_title("14-Day Rolling Net Pressure (Positive = System Load Increasing)")
    st.pyplot(fig6)

# ==================================================================
# PAGE 3: MODEL COMPARISON
# ==================================================================
elif page == "Model Comparison":
    st.title("Model Comparison & Evaluation")

    comp_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    forecasts_path = os.path.join(REPORTS_DIR, "test_forecasts.csv")

    if not os.path.exists(comp_path):
        st.warning("No trained model comparison found. Please run `src/train_models.py` first "
                    "(see README.md for instructions).")
    else:
        comp_df = pd.read_csv(comp_path, index_col=0)
        st.markdown("### Metric Comparison Across Models (Lower = Better)")
        st.dataframe(comp_df.style.highlight_min(color="#d4edda", axis=0), use_container_width=True)

        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        for i, metric in enumerate(["MAE", "RMSE", "MAPE"]):
            comp_df[metric].sort_values().plot(kind="barh", ax=ax[i], color="#2980b9")
            ax[i].set_title(metric)
        plt.tight_layout()
        st.pyplot(fig)

        if os.path.exists(forecasts_path):
            st.markdown("### Actual vs Forecast — Holdout Test Period")
            fc_df = pd.read_csv(forecasts_path, parse_dates=["date"])
            model_options = [c for c in fc_df.columns if c not in ["date", "actual"]]
            selected_models = st.multiselect("Models to display", model_options, default=model_options[:3])

            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(fc_df["date"], fc_df["actual"], label="Actual", color="black", linewidth=2)
            colors = plt.cm.tab10.colors
            for i, m in enumerate(selected_models):
                ax2.plot(fc_df["date"], fc_df[m], label=m, linestyle="--", color=colors[i % len(colors)])
            ax2.legend()
            ax2.set_title("Holdout Test Set: Actual vs Model Forecasts")
            ax2.tick_params(axis="x", rotation=30)
            st.pyplot(fig2)

        if metadata:
            st.success(f"**Selected production model: {metadata['best_model_name']}** "
                       f"(lowest RMSE = {metadata['metrics'][metadata['best_model_name']]['RMSE']})")

# ==================================================================
# PAGE 4: FUTURE FORECAST
# ==================================================================
elif page == "Future Forecast":
    st.title("Future Care Load & Discharge Demand Forecast")

    if model is None:
        st.warning("No trained model found. Please run `src/train_models.py` first.")
    else:
        horizon = st.slider("Forecast horizon (days ahead)", min_value=7, max_value=60, value=14, step=1)
        model_note = "Random Forest / Gradient Boosting (recursive multi-step)" if metadata and metadata.get("best_model_is_ml") else "Gradient Boosting (interactive fallback)"
        st.caption(f"Using model: {model_note}")

        with st.spinner("Generating forecast..."):
            forecast_df = recursive_forecast(model, featured, feature_cols, TARGET_COL, horizon, discharge_model)

        # Simple uncertainty band using residual std from holdout metrics
        rmse = None
        if metadata and metadata.get("best_model_name") in metadata.get("metrics", {}):
            rmse = metadata["metrics"][metadata["best_model_name"]]["RMSE"]
        elif metadata:
            rmse = metadata["metrics"].get("Gradient Boosting", {}).get("RMSE")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(12, 5))
            recent_hist = continuous.tail(60)
            ax.plot(recent_hist["date"], recent_hist["hhs_care"], label="Historical", color="#1f4e79")
            ax.plot(forecast_df["date"], forecast_df["forecast_hhs_care"], label="Forecast", color="#c0392b", linestyle="--", marker="o", markersize=3)
            if rmse:
                upper = forecast_df["forecast_hhs_care"] + 1.96 * rmse
                lower = forecast_df["forecast_hhs_care"] - 1.96 * rmse
                ax.fill_between(forecast_df["date"], lower, upper, color="#c0392b", alpha=0.15, label="95% Confidence Interval")
            ax.axvline(continuous["date"].max(), color="gray", linestyle=":", linewidth=1)
            ax.legend()
            ax.set_title(f"Children in HHS Care — {horizon}-Day Forecast")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)

        with col2:
            st.markdown("### Forecast Summary")
            st.metric("Last Known Value", f"{int(continuous['hhs_care'].iloc[-1]):,}")
            st.metric(f"Forecast Day {horizon}", f"{int(forecast_df['forecast_hhs_care'].iloc[-1]):,}",
                      f"{forecast_df['forecast_hhs_care'].iloc[-1] - continuous['hhs_care'].iloc[-1]:+.0f}")
            if rmse:
                st.caption(f"Model RMSE (holdout): ±{rmse:.0f} children")

            # Capacity breach probability (simple heuristic KPI)
            capacity_threshold = st.number_input("Capacity threshold (children)", min_value=0,
                                                    value=int(continuous["hhs_care"].max() * 1.05), step=50)
            breach_days = (forecast_df["forecast_hhs_care"] > capacity_threshold).sum()
            breach_prob = breach_days / horizon * 100
            st.metric("Capacity Breach Probability", f"{breach_prob:.0f}%", f"{breach_days} of {horizon} days")

        if "forecast_discharged" in forecast_df.columns:
            st.markdown("### Discharge (Placement) Demand Forecast")
            fig3, ax3 = plt.subplots(figsize=(12, 4))
            ax3.bar(forecast_df["date"], forecast_df["forecast_discharged"], color="#27ae60")
            ax3.set_title(f"Forecasted Daily Discharges — Next {horizon} Days")
            ax3.tick_params(axis="x", rotation=30)
            st.pyplot(fig3)

        st.markdown("### Forecast Data Table")
        st.dataframe(forecast_df.style.format({"forecast_hhs_care": "{:.0f}", "forecast_discharged": "{:.0f}"}),
                     use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecast (CSV)", csv, "uac_forecast.csv", "text/csv")

# ==================================================================
# PAGE 5: SCENARIO / WHAT-IF
# ==================================================================
elif page == "Scenario / What-If":
    st.title("Scenario Analysis — What-If Prediction")
    st.markdown(
        "Manually set recent-condition inputs to predict tomorrow's HHS care load. "
        "Useful for stress-testing sudden surges in border activity."
    )

    if model is None:
        st.warning("No trained model found. Please run `src/train_models.py` first.")
    else:
        last_row = continuous.iloc[-1]
        c1, c2, c3 = st.columns(3)
        with c1:
            current_care = st.number_input("Current children in HHS care", value=int(last_row["hhs_care"]))
            avg_intake_7d = st.number_input("Avg. daily CBP intake (last 7d)", value=int(continuous["cbp_intake"].tail(7).mean()))
        with c2:
            avg_discharge_7d = st.number_input("Avg. daily discharges (last 7d)", value=int(continuous["hhs_discharged"].tail(7).mean()))
            avg_cbp_active = st.number_input("Avg. children in CBP custody (last 7d)", value=int(continuous["cbp_active"].tail(7).mean()))
        with c3:
            avg_transferred = st.number_input("Avg. daily CBP transfers (last 7d)", value=int(continuous["cbp_transferred_out"].tail(7).mean()))
            day_choice = st.selectbox("Day of week for prediction", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

        if st.button("Run Scenario Prediction", type="primary"):
            dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
            dow = dow_map[day_choice]
            next_date = continuous["date"].max() + pd.Timedelta(days=1)

            scenario_row = {c: 0 for c in feature_cols}
            scenario_row["day_of_week"] = dow
            scenario_row["is_weekend"] = int(dow >= 5)
            scenario_row["day_of_month"] = next_date.day
            scenario_row["month"] = next_date.month
            scenario_row["quarter"] = next_date.quarter
            scenario_row["year"] = next_date.year
            scenario_row["day_of_year"] = next_date.dayofyear

            for lag in (1, 7, 14):
                scenario_row[f"hhs_care_lag_{lag}"] = current_care
                scenario_row[f"hhs_discharged_lag_{lag}"] = avg_discharge_7d
                scenario_row[f"cbp_intake_lag_{lag}"] = avg_intake_7d
                scenario_row[f"cbp_active_lag_{lag}"] = avg_cbp_active
                scenario_row[f"cbp_transferred_out_lag_{lag}"] = avg_transferred
            for w in (7, 14):
                scenario_row[f"hhs_care_rollmean_{w}"] = current_care
                scenario_row[f"hhs_care_rollstd_{w}"] = 20
                scenario_row[f"hhs_discharged_rollmean_{w}"] = avg_discharge_7d
                scenario_row[f"hhs_discharged_rollstd_{w}"] = 3
                scenario_row[f"cbp_intake_rollmean_{w}"] = avg_intake_7d
                scenario_row[f"cbp_intake_rollstd_{w}"] = 3
                scenario_row[f"cbp_active_rollmean_{w}"] = avg_cbp_active
                scenario_row[f"cbp_active_rollstd_{w}"] = 5
                scenario_row[f"cbp_transferred_out_rollmean_{w}"] = avg_transferred
                scenario_row[f"cbp_transferred_out_rollstd_{w}"] = 3
            scenario_row["net_pressure_lag_1"] = avg_transferred - avg_discharge_7d
            scenario_row["net_pressure_roll7"] = avg_transferred - avg_discharge_7d

            X_scenario = pd.DataFrame([scenario_row])[feature_cols]
            pred = max(0, model.predict(X_scenario)[0])

            st.markdown("---")
            colr1, colr2 = st.columns(2)
            colr1.metric("Predicted HHS Care Load (Next Day)", f"{int(pred):,}",
                         f"{pred - current_care:+.0f} vs current")
            if discharge_model is not None:
                pred_disc = max(0, discharge_model.predict(X_scenario)[0])
                colr2.metric("Predicted Discharges (Next Day)", f"{int(pred_disc):,}")

            st.caption("This is a single-step, static what-if scenario intended for stress-testing and planning "
                       "discussions — not a substitute for the full recursive multi-day forecast.")

# ==================================================================
# PAGE 6: ABOUT
# ==================================================================
elif page == "About & Documentation":
    st.title("About This Project")
    st.markdown("""
This dashboard implements **Predictive Forecasting of Care Load & Placement Demand**
for the U.S. Department of Health and Human Services (HHS) Unaccompanied Alien
Children (UAC) Program, addressing the objectives defined in the project brief:

**Primary Objectives**
- Forecast the number of children in HHS care
- Estimate future imbalance between intake and exits (net pressure)
- Predict short-term discharge demand

**Secondary Objectives**
- Early-warning indicators for capacity stress (capacity breach probability)
- Quantified forecast uncertainty (confidence intervals from holdout RMSE)
- Statistical vs. Machine-Learning model comparison

**Methodology**
1. Data cleaning — comma/thousands-separator handling, date parsing, deduplication
2. Continuous daily reindexing with interpolation (source data skips weekends/holidays)
3. Feature engineering — lag features (1/7/14 day), rolling mean/std (7/14 day),
   calendar effects, and a net-pressure flow signal
4. Strict time-based train/test split with a 30-day holdout — no random sampling
5. Model training — Naive Persistence & Moving Average baselines; ARIMA, SARIMA
   and Exponential Smoothing statistical models; Linear Regression, Random Forest
   and Gradient Boosting ML models
6. Best model selected by holdout RMSE and saved for production use

**Data leakage control**: same-day (contemporaneous) CBP metrics are excluded from
the ML feature set — only their lagged/rolling versions are used, since these are
the only values realistically known at forecast time.
""")
    if metadata:
        st.markdown("### Current Production Model Metadata")
        st.json(metadata)

    st.markdown("### Project Structure")
    st.code("""
UAC_Forecasting_Project/
├── app/streamlit_app.py        # This dashboard
├── data/raw_data.csv           # Source dataset
├── models/                     # Saved trained models
├── notebooks/                  # Jupyter notebook (EDA + modeling)
├── reports/                    # Model comparison & forecast CSVs
├── src/                        # Reusable pipeline code
├── requirements.txt
├── run.bat
└── README.md
    """, language="text")
