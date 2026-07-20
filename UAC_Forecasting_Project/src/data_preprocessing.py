"""
data_preprocessing.py
----------------------
Data cleaning, preprocessing, and feature engineering pipeline for the
HHS Unaccompanied Alien Children (UAC) Program time-series dataset.

This module is used by BOTH the Jupyter notebook and the Streamlit app,
so all cleaning / feature-engineering logic lives here in ONE place.

Author: Data Science Project - UAC Predictive Forecasting
"""

import pandas as pd
import numpy as np
import os

# ----------------------------------------------------------------------
# Column name constants (renamed to short, code-friendly names)
# ----------------------------------------------------------------------
RAW_COLUMNS_MAP = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "cbp_intake",
    "Children apprehended and placed in CBP custody": "cbp_intake",
    "Children in CBP custody": "cbp_active",
    "Children transferred out of CBP custody": "cbp_transferred_out",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "hhs_discharged",
}

TARGET_COL = "hhs_care"                # primary forecasting target
SECONDARY_TARGET_COL = "hhs_discharged"  # secondary target (discharge demand)

NUMERIC_COLS = [
    "cbp_intake", "cbp_active", "cbp_transferred_out",
    "hhs_care", "hhs_discharged"
]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as provided by HHS."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw HHS UAC dataset:
      - Drop fully empty trailing rows
      - Rename columns to code-friendly names
      - Convert Date to datetime
      - Strip thousands separators (commas) from numeric-as-string columns
      - Coerce all metric columns to numeric
      - Sort chronologically ascending
      - Drop exact duplicate dates (keep first)
    """
    df = df.copy()

    # Drop rows that are entirely empty (trailing blank rows common in HHS exports)
    df = df.dropna(how="all")

    # Rename columns
    rename_map = {c: RAW_COLUMNS_MAP[c] for c in df.columns if c in RAW_COLUMNS_MAP}
    df = df.rename(columns=rename_map)

    # Drop rows with no date at all
    df = df[df["date"].notna()]

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]

    # Clean numeric columns: remove commas, coerce to numeric
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("*", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort chronologically, drop duplicate dates
    df = df.sort_values("date").drop_duplicates(subset="date", keep="first")
    df = df.reset_index(drop=True)

    return df


def build_continuous_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    HHS reports are NOT published every calendar day (weekends / holidays are
    frequently skipped). To build a proper time series we reindex onto a full
    continuous daily calendar and interpolate the gaps. This preserves trend
    and avoids introducing artificial jumps.
    """
    df = df.set_index("date")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"

    # Linear interpolation for gaps (bounded to reasonable gap sizes),
    # then forward/backward fill any remaining edge NaNs.
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df = df.reset_index()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, quarter, weekend flag."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek       # 0=Mon
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def add_lag_features(df: pd.DataFrame, col: str, lags=(1, 7, 14)) -> pd.DataFrame:
    """Add lag features for a given column."""
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows=(7, 14)) -> pd.DataFrame:
    """Add rolling mean / std features for a given column (shifted by 1 to avoid leakage)."""
    df = df.copy()
    shifted = df[col].shift(1)  # avoid look-ahead leakage
    for w in windows:
        df[f"{col}_rollmean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
        df[f"{col}_rollstd_{w}"] = shifted.rolling(window=w, min_periods=1).std()
    return df


def add_flow_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Net pressure indicator = inflow (transfers into HHS) - outflow (discharges).
    Positive values => system load increasing; negative => load decreasing.
    """
    df = df.copy()
    df["net_pressure"] = df["cbp_transferred_out"] - df["hhs_discharged"]
    df["net_pressure_lag_1"] = df["net_pressure"].shift(1)
    df["net_pressure_roll7"] = df["net_pressure"].shift(1).rolling(7, min_periods=1).mean()
    df["care_change_1d"] = df[TARGET_COL].diff(1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline applied AFTER clean_data() and
    build_continuous_daily_index(). Produces a model-ready dataframe
    with lag, rolling, calendar, and flow features. Drops initial rows
    with NaNs introduced by lagging.
    """
    df = df.copy()
    df = add_calendar_features(df)
    df = add_flow_signals(df)

    for col in [TARGET_COL, SECONDARY_TARGET_COL, "cbp_intake", "cbp_active", "cbp_transferred_out"]:
        df = add_lag_features(df, col, lags=(1, 7, 14))
        df = add_rolling_features(df, col, windows=(7, 14))

    # Drop rows where lag/rolling features are still NaN (first ~14 rows)
    feature_cols = [c for c in df.columns if "_lag_" in c or "_roll" in c]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


def full_pipeline(raw_csv_path: str) -> pd.DataFrame:
    """Convenience wrapper: raw CSV -> cleaned, continuous, feature-engineered dataframe."""
    df = load_raw_data(raw_csv_path)
    df = clean_data(df)
    df = build_continuous_daily_index(df)
    df = engineer_features(df)
    return df


def get_model_feature_columns(df: pd.DataFrame, target: str = TARGET_COL):
    """
    Return the list of feature columns to feed into ML models.

    Excludes:
      - date, and both raw targets (hhs_care, hhs_discharged)
      - same-day (contemporaneous) exogenous raw metrics, since these are NOT
        known in advance at forecast time in a real deployment (only their
        lagged / rolling versions are legitimately available)
    This avoids data leakage in the multi-step recursive forecasting setup.
    """
    same_day_leaky_cols = {
        "cbp_intake", "cbp_active", "cbp_transferred_out",
        "net_pressure", "care_change_1d",
    }
    exclude = {"date", TARGET_COL, SECONDARY_TARGET_COL} | same_day_leaky_cols
    features = [c for c in df.columns if c not in exclude]
    return features


if __name__ == "__main__":
    RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data.csv")
    CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
    FEATURED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "featured_data.csv")

    raw = load_raw_data(RAW_PATH)
    cleaned = clean_data(raw)
    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data saved: {cleaned.shape} -> {CLEAN_PATH}")

    continuous = build_continuous_daily_index(cleaned)
    featured = engineer_features(continuous)
    featured.to_csv(FEATURED_PATH, index=False)
    print(f"Feature-engineered data saved: {featured.shape} -> {FEATURED_PATH}")
