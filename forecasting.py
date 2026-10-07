"""
Forecasting Engine Module for AI Sales Forecasting
Implements feature engineering, chronological train/test split, multiple forecasting models:
1. Baseline (7-day Moving Average)
2. Linear Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. ARIMA Time Series Model
Provides recursive multi-step future forecasting with prediction intervals, model saving, and loading.
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA

from evaluation import calculate_metrics

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved_models")


# ==========================================
# 1. TIME SERIES PREPARATION & AGGREGATION
# ==========================================

def prepare_daily_series(
    df: pd.DataFrame,
    product: str = "All",
    region: str = "All"
) -> pd.DataFrame:
    """
    Filters data by product and region, aggregates to daily sales,
    and ensures a continuous calendar timeline without date gaps.
    """
    df_filtered = df.copy()
    if product != "All" and "product" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["product"] == product]
    if region != "All" and "region" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["region"] == region]

    if df_filtered.empty:
        return pd.DataFrame(columns=["date", "sales"])

    # Aggregate daily
    daily = df_filtered.groupby("date")["sales"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Reindex to full continuous date range
    min_date = daily["date"].min()
    max_date = daily["date"].max()
    full_idx = pd.date_range(start=min_date, end=max_date, freq="D")

    daily = daily.set_index("date").reindex(full_idx, fill_value=0.0).rename_axis("date").reset_index()
    daily["sales"] = daily["sales"].astype(float)
    return daily


# ==========================================
# 2. FEATURE ENGINEERING (NO DATA LEAKAGE)
# ==========================================

def create_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates time-calendar features, strict lag features, and rolling features.
    Notice: All rolling features are computed on shift(1) to avoid looking ahead at the current day's target!
    """
    df = series_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Calendar / Time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # Cyclical representations
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Strict Lag Features (past values only)
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)
    df["lag_30"] = df["sales"].shift(30)

    # Rolling window statistics (computed on sales shifted by 1 to prevent data leakage)
    shifted_sales = df["sales"].shift(1)
    df["rolling_mean_7"] = shifted_sales.rolling(window=7, min_periods=1).mean()
    df["rolling_mean_14"] = shifted_sales.rolling(window=14, min_periods=1).mean()
    df["rolling_mean_30"] = shifted_sales.rolling(window=30, min_periods=1).mean()
    df["rolling_std_7"] = shifted_sales.rolling(window=7, min_periods=1).std().fillna(0)
    df["rolling_median_7"] = shifted_sales.rolling(window=7, min_periods=1).median()

    # Backfill earliest rows for lag_30 with expanding mean to keep rows usable
    for lag_col in ["lag_1", "lag_7", "lag_14", "lag_30"]:
        df[lag_col] = df[lag_col].bfill().fillna(df["sales"].mean())

    df["rolling_mean_7"] = df["rolling_mean_7"].bfill().fillna(df["sales"].mean())
    df["rolling_mean_14"] = df["rolling_mean_14"].bfill().fillna(df["sales"].mean())
    df["rolling_mean_30"] = df["rolling_mean_30"].bfill().fillna(df["sales"].mean())
    df["rolling_std_7"] = df["rolling_std_7"].bfill().fillna(0)
    df["rolling_median_7"] = df["rolling_median_7"].bfill().fillna(df["sales"].mean())

    return df


FEATURE_COLUMNS = [
    "year", "month", "day", "dayofweek", "quarter", "is_weekend",
    "sin_month", "cos_month", "sin_dayofweek", "cos_dayofweek",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7", "rolling_median_7"
]


# ==========================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================

def split_chronological(
    featured_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_test_days: int = 14,
    max_test_days: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Splits dataset chronologically into older train set and recent test set.
    Strictly forbids random shuffling to preserve temporal integrity.
    """
    n = len(featured_df)
    test_size = int(n * test_ratio)
    test_size = max(min_test_days, min(test_size, max_test_days))

    # Guard if dataset is small
    if test_size >= n - 7:
        test_size = max(1, int(n * 0.15))

    train_df = featured_df.iloc[:-test_size].copy().reset_index(drop=True)
    test_df = featured_df.iloc[-test_size:].copy().reset_index(drop=True)

    split_info = {
        "train_start": str(train_df["date"].min().strftime("%Y-%m-%d")),
        "train_end": str(train_df["date"].max().strftime("%Y-%m-%d")),
        "train_count": str(len(train_df)),
        "test_start": str(test_df["date"].min().strftime("%Y-%m-%d")),
        "test_end": str(test_df["date"].max().strftime("%Y-%m-%d")),
        "test_count": str(len(test_df)),
    }

    return train_df, test_df, split_info


# ==========================================
# 4. MODEL TRAINING & COMPARISON
# ==========================================

def train_and_evaluate_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """
    Trains all candidate models chronologically:
    1. Baseline (7-Day Rolling Moving Average)
    2. Linear Regression
    3. Random Forest Regressor
    4. Gradient Boosting Regressor
    5. ARIMA Time Series Model

    Returns:
    - model_store: dictionary containing trained models and predictions
    - comparison_df: sorted comparative metrics table
    - best_model_name: name of best performing model
    """
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["sales"].values

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["sales"].values

    model_store: Dict[str, Any] = {}
    model_results: Dict[str, Dict[str, Any]] = {}

    # 1. Baseline Model (7-Day Moving Average / Naive)
    baseline_val = float(train_df["sales"].tail(7).mean())
    y_pred_baseline = np.full(len(y_test), baseline_val)
    model_store["Baseline (Moving Avg)"] = {"model": "baseline", "value": baseline_val}
    model_results["Baseline (Moving Avg)"] = {"y_true": y_test, "y_pred": y_pred_baseline}

    # 2. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = np.clip(lr.predict(X_test), a_min=0, a_max=None)
    model_store["Linear Regression"] = {"model": lr}
    model_results["Linear Regression"] = {"y_true": y_test, "y_pred": y_pred_lr}

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = np.clip(rf.predict(X_test), a_min=0, a_max=None)
    model_store["Random Forest"] = {"model": rf}
    model_results["Random Forest"] = {"y_true": y_test, "y_pred": y_pred_rf}

    # 4. Gradient Boosting Regressor
    gb = HistGradientBoostingRegressor(max_iter=100, max_depth=8, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = np.clip(gb.predict(X_test), a_min=0, a_max=None)
    model_store["Gradient Boosting"] = {"model": gb}
    model_results["Gradient Boosting"] = {"y_true": y_test, "y_pred": y_pred_gb}

    # 5. ARIMA Time Series Model
    try:
        arima_model = ARIMA(y_train, order=(1, 1, 1))
        arima_fitted = arima_model.fit()
        arima_forecast = arima_fitted.forecast(steps=len(y_test))
        y_pred_arima = np.clip(np.asarray(arima_forecast), a_min=0, a_max=None)
        model_store["ARIMA"] = {"model": arima_fitted}
        model_results["ARIMA"] = {"y_true": y_test, "y_pred": y_pred_arima}
    except Exception as e:
        # Fallback if ARIMA fails convergence on small or uniform data
        y_pred_arima = np.full(len(y_test), baseline_val)
        model_store["ARIMA"] = {"model": "fallback", "error": str(e)}
        model_results["ARIMA"] = {"y_true": y_test, "y_pred": y_pred_arima}

    # Model Comparison Table
    records = []
    for name, res in model_results.items():
        metrics = calculate_metrics(res["y_true"], res["y_pred"])
        records.append({
            "Model": name,
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "MAPE (%)": metrics["mape"],
            "WAPE (%)": metrics["wape"],
            "R² Score": metrics["r2"],
            "MAPE Note": metrics["mape_note"]
        })
        # Save predictions in model_store for visualization
        model_store[name]["predictions"] = res["y_pred"]
        model_store[name]["metrics"] = metrics

    comparison_df = pd.DataFrame(records).sort_values(by=["RMSE", "MAE"]).reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["Model"]

    return model_store, comparison_df, best_model_name


# ==========================================
# 5. FUTURE FORECAST GENERATION
# ==========================================

def generate_future_forecast(
    model_store: Dict[str, Any],
    model_name: str,
    historical_daily_df: pd.DataFrame,
    horizon_days: int = 30
) -> pd.DataFrame:
    """
    Generates out-of-sample future forecasts for the specified horizon (e.g. 7, 14, 30, 60, 90 days).
    Uses recursive multi-step forecasting for ML models, projecting lag and rolling features step by step.
    Provides 95% confidence intervals (lower and upper bounds).
    """
    last_date = pd.to_datetime(historical_daily_df["date"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    # Estimate residual standard deviation for prediction intervals
    best_info = model_store.get(model_name, {})
    metrics = best_info.get("metrics", {})
    rmse = metrics.get("rmse", 50.0)
    if rmse <= 0:
        rmse = 50.0

    if model_name == "Baseline (Moving Avg)":
        base_val = best_info.get("value", historical_daily_df["sales"].tail(7).mean())
        forecast_vals = np.full(horizon_days, base_val)
        lower_vals = np.clip(forecast_vals - 1.96 * rmse, 0, None)
        upper_vals = forecast_vals + 1.96 * rmse

    elif model_name == "ARIMA" and hasattr(best_info.get("model"), "get_forecast"):
        try:
            # Re-fit ARIMA on the entire historical dataset for optimal future prediction
            full_series = historical_daily_df["sales"].values
            full_arima = ARIMA(full_series, order=(1, 1, 1)).fit()
            arima_res = full_arima.get_forecast(steps=horizon_days)
            forecast_vals = np.clip(arima_res.predicted_mean, 0, None)
            conf_int = arima_res.conf_int(alpha=0.05)
            lower_vals = np.clip(conf_int[:, 0], 0, None)
            upper_vals = np.clip(conf_int[:, 1], 0, None)
        except Exception:
            # Fallback recursive if ARIMA re-fit fails
            forecast_vals = np.full(horizon_days, float(historical_daily_df["sales"].tail(7).mean()))
            lower_vals = np.clip(forecast_vals - 1.96 * rmse, 0, None)
            upper_vals = forecast_vals + 1.96 * rmse

    else:
        # ML Recursive multi-step prediction (Linear Regression, Random Forest, Gradient Boosting)
        model = best_info.get("model")
        # Start with historical sales sequence
        sales_history = list(historical_daily_df["sales"].values)
        date_history = list(pd.to_datetime(historical_daily_df["date"]).values)

        forecast_vals = []
        lower_vals = []
        upper_vals = []

        for step, f_date in enumerate(future_dates):
            # Construct feature row dynamically
            f_year = f_date.year
            f_month = f_date.month
            f_day = f_date.day
            f_dow = f_date.dayofweek
            f_quarter = f_date.quarter
            f_is_weekend = int(f_dow >= 5)

            sin_m = np.sin(2 * np.pi * f_month / 12)
            cos_m = np.cos(2 * np.pi * f_month / 12)
            sin_d = np.sin(2 * np.pi * f_dow / 7)
            cos_d = np.cos(2 * np.pi * f_dow / 7)

            # Lags from history (which includes previously generated forecasts)
            lag_1 = sales_history[-1] if len(sales_history) >= 1 else 0
            lag_7 = sales_history[-7] if len(sales_history) >= 7 else sales_history[-1]
            lag_14 = sales_history[-14] if len(sales_history) >= 14 else sales_history[-1]
            lag_30 = sales_history[-30] if len(sales_history) >= 30 else sales_history[-1]

            # Rolling stats from history
            roll_7 = np.mean(sales_history[-7:])
            roll_14 = np.mean(sales_history[-14:])
            roll_30 = np.mean(sales_history[-30:])
            roll_std_7 = np.std(sales_history[-7:])
            roll_med_7 = np.median(sales_history[-7:])

            feat_vector = pd.DataFrame([{
                "year": f_year, "month": f_month, "day": f_day,
                "dayofweek": f_dow, "quarter": f_quarter, "is_weekend": f_is_weekend,
                "sin_month": sin_m, "cos_month": cos_m,
                "sin_dayofweek": sin_d, "cos_dayofweek": cos_d,
                "lag_1": lag_1, "lag_7": lag_7, "lag_14": lag_14, "lag_30": lag_30,
                "rolling_mean_7": roll_7, "rolling_mean_14": roll_14,
                "rolling_mean_30": roll_30, "rolling_std_7": roll_std_7,
                "rolling_median_7": roll_med_7
            }])[FEATURE_COLUMNS]

            pred_val = float(model.predict(feat_vector)[0])
            pred_val = max(0.0, pred_val)

            # Increasing forecast uncertainty as horizon expands
            uncertainty_multiplier = 1.0 + (step * 0.015)
            band = 1.96 * rmse * uncertainty_multiplier
            low = max(0.0, pred_val - band)
            high = pred_val + band

            forecast_vals.append(pred_val)
            lower_vals.append(low)
            upper_vals.append(high)

            # Append to history for subsequent steps
            sales_history.append(pred_val)
            date_history.append(f_date)

    forecast_df = pd.DataFrame({
        "date": future_dates.strftime("%Y-%m-%d"),
        "forecast": np.round(forecast_vals, 2),
        "lower_bound": np.round(lower_vals, 2),
        "upper_bound": np.round(upper_vals, 2),
        "model_used": model_name
    })

    return forecast_df


# ==========================================
# 6. MODEL PERSISTENCE
# ==========================================

def save_forecasting_pipeline(
    model_store: Dict[str, Any],
    best_model_name: str,
    comparison_df: pd.DataFrame,
    target_info: Dict[str, str],
    save_dir: str = SAVED_MODELS_DIR
) -> str:
    """Saves the best model and pipeline metadata using joblib."""
    os.makedirs(save_dir, exist_ok=True)
    best_item = model_store[best_model_name]

    payload = {
        "best_model_name": best_model_name,
        "model": best_item.get("model"),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": best_item.get("metrics"),
        "comparison_table": comparison_df.to_dict(orient="records"),
        "target_info": target_info
    }

    filepath = os.path.join(save_dir, "best_forecasting_model.pkl")
    joblib.dump(payload, filepath)
    return filepath


def load_forecasting_pipeline(save_dir: str = SAVED_MODELS_DIR) -> Optional[Dict[str, Any]]:
    """Loads saved model pipeline from disk if available."""
    filepath = os.path.join(save_dir, "best_forecasting_model.pkl")
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None
