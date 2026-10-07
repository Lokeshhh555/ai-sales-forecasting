"""
Unit & Integration Test Suite for AI Sales Forecasting
Tests data validation, cleaning, feature engineering, chronological splitting,
model training, evaluation metrics, forecasting, and database persistence.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from preprocessing import (
    detect_and_standardize_columns,
    validate_raw_dataset,
    clean_and_prepare_data,
    load_dataset_from_file
)
from forecasting import (
    prepare_daily_series,
    create_features,
    split_chronological,
    train_and_evaluate_all_models,
    generate_future_forecast,
    save_forecasting_pipeline,
    load_forecasting_pipeline,
    FEATURE_COLUMNS
)
from evaluation import calculate_metrics, compare_models, calculate_residuals
from database import (
    init_db,
    log_dataset,
    log_model_metrics,
    log_forecast,
    get_dataset_history,
    get_saved_metrics,
    get_saved_forecasts
)


# ==========================================
# 1. PREPROCESSING & VALIDATION TESTS
# ==========================================

def test_column_standardization():
    raw_df = pd.DataFrame({
        "Order Date": ["2023-01-01", "2023-01-02"],
        "Item Name": ["Product A", "Product B"],
        "Qty": [10, 20],
        "Price": [15.0, 25.0],
        "Total Revenue": [150.0, 500.0],
        "Store Name": ["North", "South"]
    })
    standardized, mapping = detect_and_standardize_columns(raw_df)

    assert "date" in standardized.columns
    assert "product" in standardized.columns
    assert "quantity" in standardized.columns
    assert "unit_price" in standardized.columns
    assert "sales" in standardized.columns
    assert "region" in standardized.columns


def test_validation_valid_data():
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=45, freq="D"),
        "Sales": np.random.uniform(100, 500, 45),
        "Product": ["Gadget"] * 45,
        "Region": ["North"] * 45
    })
    report = validate_raw_dataset(df)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["summary"]["total_rows"] == 45


def test_validation_missing_date():
    df = pd.DataFrame({
        "Sales": [100, 200, 300],
        "Product": ["A", "B", "C"]
    })
    report = validate_raw_dataset(df)
    assert report["is_valid"] is False
    assert any("Date" in err for err in report["errors"])


def test_validation_insufficient_data():
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=5, freq="D"),
        "Sales": [10, 20, 30, 40, 50]
    })
    report = validate_raw_dataset(df)
    assert report["is_valid"] is False
    assert any("Insufficient" in err for err in report["errors"])


def test_cleaning_and_deduplication():
    df = pd.DataFrame({
        "Date": ["2023-01-02", "2023-01-01", "2023-01-01", "invalid_date"],
        "Sales": [200.0, 100.0, 100.0, 50.0],
        "Quantity": [2, 1, 1, 1],
        "Product": ["A", "B", "B", "C"]
    })
    cleaned, stats = clean_and_prepare_data(df)

    # Invalid date dropped, duplicate dropped
    assert stats["invalid_dates_dropped"] == 1
    assert stats["duplicates_removed"] == 1
    assert len(cleaned) == 2
    # Chronological sort check
    assert cleaned.iloc[0]["date"] < cleaned.iloc[1]["date"]


# ==========================================
# 2. FEATURE ENGINEERING & SPLIT TESTS
# ==========================================

def test_feature_engineering_no_leakage():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    sales = [float(i * 10) for i in range(60)]
    df = pd.DataFrame({"date": dates, "sales": sales})

    featured = create_features(df)

    for col in FEATURE_COLUMNS:
        assert col in featured.columns

    # Verify strict lag_1 does not look ahead (lag_1 at row t must equal sales at row t-1)
    for idx in range(1, len(featured)):
        assert featured.loc[idx, "lag_1"] == df.loc[idx - 1, "sales"]

    # Verify rolling_mean_7 at row t uses sales up to t-1
    for idx in range(7, 15):
        expected_roll = np.mean(df.loc[idx-7:idx-1, "sales"])
        assert np.isclose(featured.loc[idx, "rolling_mean_7"], expected_roll)


def test_chronological_split():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "sales": np.ones(100) * 50.0})
    featured = create_features(df)

    train_df, test_df, split_info = split_chronological(featured, test_ratio=0.2)

    # Train period must precede test period chronologically
    assert train_df["date"].max() < test_df["date"].min()
    assert len(train_df) + len(test_df) == len(featured)
    assert split_info["train_start"] == "2023-01-01"


# ==========================================
# 3. EVALUATION METRICS TESTS
# ==========================================

def test_metrics_calculation_standard():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])

    m = calculate_metrics(y_true, y_pred)
    assert m["mae"] == 10.0
    assert m["rmse"] == 10.0
    assert m["has_zeros"] is False
    assert 6.0 < m["mape"] < 6.2


def test_metrics_safe_mape_with_zeros():
    # Actual has a zero; standard division by zero should be avoided
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 190.0])

    m = calculate_metrics(y_true, y_pred)
    assert m["has_zeros"] is True
    assert not np.isnan(m["mape"])
    assert not np.isinf(m["mape"])
    assert m["wape"] > 0


# ==========================================
# 4. MODEL TRAINING & FORECASTING TESTS
# ==========================================

def test_model_training_and_comparison():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    sales = 100 + 2 * np.arange(120) + np.random.normal(0, 5, 120)
    df = pd.DataFrame({"date": dates, "sales": sales})

    featured = create_features(df)
    train_df, test_df, _ = split_chronological(featured, test_ratio=0.2)

    model_store, comp_df, best_model = train_and_evaluate_all_models(train_df, test_df)

    assert len(model_store) == 5
    assert "Linear Regression" in model_store
    assert "Random Forest" in model_store
    assert "Gradient Boosting" in model_store
    assert "Baseline (Moving Avg)" in model_store
    assert "ARIMA" in model_store

    assert len(comp_df) == 5
    assert best_model in comp_df["Model"].values


def test_future_forecast_generation():
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    sales = 200 + 0.5 * np.arange(90) + np.random.normal(0, 10, 90)
    df = pd.DataFrame({"date": dates, "sales": sales})

    featured = create_features(df)
    train_df, test_df, _ = split_chronological(featured, test_ratio=0.2)
    model_store, _, best_model = train_and_evaluate_all_models(train_df, test_df)

    for horizon in [7, 14, 30]:
        forecast_df = generate_future_forecast(model_store, best_model, df, horizon_days=horizon)
        assert len(forecast_df) == horizon
        assert "date" in forecast_df.columns
        assert "forecast" in forecast_df.columns
        assert "lower_bound" in forecast_df.columns
        assert "upper_bound" in forecast_df.columns
        # Lower bound should not exceed upper bound
        assert (forecast_df["lower_bound"] <= forecast_df["upper_bound"]).all()


# ==========================================
# 5. DATABASE TESTS
# ==========================================

def test_sqlite_database_lifecycle():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_path = tmp.name

        init_db(db_path)

        # Log dataset
        d_id = log_dataset(
            filename="test.csv",
            row_count=100,
            col_count=5,
            min_date="2023-01-01",
            max_date="2023-04-10",
            total_sales=50000.0,
            total_quantity=500,
            num_products=3,
            num_regions=2,
            db_path=db_path
        )
        assert d_id > 0

        # Retrieve dataset history
        d_history = get_dataset_history(limit=5, db_path=db_path)
        assert len(d_history) == 1
        assert d_history.iloc[0]["filename"] == "test.csv"

        # Log model metrics
        metrics = [{
            "model_name": "Random Forest",
            "train_start": "2023-01-01",
            "train_end": "2023-03-31",
            "test_start": "2023-04-01",
            "test_end": "2023-04-10",
            "mae": 15.2,
            "rmse": 20.4,
            "mape": 4.5,
            "r2": 0.88
        }]
        log_model_metrics(metrics, dataset_name="test.csv", db_path=db_path)
        saved_metrics = get_saved_metrics(dataset_name="test.csv", db_path=db_path)
        assert len(saved_metrics) == 1
        assert saved_metrics.iloc[0]["model_name"] == "Random Forest"

        # Log forecast
        f_df = pd.DataFrame({
            "date": ["2023-04-11", "2023-04-12"],
            "forecast": [120.0, 125.0],
            "lower_bound": [100.0, 105.0],
            "upper_bound": [140.0, 145.0]
        })
        log_forecast(f_df, dataset_name="test.csv", model_name="Random Forest", horizon_days=2, db_path=db_path)
        saved_f = get_saved_forecasts(db_path=db_path)
        assert len(saved_f) == 2
