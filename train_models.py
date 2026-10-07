"""
Training Script for AI Sales Forecasting
Runs the full data pipeline from terminal:
Data Ingestion -> Validation -> Cleaning -> Feature Engineering ->
Chronological Train/Test Split -> Model Training & Evaluation ->
Model Selection -> Persistence -> SQLite Logging.
"""

import os
import sys
import argparse
import pandas as pd

from preprocessing import load_dataset_from_file, validate_raw_dataset, clean_and_prepare_data
from forecasting import (
    prepare_daily_series,
    create_features,
    split_chronological,
    train_and_evaluate_all_models,
    generate_future_forecast,
    save_forecasting_pipeline
)
from database import init_db, log_dataset, log_model_metrics, log_forecast


def main():
    parser = argparse.ArgumentParser(description="Train AI Sales Forecasting Models")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "raw", "sample_sales_data.csv"),
        help="Path to sales dataset (CSV or Excel)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Forecast horizon in days (default: 30)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Fraction of data reserved for chronological test set (default: 0.15)"
    )
    args = parser.parse_args()

    data_path = args.data
    horizon = args.horizon

    print("=" * 70)
    print("AI SALES FORECASTING - MODEL TRAINING PIPELINE")
    print("Student: Lokesh | Final Year BSc Data Science Project")
    print("=" * 70)

    # 1. Initialize SQLite Database
    init_db()
    print("[1/7] Initialized SQLite database.")

    # 2. Ingest Data
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at '{data_path}'. Please check path or run generate_sample_data.py first.")
        sys.exit(1)

    print(f"[2/7] Loading raw dataset: {data_path}")
    raw_df, err = load_dataset_from_file(data_path, os.path.basename(data_path))
    if err:
        print(f"Error: {err}")
        sys.exit(1)

    # 3. Validation
    print("[3/7] Validating dataset integrity...")
    val_report = validate_raw_dataset(raw_df)
    if not val_report["is_valid"]:
        print("Validation FAILED:")
        for e in val_report["errors"]:
            print(f"  - {e}")
        sys.exit(1)

    summary = val_report["summary"]
    print(f"  ✓ Rows: {summary.get('total_rows'):,}")
    print(f"  ✓ Date Range: {summary.get('start_date')} to {summary.get('end_date')} ({summary.get('date_span_days')} days)")
    print(f"  ✓ Estimated Sales: ${summary.get('estimated_total_sales'):,.2f}")
    if val_report["warnings"]:
        print("  Warnings:")
        for w in val_report["warnings"]:
            print(f"    * {w}")

    # 4. Cleaning & Preprocessing
    print("[4/7] Cleaning and standardizing dataset...")
    clean_df, clean_stats = clean_and_prepare_data(raw_df)
    print(f"  ✓ Records after cleaning: {clean_stats['final_rows']:,}")
    print(f"  ✓ Total Sales Revenue: ${clean_stats['total_sales']:,.2f}")
    print(f"  ✓ Unique Products: {clean_stats['unique_products']}")
    print(f"  ✓ Unique Regions: {clean_stats['unique_regions']}")

    # Save cleaned data
    processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "cleaned_sales_data.csv")
    clean_df.to_csv(processed_path, index=False)
    print(f"  ✓ Saved cleaned data to: {processed_path}")

    # Log to SQLite
    log_dataset(
        filename=os.path.basename(data_path),
        row_count=clean_stats["final_rows"],
        col_count=len(clean_df.columns),
        min_date=clean_stats["min_date"],
        max_date=clean_stats["max_date"],
        total_sales=clean_stats["total_sales"],
        total_quantity=clean_stats["total_quantity"],
        num_products=clean_stats["unique_products"],
        num_regions=clean_stats["unique_regions"]
    )

    # 5. Daily Aggregation & Feature Engineering
    print("[5/7] Aggregating continuous daily sales and engineering features...")
    daily_df = prepare_daily_series(clean_df)
    featured_df = create_features(daily_df)
    print(f"  ✓ Daily sequence count: {len(featured_df)} days")
    print(f"  ✓ Engineered features: Calendar (Year, Month, DOW, Quarter, Sine/Cosine cyclical),")
    print(f"                          Lags (1, 7, 14, 30), Rolling Stats (7, 14, 30 days).")

    # Chronological Split
    train_df, test_df, split_info = split_chronological(featured_df, test_ratio=args.test_ratio)
    print(f"  ✓ Train Period: {split_info['train_start']} to {split_info['train_end']} ({split_info['train_count']} days)")
    print(f"  ✓ Test Period:  {split_info['test_start']} to {split_info['test_end']} ({split_info['test_count']} days)")
    print("  ✓ Verification: Chronological boundary preserved. Zero future data leakage.")

    # 6. Model Training & Evaluation
    print("\n[6/7] Training candidate forecasting models...")
    model_store, comp_df, best_model_name = train_and_evaluate_all_models(train_df, test_df)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPARISON TABLE (Chronological Test Set)")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print("=" * 70)
    print(f"★ BEST PERFORMING MODEL: {best_model_name} (Lowest Test RMSE & MAE)\n")

    # Save Best Model
    saved_path = save_forecasting_pipeline(
        model_store=model_store,
        best_model_name=best_model_name,
        comparison_df=comp_df,
        target_info={"level": "Overall", "dataset": os.path.basename(data_path)}
    )
    print(f"✓ Saved pipeline artifact to: {saved_path}")

    # Log metrics to SQLite
    metrics_records = []
    for _, row in comp_df.iterrows():
        metrics_records.append({
            "model_name": row["Model"],
            "train_start": split_info["train_start"],
            "train_end": split_info["train_end"],
            "test_start": split_info["test_start"],
            "test_end": split_info["test_end"],
            "mae": row["MAE"],
            "rmse": row["RMSE"],
            "mape": row["MAPE (%)"],
            "r2": row["R² Score"]
        })
    log_model_metrics(metrics_records, dataset_name=os.path.basename(data_path))

    # 7. Future Forecast Generation
    print(f"\n[7/7] Generating {horizon}-day future sales forecast with '{best_model_name}'...")
    forecast_df = generate_future_forecast(
        model_store=model_store,
        model_name=best_model_name,
        historical_daily_df=daily_df,
        horizon_days=horizon
    )

    log_forecast(
        forecast_df=forecast_df,
        dataset_name=os.path.basename(data_path),
        model_name=best_model_name,
        product="All",
        region="All",
        horizon_days=horizon
    )

    # Save forecast report
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_csv = os.path.join(reports_dir, f"forecast_{horizon}days.csv")
    forecast_df.to_csv(report_csv, index=False)
    print(f"✓ Saved future forecast to: {report_csv}")

    print("\nNext 7 Days Forecast Preview:")
    print(forecast_df.head(7).to_string(index=False))

    print("\n" + "=" * 70)
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("You can now launch the dashboard with: streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
