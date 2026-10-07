"""
Database Module for AI Sales Forecasting
Manages SQLite storage for datasets, trained model evaluation metrics, and generated forecasts.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "sales_forecasting.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a SQLite database connection, ensuring the parent directory exists."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes required tables in the SQLite database if they do not exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            row_count INTEGER,
            col_count INTEGER,
            min_date TEXT,
            max_date TEXT,
            total_sales REAL,
            total_quantity REAL,
            num_products INTEGER,
            num_regions INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            target_level TEXT,
            train_start TEXT,
            train_end TEXT,
            test_start TEXT,
            test_end TEXT,
            mae REAL,
            rmse REAL,
            mape REAL,
            r2 REAL,
            trained_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecast_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            product TEXT,
            region TEXT,
            forecast_horizon_days INTEGER,
            forecast_date TEXT NOT NULL,
            forecast_value REAL NOT NULL,
            lower_bound REAL,
            upper_bound REAL,
            generated_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def log_dataset(
    filename: str,
    row_count: int,
    col_count: int,
    min_date: str,
    max_date: str,
    total_sales: float,
    total_quantity: float,
    num_products: int,
    num_regions: int,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Logs dataset summary information upon upload."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO datasets (
            filename, upload_time, row_count, col_count, min_date, max_date,
            total_sales, total_quantity, num_products, num_regions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename, now_str, row_count, col_count, min_date, max_date,
        total_sales, total_quantity, num_products, num_regions
    ))

    dataset_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return dataset_id


def log_model_metrics(
    metrics_list: List[Dict[str, Any]],
    dataset_name: str,
    target_level: str = "Overall",
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Logs model evaluation metrics to the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for m in metrics_list:
        cursor.execute("""
            INSERT INTO model_metrics (
                dataset_name, model_name, target_level, train_start, train_end,
                test_start, test_end, mae, rmse, mape, r2, trained_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_name,
            m.get("model_name", "Unknown"),
            target_level,
            str(m.get("train_start", "")),
            str(m.get("train_end", "")),
            str(m.get("test_start", "")),
            str(m.get("test_end", "")),
            float(m.get("mae", 0.0)) if m.get("mae") is not None else None,
            float(m.get("rmse", 0.0)) if m.get("rmse") is not None else None,
            float(m.get("mape", 0.0)) if m.get("mape") is not None else None,
            float(m.get("r2", 0.0)) if m.get("r2") is not None else None,
            now_str
        ))

    conn.commit()
    conn.close()


def log_forecast(
    forecast_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    product: str = "All",
    region: str = "All",
    horizon_days: int = 30,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Logs generated forecast records to the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    records = []
    for _, row in forecast_df.iterrows():
        f_date = str(row.get("date", row.get("Date", "")))
        f_val = float(row.get("forecast", row.get("Forecast", 0.0)))
        low = float(row.get("lower_bound", f_val * 0.9)) if "lower_bound" in row else None
        high = float(row.get("upper_bound", f_val * 1.1)) if "upper_bound" in row else None

        records.append((
            dataset_name, model_name, product, region, horizon_days,
            f_date, f_val, low, high, now_str
        ))

    cursor.executemany("""
        INSERT INTO forecast_results (
            dataset_name, model_name, product, region, forecast_horizon_days,
            forecast_date, forecast_value, lower_bound, upper_bound, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()


def get_dataset_history(limit: int = 10, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves recent dataset upload history as a DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    query = "SELECT * FROM datasets ORDER BY id DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def get_saved_metrics(dataset_name: Optional[str] = None, limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves saved model evaluation metrics."""
    init_db(db_path)
    conn = get_connection(db_path)
    if dataset_name:
        query = "SELECT * FROM model_metrics WHERE dataset_name = ? ORDER BY id DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(dataset_name, limit))
    else:
        query = "SELECT * FROM model_metrics ORDER BY id DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def get_saved_forecasts(limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves recent forecast results from the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    query = "SELECT * FROM forecast_results ORDER BY id DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df
