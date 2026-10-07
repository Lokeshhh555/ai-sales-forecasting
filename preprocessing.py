"""
Preprocessing and Validation Module for AI Sales Forecasting
Handles column auto-detection, schema validation, data cleaning, and chronological ordering.
"""

import io
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np


# Canonical column mappings (lower-cased synonyms -> canonical name)
SYNONYM_MAP = {
    # Date synonyms
    "date": "date",
    "order date": "date",
    "order_date": "date",
    "sale date": "date",
    "sales date": "date",
    "sales_date": "date",
    "transaction_date": "date",
    "invoice_date": "date",
    "invoicedate": "date",
    "period": "date",
    "timestamp": "date",
    "day": "date",

    # Sales / Revenue synonyms
    "sales": "sales",
    "sale": "sales",
    "revenue": "sales",
    "total sales": "sales",
    "total_sales": "sales",
    "total revenue": "sales",
    "total_revenue": "sales",
    "amount": "sales",
    "total amount": "sales",
    "total_amount": "sales",
    "turnover": "sales",
    "net sales": "sales",
    "net_sales": "sales",

    # Product synonyms
    "product": "product",
    "product name": "product",
    "product_name": "product",
    "item": "product",
    "item name": "product",
    "item_name": "product",
    "sku": "product",
    "category": "product",
    "product category": "product",
    "product_category": "product",

    # Quantity synonyms
    "quantity": "quantity",
    "qty": "quantity",
    "units": "quantity",
    "units sold": "quantity",
    "units_sold": "quantity",
    "volume": "quantity",
    "order_qty": "quantity",

    # Unit Price synonyms
    "unit price": "unit_price",
    "unit_price": "unit_price",
    "price": "unit_price",
    "price per unit": "unit_price",
    "rate": "unit_price",
    "cost per unit": "unit_price",

    # Region synonyms
    "region": "region",
    "store": "region",
    "store name": "region",
    "store_name": "region",
    "location": "region",
    "branch": "region",
    "territory": "region",
    "zone": "region",
    "city": "region",
    "state": "region",

    # Promotion synonyms
    "promotion": "promotion",
    "promo": "promotion",
    "is_promotion": "promotion",
    "on_promo": "promotion",
    "discount_applied": "promotion",

    # Holiday synonyms
    "holiday": "holiday",
    "is_holiday": "holiday",
    "holiday_flag": "holiday",
}


def detect_and_standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Detects equivalent column names and maps them to canonical schema.
    Returns the mapped DataFrame copy and the mapping dictionary used.
    """
    df = df.copy()
    raw_cols = df.columns.tolist()
    column_mapping = {}

    for col in raw_cols:
        clean_name = str(col).strip().lower().replace("-", "_").replace("  ", " ")
        if clean_name in SYNONYM_MAP:
            canonical = SYNONYM_MAP[clean_name]
            column_mapping[col] = canonical
        else:
            # Fallback fuzzy substring matching for essential attributes
            if any(k in clean_name for k in ["date", "time"]) and "date" not in column_mapping.values():
                column_mapping[col] = "date"
            elif any(k in clean_name for k in ["sales", "revenue", "amount"]) and "sales" not in column_mapping.values():
                column_mapping[col] = "sales"
            elif any(k in clean_name for k in ["item", "product"]) and "product" not in column_mapping.values():
                column_mapping[col] = "product"
            elif any(k in clean_name for k in ["qty", "quant"]) and "quantity" not in column_mapping.values():
                column_mapping[col] = "quantity"
            elif any(k in clean_name for k in ["price", "rate"]) and "unit_price" not in column_mapping.values():
                column_mapping[col] = "unit_price"
            elif any(k in clean_name for k in ["region", "store", "zone", "branch"]) and "region" not in column_mapping.values():
                column_mapping[col] = "region"
            elif "promo" in clean_name and "promotion" not in column_mapping.values():
                column_mapping[col] = "promotion"
            elif "holiday" in clean_name and "holiday" not in column_mapping.values():
                column_mapping[col] = "holiday"
            else:
                column_mapping[col] = col

    df = df.rename(columns=column_mapping)
    return df, column_mapping


def validate_raw_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates dataset before and after standardization.
    Returns a comprehensive validation report dictionary with errors, warnings, and summary stats.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        return {
            "is_valid": False,
            "errors": ["The uploaded dataset is empty."],
            "warnings": warnings,
            "summary": {}
        }

    standardized_df, mapping = detect_and_standardize_columns(df)
    cols = standardized_df.columns.tolist()

    # Check required columns
    has_date = "date" in cols
    has_sales = "sales" in cols
    has_qty = "quantity" in cols
    has_price = "unit_price" in cols

    if not has_date:
        errors.append("Missing required 'Date' column (or date synonym like 'Order Date', 'Transaction Date').")

    if not has_sales:
        if has_qty and has_price:
            warnings.append("Sales column missing, but found 'Quantity' and 'Unit Price'. Sales can be calculated automatically as (Quantity * Unit Price).")
        else:
            errors.append("Missing required 'Sales' / 'Revenue' column (or both 'Quantity' and 'Unit Price' to compute it).")

    # If critical errors exist, exit early
    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "raw_rows": len(df),
                "raw_columns": len(df.columns),
                "detected_columns": list(mapping.values())
            }
        }

    # Inspect date column
    date_series = pd.to_datetime(standardized_df["date"], errors="coerce")
    null_dates = date_series.isna().sum()
    if null_dates > 0:
        pct_null_dates = (null_dates / len(df)) * 100
        if pct_null_dates > 30:
            errors.append(f"Critical: {null_dates} rows ({pct_null_dates:.1f}%) contain invalid or unparseable dates.")
        else:
            warnings.append(f"Notice: {null_dates} rows contain unparseable dates and will be removed during cleaning.")

    valid_dates = date_series.dropna()
    if len(valid_dates) < 14:
        errors.append(f"Insufficient historical observations: dataset has only {len(valid_dates)} valid dates (at least 14 daily records required for time series forecasting).")

    # Inspect sales column
    if has_sales:
        numeric_sales = pd.to_numeric(standardized_df["sales"], errors="coerce")
        null_sales = numeric_sales.isna().sum()
        if null_sales > 0:
            warnings.append(f"{null_sales} missing or non-numeric values found in 'Sales' column.")

        neg_sales = (numeric_sales < 0).sum()
        if neg_sales > 0:
            warnings.append(f"{neg_sales} negative sales values detected (e.g. returns/adjustments). These will be handled during cleaning.")

    # Inspect duplicates
    dup_rows = standardized_df.duplicated().sum()
    if dup_rows > 0:
        warnings.append(f"{dup_rows} duplicate rows detected in dataset. They will be deduplicated.")

    # Summary statistics
    clean_numeric_sales = pd.to_numeric(standardized_df["sales"], errors="coerce") if has_sales else pd.Series(dtype=float)
    total_sales_est = float(clean_numeric_sales[clean_numeric_sales > 0].sum()) if not clean_numeric_sales.empty else 0.0

    products = standardized_df["product"].nunique() if "product" in standardized_df else 1
    regions = standardized_df["region"].nunique() if "region" in standardized_df else 1

    min_date_str = str(valid_dates.min().strftime("%Y-%m-%d")) if not valid_dates.empty else "N/A"
    max_date_str = str(valid_dates.max().strftime("%Y-%m-%d")) if not valid_dates.empty else "N/A"

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns_list": cols,
            "missing_values_total": int(df.isna().sum().sum()),
            "duplicate_rows": int(dup_rows),
            "start_date": min_date_str,
            "end_date": max_date_str,
            "date_span_days": int((valid_dates.max() - valid_dates.min()).days) if not valid_dates.empty else 0,
            "estimated_total_sales": round(total_sales_est, 2),
            "unique_products": int(products),
            "unique_regions": int(regions),
        }
    }


def clean_and_prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw dataframe into standardized, model-ready format:
    1. Standardizes columns
    2. Parses dates & removes invalid dates
    3. Handles duplicates
    4. Computes/fixes sales, quantity, unit_price
    5. Fills missing optional categories (product, region)
    6. Sorts chronologically
    Returns (cleaned_df, cleaning_summary).
    """
    initial_rows = len(df)
    df_clean, mapping = detect_and_standardize_columns(df)

    # 1. Parse dates & drop rows with unparseable date
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    invalid_dates_dropped = int(df_clean["date"].isna().sum())
    df_clean = df_clean.dropna(subset=["date"])

    # 2. Deduplicate
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = before_dedup - len(df_clean)

    # 3. Numeric conversions
    if "quantity" in df_clean.columns:
        df_clean["quantity"] = pd.to_numeric(df_clean["quantity"], errors="coerce")
    else:
        df_clean["quantity"] = 1.0

    if "unit_price" in df_clean.columns:
        df_clean["unit_price"] = pd.to_numeric(df_clean["unit_price"], errors="coerce")

    if "sales" in df_clean.columns:
        df_clean["sales"] = pd.to_numeric(df_clean["sales"], errors="coerce")
    else:
        if "quantity" in df_clean.columns and "unit_price" in df_clean.columns:
            df_clean["sales"] = df_clean["quantity"] * df_clean["unit_price"]
        else:
            df_clean["sales"] = 0.0

    # If sales is null but qty and unit_price exist, compute sales
    mask_calc_sales = df_clean["sales"].isna() & df_clean["quantity"].notna() & df_clean.get("unit_price", pd.Series(dtype=float)).notna()
    if mask_calc_sales.any():
        df_clean.loc[mask_calc_sales, "sales"] = df_clean.loc[mask_calc_sales, "quantity"] * df_clean.loc[mask_calc_sales, "unit_price"]

    # Fill remaining missing sales with median or drop
    missing_sales_filled = int(df_clean["sales"].isna().sum())
    if missing_sales_filled > 0:
        median_sales = df_clean["sales"].median()
        if pd.isna(median_sales) or median_sales <= 0:
            median_sales = 100.0
        df_clean["sales"] = df_clean["sales"].fillna(median_sales)

    # Handle negative sales (clip to 0 or replace with 0)
    negative_sales_handled = int((df_clean["sales"] < 0).sum())
    df_clean["sales"] = df_clean["sales"].clip(lower=0.0)

    # Fill quantity if null
    df_clean["quantity"] = df_clean["quantity"].fillna(1.0).clip(lower=0.0)

    # Ensure product & region exist
    if "product" not in df_clean.columns:
        df_clean["product"] = "All Products"
    else:
        df_clean["product"] = df_clean["product"].fillna("All Products").astype(str).str.strip()

    if "region" not in df_clean.columns:
        df_clean["region"] = "All Regions"
    else:
        df_clean["region"] = df_clean["region"].fillna("All Regions").astype(str).str.strip()

    # Promotion & Holiday flags
    if "promotion" not in df_clean.columns:
        df_clean["promotion"] = 0
    else:
        df_clean["promotion"] = pd.to_numeric(df_clean["promotion"], errors="coerce").fillna(0).astype(int)

    if "holiday" not in df_clean.columns:
        df_clean["holiday"] = 0
    else:
        df_clean["holiday"] = pd.to_numeric(df_clean["holiday"], errors="coerce").fillna(0).astype(int)

    # 4. Sort chronologically
    df_clean = df_clean.sort_values(by=["date"]).reset_index(drop=True)

    final_rows = len(df_clean)

    cleaning_summary = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "invalid_dates_dropped": invalid_dates_dropped,
        "duplicates_removed": duplicates_removed,
        "missing_sales_filled": missing_sales_filled,
        "negative_sales_clipped": negative_sales_handled,
        "min_date": df_clean["date"].min().strftime("%Y-%m-%d") if not df_clean.empty else "N/A",
        "max_date": df_clean["date"].max().strftime("%Y-%m-%d") if not df_clean.empty else "N/A",
        "total_sales": round(float(df_clean["sales"].sum()), 2),
        "total_quantity": round(float(df_clean["quantity"].sum()), 2),
        "unique_products": int(df_clean["product"].nunique()),
        "unique_regions": int(df_clean["region"].nunique()),
    }

    return df_clean, cleaning_summary


def load_dataset_from_file(file_obj, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely reads CSV or Excel files from an uploaded file object or file path.
    Returns (DataFrame, error_message).
    """
    try:
        lower_name = filename.lower()
        if lower_name.endswith(".csv"):
            if hasattr(file_obj, "read"):
                # Handle bytes / file buffer
                df = pd.read_csv(file_obj)
            else:
                df = pd.read_csv(str(file_obj))
        elif lower_name.endswith((".xlsx", ".xls")):
            if hasattr(file_obj, "read"):
                df = pd.read_excel(file_obj)
            else:
                df = pd.read_excel(str(file_obj))
        else:
            return None, f"Unsupported file format '{filename}'. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file."
        return df, None
    except Exception as e:
        return None, f"Error reading file '{filename}': {str(e)}"
