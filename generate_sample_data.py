"""
Synthetic Sample Dataset Generator for AI Sales Forecasting
Generates realistic multi-year sales data with upward trends, day-of-week seasonality,
holiday peaks, promotional spikes, and product/regional variations.
Outputs both CSV and Excel formats in data/raw/
"""

import os
import numpy as np
import pandas as pd


def generate_sample_sales_dataset(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42
) -> pd.DataFrame:
    """Generates a realistic synthetic sales dataset for demonstration and testing."""
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    total_days = len(date_range)

    products = [
        {"name": "Laptop Pro 15", "base_price": 1200.0, "base_qty": 6, "q4_boost": 1.4, "trend_factor": 0.0003},
        {"name": "Smartphone Ultra", "base_price": 750.0, "base_qty": 14, "q4_boost": 1.35, "trend_factor": 0.0004},
        {"name": "Wireless Noise-Canceling Headphones", "base_price": 180.0, "base_qty": 25, "q4_boost": 1.5, "trend_factor": 0.0002},
        {"name": "Smart Fitness Watch", "base_price": 220.0, "base_qty": 18, "q4_boost": 1.3, "trend_factor": 0.00025},
        {"name": "Ergonomic Office Chair", "base_price": 320.0, "base_qty": 8, "q4_boost": 1.1, "trend_factor": 0.00015},
    ]

    regions = [
        {"name": "North", "weight": 1.15},
        {"name": "West", "weight": 1.10},
        {"name": "South", "weight": 0.95},
        {"name": "East", "weight": 0.80},
    ]

    records = []

    # Known calendar holidays (month, day)
    holiday_dates = {
        (1, 1),    # New Year
        (7, 4),    # Independence Day
        (11, 24),  # Black Friday / Thanksgiving window
        (11, 25),
        (11, 26),
        (11, 27),
        (12, 24),  # Christmas Eve
        (12, 25),  # Christmas Day
        (12, 26),  # Boxing Day
        (12, 31),  # New Year Eve
    }

    for day_idx, current_date in enumerate(date_range):
        month = current_date.month
        day = current_date.day
        dow = current_date.dayofweek  # 0=Monday, 6=Sunday

        is_holiday = int((month, day) in holiday_dates)

        # Scheduled promotional events (mid-month weekend flash sales or holiday weeks)
        is_promo = int(
            (day in [14, 15, 16] and dow in [4, 5, 6]) or
            (month == 11 and day >= 20) or
            (month == 12 and day >= 15 and day <= 24)
        )

        # Weekly seasonality multiplier (weekends have higher retail traffic)
        weekend_mult = 1.30 if dow in [5, 6] else (1.10 if dow == 4 else 0.95)

        # Holiday multiplier
        holiday_mult = 1.65 if is_holiday else 1.0

        # Promo multiplier
        promo_mult = 1.45 if is_promo else 1.0

        for prod in products:
            for reg in regions:
                # Overall trend (slight upward expansion over 2 years)
                trend_mult = 1.0 + (day_idx * prod["trend_factor"])

                # Seasonal Q4 boost
                q4_mult = prod["q4_boost"] if month in [10, 11, 12] else 1.0

                # Mean expected quantity
                expected_qty = (
                    prod["base_qty"]
                    * reg["weight"]
                    * weekend_mult
                    * holiday_mult
                    * promo_mult
                    * trend_mult
                    * q4_mult
                )

                # Add Poisson / normal noise
                qty = max(1, int(np.random.poisson(lam=max(1.0, expected_qty))))

                # Price adjustments (promotional discount 10-15%)
                unit_price = prod["base_price"]
                if is_promo:
                    unit_price = round(unit_price * np.random.uniform(0.85, 0.92), 2)
                else:
                    unit_price = round(unit_price * np.random.uniform(0.98, 1.02), 2)

                sales_amount = round(qty * unit_price, 2)

                records.append({
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Product": prod["name"],
                    "Region": reg["name"],
                    "Quantity": qty,
                    "Unit_Price": unit_price,
                    "Sales": sales_amount,
                    "Promotion": is_promo,
                    "Holiday": is_holiday,
                })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "sample_sales_data.csv")
    xlsx_path = os.path.join(output_dir, "sample_sales_data.xlsx")

    print("Generating synthetic retail sales dataset...")
    df = generate_sample_sales_dataset()
    print(f"Generated {len(df):,} records spanning {df['Date'].min()} to {df['Date'].max()}.")
    print(f"Total Sales: ${df['Sales'].sum():,.2f}")

    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"Saved Excel to: {xlsx_path}")
