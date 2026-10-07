# AI Sales Forecasting

**Intelligent Sales Prediction & Business Planning Dashboard**  
**Final Year BSc Data Science Capstone Project**  
**Student:** Lokesh  

---

## 1. Introduction
Accurate sales forecasting is a cornerstone of modern business intelligence, supply chain management, and financial planning. Retail and enterprise organizations often struggle with traditional static forecasting techniques such as simple moving averages or intuitive guesswork. These manual approaches frequently lead to either inventory stockouts (causing lost sales opportunities and customer dissatisfaction) or excess inventory accumulation (tying up working capital and increasing holding and spoilage costs).

This project presents **AI Sales Forecasting**, an end-to-end machine learning web application developed in Python and Streamlit. The system ingests multi-product, multi-regional historical sales records, performs automated schema detection and sanitization, generates leak-free temporal features, trains multiple competitive forecasting algorithms chronologically, evaluates them using standard regression metrics (MAE, RMSE, safe MAPE, WAPE, $R^2$), generates out-of-sample future projections with confidence intervals, and provides interactive visualization and reporting capabilities.

---

## 2. Problem Statement
Traditional retail sales forecasting suffers from significant challenges:
1. **Manual Inefficiencies:** Spreadsheet-based forecasting is labor-intensive, error-prone, and unsustainable for multi-SKU retail operations.
2. **Ignoring Complex Non-Linear Patterns:** Linear extrapolations fail to capture weekly seasonality, promotion uplift, holiday spikes, and non-linear macroeconomic trends.
3. **Data Leakage in Naive ML Implementations:** Standard machine learning workflows frequently use random train/test splits, which inadvertently shuffle future information into historical training sets, yielding artificially high accuracy metrics that fail in real-world deployment.
4. **Lack of Decision-Ready Interfaces:** Business stakeholders need actionable, interactive dashboards with confidence bands to inform inventory safety buffers, procurement lead times, and revenue targets.

---

## 3. Objectives
The core objectives of the system are:
- Build an automated data ingestion and validation pipeline supporting CSV and Excel formats.
- Preprocess and standardize raw data with automated column detection, deduplication, and handling of missing or negative records.
- Conduct comprehensive Exploratory Data Analysis (EDA) on sales trajectory, seasonal patterns, and regional/product distributions using Plotly.
- Implement strict chronological train/test splitting to prevent future data leakage.
- Engineer domain-relevant features: calendar indicators, cyclical trigonometric encodings, historical lags ($t-1, t-7, t-14, t-30$), and rolling statistics ($7, 14, 30$ days) computed strictly on past values.
- Train and evaluate 5 candidate models: Naive Moving Average Baseline, Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, and ARIMA.
- Provide zero-resilient evaluation metrics (MAE, RMSE, WAPE, and safe non-zero MAPE).
- Generate multi-step recursive forecasts across 7, 14, 30, 60, and 90-day horizons with 95% confidence intervals.
- Persist dataset metadata, model evaluation metrics, and forecast records in an embedded SQLite database.
- Provide export capabilities for CSV, Excel (.xlsx), and Markdown executive reports.

---

## 4. Proposed System
The proposed application is a self-contained, dashboard-driven Python web system built with Streamlit. It abstracts complex machine learning operations behind an intuitive user interface, allowing non-technical managers and business analysts to upload sales files, train models, inspect diagnostic residual plots, and export future sales schedules in seconds.

---

## 5. Methodology
The development follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology tailored for temporal data:
1. **Business Understanding:** Identifying inventory replenishment and sales quota requirements.
2. **Data Understanding & Validation:** Checking schema integrity, null ratios, negative sales, and temporal coverage.
3. **Data Cleaning:** Standardizing headers, parsing timestamps, removing duplicates, clipping negative records.
4. **Feature Engineering:** Generating calendar and lag/rolling features with $t-1$ shifting to avoid lookahead leakage.
5. **Chronological Splitting:** Partitioning historical data chronologically (e.g. 85% train, 15% holdout test).
6. **Model Training & Comparison:** Fitting candidate parametric, non-parametric ensemble, and econometric time series models.
7. **Model Evaluation:** Computing test set MAE, RMSE, WAPE, and residual variance.
8. **Recursive Forecasting:** Iteratively predicting future horizons step-by-step and compounding lag features forward.
9. **Deployment & Logging:** Serving results via Streamlit and persisting audit logs in SQLite.

---

## 6. Technology Stack

| Domain | Technology / Library | Version | Purpose |
|---|---|---|---|
| **Programming Language** | Python | 3.13.x | Core system language |
| **Data Manipulation** | Pandas, NumPy | >= 2.0.0 | Tabular data processing and numerical arrays |
| **Machine Learning** | Scikit-learn | >= 1.3.0 | Regression models, ensembles, and evaluation |
| **Time Series Modeling** | Statsmodels | >= 0.14.0 | ARIMA time series econometric modeling |
| **Interactive Visualization** | Plotly | >= 5.18.0 | Interactive line charts, distributions, and heatmaps |
| **Web Dashboard** | Streamlit | >= 1.30.0 | Reactive user interface and session management |
| **Spreadsheet Engine** | OpenPyXL | >= 3.1.0 | Excel file ingestion and multi-sheet export |
| **Model Persistence** | Joblib | >= 1.3.0 | Serialization of trained models and metadata |
| **Database** | SQLite3 | Built-in | Embedded relational storage for audit logs |
| **Testing** | Pytest | >= 8.0.0 | Unit and integration test verification |

---

## 7. System Architecture

```
Data Ingestion (CSV / Excel)
            │
            ▼
Data Validation & Schema Auto-Detection
            │
            ▼
Data Cleaning & Chronological Sorting
            │
            ▼
Exploratory Data Analysis (Plotly Visualizations)
            │
            ▼
Feature Engineering (Lags, Rolling Windows, Cyclical Encodings)
            │
            ▼
Chronological Train / Test Split (Strict Zero Leakage)
            │
            ▼
ML & Time-Series Engine (Baseline, Linear, RF, GB, ARIMA)
            │
            ▼
Model Evaluation (MAE, RMSE, Safe MAPE, WAPE, R², Residuals)
            │
            ▼
Future Recursive Forecast Generation (7, 14, 30, 60, 90 Days)
            │
            ▼
Interactive Streamlit Dashboard & Reports Export (CSV / Excel / SQLite)
```

---

## 8. Application Features
- **Auto-Synonym Column Mapping:** Intelligently maps names such as `Order Date`, `Transaction Date`, `Total Revenue`, `Units Sold` to standard canonical fields.
- **Data Quality Audit:** Real-time badge indicators displaying null counts, duplicates, date spans, and negative sales warnings.
- **Interactive Multi-Level EDA:** Filter historical analysis by individual product SKUs or regional territories.
- **Leak-Free Rolling Statistics:** Rolling mean, standard deviation, and median strictly applied to shifted historical data ($t-1$).
- **Multi-Model Tournament:** Real-time side-by-side leaderboard ranking all models by test RMSE and MAE.
- **Dynamic Recursive Forecasting:** Predicts arbitrary future days by recursively updating lag vectors.
- **Prediction Uncertainty Bands:** Computes 95% confidence intervals ($1.96 \times \text{RMSE}$) showing forecast risk.
- **Multi-Format Exporting:** Downloads forecasts as CSV, formatted Excel (.xlsx) workbooks, and Markdown executive briefings.
- **Embedded SQLite Audit Trail:** Queries historical dataset uploads, model metrics, and forecast records directly within the dashboard.

---

## 9. Dataset Details
The application includes a realistic synthetic retail sales benchmark dataset (`sample_sales_data.csv` and `sample_sales_data.xlsx`):
- **Span:** 2 full calendar years (2023-01-01 to 2024-12-31, 731 continuous calendar days).
- **Volume:** 14,620 granular retail records.
- **Product SKUs:**
  - `Laptop Pro 15` (High unit value, strong Q4 back-to-school / festive uplift)
  - `Smartphone Ultra` (Consistent high-volume demand, promotional sensitivity)
  - `Wireless Noise-Canceling Headphones` (Frequent promotional flash-sale spikes)
  - `Smart Fitness Watch` (New Year resolution spike in January)
  - `Ergonomic Office Chair` (Steady corporate replacement cycle)
- **Geographic Regions:** North, South, East, West (with realistic regional purchasing power weightings).
- **Realistic Attributes:** Calendar holidays, weekend retail boosts (+30%), promotional discount markdowns (10-15%), and annual trend growth (+12% YoY).

*Note: The included dataset is synthetically generated for demonstration and academic testing purposes.*

---

## 10. Machine Learning Models

### 1. Naive Baseline (7-Day Moving Average)
- Represents the average daily sales of the most recent 7 days.
- Serves as the benchmark; any advanced machine learning model must beat this baseline.

### 2. Linear Regression
- Fast, interpretable parametric model trained on engineered time features and lag variables.
- Captures linear trends and steady baseline shifts.

### 3. Random Forest Regressor
- Non-parametric ensemble of 100 decorrelated decision trees using bootstrap aggregation (bagging).
- Captures complex non-linear interactions, weekend spikes, and sudden holiday shifts without overfitting.

### 4. Gradient Boosting Regressor (HistGradientBoosting)
- Tree ensemble that sequentially builds trees to minimize the residual loss of prior iterations.
- Highly effective on tabular time-series features with complex gradient landscapes.

### 5. ARIMA (AutoRegressive Integrated Moving Average)
- Classical econometric model $(p=1, d=1, q=1)$ that models autocorrelation and moving-average shocks in stationary differenced series.

---

## 11. Evaluation Metrics
The project implements four core regression metrics evaluated strictly on the unseen holdout test set:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
   Measures average prediction error magnitude in original dollar units.

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
   Penalizes large outlier forecast errors heavily; used as the primary ranking criterion.

3. **Safe Mean Absolute Percentage Error (MAPE):**
   $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
   *Crucial Implementation Detail:* When actual sales $y_i = 0$, standard MAPE results in division by zero. The system detects zeros and computes safe MAPE on non-zero observations with an explicit explanatory notice.

4. **Weighted Absolute Percentage Error (WAPE):**
   $$\text{WAPE} = \frac{\sum_{i=1}^n |y_i - \hat{y}_i|}{\sum_{i=1}^n |y_i|} \times 100\%$$
   The industry-standard percentage metric when zero or low-volume actual sales exist, immune to division-by-zero distortion.

5. **Coefficient of Determination ($R^2$ Score):**
   Proportion of variance in sales explained by the engineered model features.

---

## 12. Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- pip package manager

### Steps
1. Navigate to the project directory:
   ```bash
   cd /Users/tanmaydigambartaras/Desktop/lokesh
   ```
2. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

---

## 13. How to Run

### Option A: Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard opens automatically in your web browser at:
`http://localhost:8501`

### Option B: Run Standalone Terminal Training Pipeline
To run the full end-to-end training pipeline and log metrics directly from the terminal:
```bash
python3 train_models.py
```

### Option C: Run Pytest Test Suite
To verify the complete test suite:
```bash
python3 -m pytest tests/test_pipeline.py -v
```

---

## 14. Dashboard Usage Guide

1. **Tab 1: Overview:** Review executive KPI summary cards, business objectives, and system architecture.
2. **Tab 2: Data Upload & Validation:**
   - Upload any custom CSV/Excel sales file, or click **"Load Built-in Benchmark Data"**.
   - Check the automated schema validation report.
   - Click **"Execute Cleaning Pipeline"** to sanitize records and log the dataset to SQLite.
3. **Tab 3: Exploratory Data Analysis (EDA):**
   - Filter by product SKU or region.
   - Analyze daily trend line with 7-day and 30-day moving averages.
   - Inspect monthly totals, regional pie charts, day-of-week seasonality, and promotional uplift box plots.
4. **Tab 4: Model Training:**
   - Select product/region targets and test split percentage (default 15%).
   - Observe the exact chronological training and testing date boundaries.
   - Click **"Train & Compare All 5 Models"** to train models and view the leaderboard ranking.
5. **Tab 5: Model Evaluation:**
   - Review MAE, RMSE, safe MAPE, WAPE, and $R^2$ scores.
   - Inspect the interactive Plotly curve comparing actual sales vs. model predictions.
   - View the residual error distribution histogram and mean bias metrics.
6. **Tab 6: Sales Forecast:**
   - Select forecasting horizon (7, 14, 30, 60, or 90 days).
   - Select product, region, and forecasting model (defaults to top performer).
   - Click **"Generate Future Forecast"** to view projected revenue with 95% confidence bands.
7. **Tab 7: Reports & Database:**
   - Download future forecast in CSV or Excel format.
   - Download model comparison metrics.
   - Download the generated Executive Briefing markdown document.
   - Query SQLite audit records for uploaded datasets, trained models, and saved forecasts.

---

## 15. Project Structure

```
lokesh/
│
├── app.py                      # Main Streamlit interactive web dashboard (7 tabs)
├── train_models.py             # Standalone CLI training & evaluation pipeline
├── preprocessing.py            # Column detection, validation, cleaning, & file loading
├── forecasting.py              # Feature engineering, chronological split, 5 models, recursive forecast
├── evaluation.py               # Evaluation metrics (MAE, RMSE, safe MAPE, WAPE, R²), Plotly figures
├── database.py                 # SQLite database storage & query interface
├── generate_sample_data.py     # Realistic 2-year synthetic sales data generator
├── requirements.txt            # Python library dependencies
├── README.md                   # Comprehensive academic and technical documentation
├── VIVA_NOTES.md               # Viva examination preparation Q&A guide for BSc Data Science
├── .gitignore                  # Git ignore rules for virtualenvs and artifacts
│
├── data/
│   ├── raw/
│   │   ├── sample_sales_data.csv    # Benchmark retail sales dataset (CSV)
│   │   └── sample_sales_data.xlsx   # Benchmark retail sales dataset (Excel)
│   └── processed/
│       └── cleaned_sales_data.csv   # Cleaned, standardized model-ready dataset
│
├── models/
│   └── saved_models/
│       └── best_forecasting_model.pkl # Serialized top-performing model & metadata
│
├── reports/
│   └── forecast_30days.csv          # Sample exported 30-day future forecast
│
├── database/
│   └── sales_forecasting.db         # Relational SQLite database file
│
└── tests/
    └── test_pipeline.py             # 12 automated pytest unit and integration tests
```

---

## 16. Limitations
1. **Unforeseen Macro Shocks:** Pure statistical and ML models cannot anticipate black swan events (e.g. global supply chain shutdowns or competitor liquidations) without external shock regressors.
2. **Cold-Start SKUs:** Products with fewer than 14 days of historical records cannot generate robust lag or rolling features.
3. **Constant Price Assumption:** Recursive forecasting assumes future pricing remains aligned with historical averages unless explicitly parameterized.
4. **Local Hardware Constraints:** High-dimensional recursive simulation for thousands of individual SKUs simultaneously is best suited for distributed background worker queues.

---

## 17. Future Scope
1. **Exogenous Economic Regressors:** Integrate real-time inflation indices, weather patterns, and competitor pricing APIs.
2. **Hierarchical Forecasting:** Implement bottom-up and top-down reconciliation (e.g. MinT algorithm) across store, city, and state hierarchies.
3. **Deep Learning Integration:** Incorporate Temporal Fusion Transformers (TFT) or DeepAR for large-scale multi-horizon probabilistic forecasting.
4. **Automated Reorder Point Triggers:** Direct API connectivity with ERP systems (SAP, Oracle) to trigger automated purchase orders when forecasted stock falls below buffer levels.
