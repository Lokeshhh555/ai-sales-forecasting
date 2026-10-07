"""
AI SALES FORECASTING - Streamlit Web Application
Student: Lokesh | BSc Data Science Final Year Project
Subtitle: Intelligent Sales Prediction & Business Planning
"""

import os
import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import (
    load_dataset_from_file,
    validate_raw_dataset,
    clean_and_prepare_data,
    detect_and_standardize_columns
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
from evaluation import (
    calculate_metrics,
    create_actual_vs_predicted_figure,
    create_residuals_figure,
    calculate_residuals
)
from database import (
    init_db,
    log_dataset,
    log_model_metrics,
    log_forecast,
    get_dataset_history,
    get_saved_metrics,
    get_saved_forecasts
)

# Set page config
st.set_page_config(
    page_title="AI Sales Forecasting | Final Year Project",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern business analytics dashboard
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        font-weight: 500;
        margin-top: 0px;
        margin-bottom: 1.5rem;
    }
    .student-badge {
        background: linear-gradient(135deg, #1E3A8A, #3B82F6);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #10B981;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .best-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #86EFAC;
    }
</style>
""", unsafe_allow_html=True)

# Ensure DB is initialized
init_db()

# Sample dataset path
SAMPLE_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "sample_sales_data.csv")
SAMPLE_XLSX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "sample_sales_data.xlsx")

# Initialize session state variables
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "validation_report" not in st.session_state:
    st.session_state.validation_report = None
if "cleaning_summary" not in st.session_state:
    st.session_state.cleaning_summary = None
if "dataset_filename" not in st.session_state:
    st.session_state.dataset_filename = None
if "model_store" not in st.session_state:
    st.session_state.model_store = None
if "comparison_df" not in st.session_state:
    st.session_state.comparison_df = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "split_info" not in st.session_state:
    st.session_state.split_info = None
if "test_dates" not in st.session_state:
    st.session_state.test_dates = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None
if "daily_df" not in st.session_state:
    st.session_state.daily_df = None


# Auto-load sample data if nothing is loaded yet
def load_sample_dataset_action():
    if os.path.exists(SAMPLE_CSV_PATH):
        df = pd.read_csv(SAMPLE_CSV_PATH)
        st.session_state.raw_df = df
        st.session_state.dataset_filename = "sample_sales_data.csv"
        st.session_state.validation_report = validate_raw_dataset(df)
        cleaned, c_sum = clean_and_prepare_data(df)
        st.session_state.cleaned_df = cleaned
        st.session_state.cleaning_summary = c_sum
        st.session_state.daily_df = prepare_daily_series(cleaned)
        return True
    return False

if st.session_state.raw_df is None and os.path.exists(SAMPLE_CSV_PATH):
    load_sample_dataset_action()


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown('<div class="student-badge">🎓 BSc Data Science Final Year Project</div>', unsafe_allow_html=True)
    st.markdown("### 📊 AI Sales Forecasting")
    st.caption("**Student:** Lokesh  \n**Domain:** Time Series & Machine Learning")

    st.markdown("---")
    st.subheader("📁 Data Source")

    if st.button("🔄 Reload Demo Dataset", use_container_width=True, help="Load the 2-year multi-product retail benchmark dataset"):
        if load_sample_dataset_action():
            st.success("Sample Retail Dataset loaded!")
            st.rerun()

    if st.session_state.dataset_filename:
        st.info(f"**Active File:** `{st.session_state.dataset_filename}`")
    else:
        st.warning("No dataset loaded yet.")

    st.markdown("---")
    st.subheader("⚙️ System Status")
    if st.session_state.cleaned_df is not None:
        st.markdown(f"✅ **Cleaned Records:** {len(st.session_state.cleaned_df):,}")
        st.markdown(f"📅 **Span:** {st.session_state.cleaning_summary['min_date']} to {st.session_state.cleaning_summary['max_date']}")
    else:
        st.markdown("⏳ Data pending cleaning")

    if st.session_state.model_store is not None:
        st.markdown(f"✅ **Trained Models:** {len(st.session_state.model_store)}")
        st.markdown(f"★ **Top Model:** `{st.session_state.best_model_name}`")
    else:
        st.markdown("⏳ Models pending training")

    st.markdown("---")
    st.caption("Designed for Viva Examination & Decision-Support Planning: Inventory, Purchasing, Revenue Targets.")


# ==========================================
# HEADER
# ==========================================
col_hdr1, col_hdr2 = st.columns([4, 1])
with col_hdr1:
    st.markdown('<h1 class="main-title">📈 AI SALES FORECASTING</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Intelligent Sales Prediction & Business Planning Dashboard</p>', unsafe_allow_html=True)
with col_hdr2:
    st.write("")
    if st.session_state.best_model_name:
        st.markdown(f'<div style="text-align:right;"><span class="best-badge">★ Top Model: {st.session_state.best_model_name}</span></div>', unsafe_allow_html=True)


# ==========================================
# NAVIGATION TABS (7 Core Tabs per FRD/PRD)
# ==========================================
tabs = st.tabs([
    "1. Overview",
    "2. Data Upload & Validation",
    "3. Exploratory Data Analysis (EDA)",
    "4. Model Training",
    "5. Model Evaluation",
    "6. Sales Forecast",
    "7. Reports & Database"
])


# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tabs[0]:
    st.markdown("### 📌 Executive Overview")
    st.write("Welcome to the **AI Sales Forecasting System**, developed by **Lokesh** for final-year graduation in **BSc Data Science**. This application bridges machine learning with operational retail analytics to generate high-accuracy, leak-free sales projections.")

    # High-level KPIs if data loaded
    if st.session_state.cleaning_summary is not None:
        c_sum = st.session_state.cleaning_summary
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-value">${c_sum['total_sales']:,.0f}</div>
                <div class="kpi-sub">Cleaned Historic Total</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Units Sold</div>
                <div class="kpi-value">{c_sum['total_quantity']:,.0f}</div>
                <div class="kpi-sub">Volume Traded</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Products</div>
                <div class="kpi-value">{c_sum['unique_products']}</div>
                <div class="kpi-sub">Active SKUs</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Regions</div>
                <div class="kpi-value">{c_sum['unique_regions']}</div>
                <div class="kpi-sub">Sales Territories</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Date Span</div>
                <div class="kpi-value">{st.session_state.validation_report['summary']['date_span_days']} Days</div>
                <div class="kpi-sub">{c_sum['min_date']} → {c_sum['max_date']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

    col_ov1, col_ov2 = st.columns([1, 1])

    with col_ov1:
        st.markdown("#### 🎯 Business Objectives Supported")
        st.markdown("""
        * **Inventory Planning:** Prevent overstocking capital locks and stockout penalties by projecting daily inventory velocity.
        * **Purchasing & Procurement:** Enable procurement managers to place bulk orders ahead of seasonal demand spikes.
        * **Revenue Target Allocation:** Provide data-backed baseline forecasts for realistic regional quota setting.
        * **Marketing & Promotional Optimization:** Evaluate the volume uplift driven by promotional events and calendar holidays.
        """)

    with col_ov2:
        st.markdown("#### 🔬 Machine Learning Architecture")
        st.markdown("""
        1. **Data Ingestion:** Upload CSV or Excel files with flexible column synonym auto-mapping.
        2. **Validation & Cleaning:** Deduplication, negative value handling, missing data imputation, chronological sort.
        3. **Strict Feature Engineering:** Lag terms ($t-1, t-7, t-14, t-30$), rolling statistics ($7, 14, 30$ days), and calendar Fourier encodings with strict lookahead prevention.
        4. **Chronological Splitting:** Preserves temporal causality (no future-data leakage).
        5. **Multi-Model Tournament:** Compares 5 candidate algorithms: Baseline, Linear Regression, Random Forest, Gradient Boosting, and ARIMA.
        6. **Model Governance & Export:** Metrics logged to SQLite database, downloadable CSV/Excel forecasts.
        """)


# ==============================================================================
# TAB 2: DATA UPLOAD & VALIDATION
# ==============================================================================
with tabs[1]:
    st.markdown("### 📤 Data Ingestion & Preprocessing Pipeline")
    st.write("Upload your historical enterprise sales dataset in **CSV** or **Excel** format, or verify the loaded demonstration benchmark.")

    upload_col, preset_col = st.columns([3, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Files should contain Date, Sales (or Quantity & Unit Price), and optional Product/Region."
        )
    with preset_col:
        st.write("")
        st.write("")
        if st.button("📥 Load Built-in Benchmark Data", use_container_width=True):
            if load_sample_dataset_action():
                st.success("Benchmark dataset loaded successfully!")
                st.rerun()

    # Handle newly uploaded file
    if uploaded_file is not None:
        file_df, err = load_dataset_from_file(uploaded_file, uploaded_file.name)
        if err:
            st.error(f"❌ File Load Error: {err}")
        else:
            st.session_state.raw_df = file_df
            st.session_state.dataset_filename = uploaded_file.name
            st.session_state.validation_report = validate_raw_dataset(file_df)
            st.success(f"File `{uploaded_file.name}` uploaded and read successfully.")

    if st.session_state.raw_df is not None:
        raw_df = st.session_state.raw_df
        val_rep = st.session_state.validation_report

        st.markdown("---")
        st.markdown("#### 1. Data Schema & Validation Report")

        if val_rep["is_valid"]:
            st.success("✅ **Dataset Validation Passed:** The schema satisfies time series modeling prerequisites.")
        else:
            st.error("❌ **Dataset Validation Failed:** Missing critical prerequisites.")
            for e in val_rep["errors"]:
                st.write(f"- 🔴 {e}")

        if val_rep["warnings"]:
            for w in val_rep["warnings"]:
                st.warning(f"⚠️ {w}")

        # Summary statistics metrics
        s = val_rep["summary"]
        v1, v2, v3, v4, v5, v6 = st.columns(6)
        v1.metric("Total Records", f"{s.get('total_rows', len(raw_df)):,}")
        v2.metric("Total Columns", f"{s.get('total_columns', len(raw_df.columns))}")
        v3.metric("Duplicate Rows", f"{s.get('duplicate_rows', 0):,}")
        v4.metric("Missing Values", f"{s.get('missing_values_total', 0):,}")
        v5.metric("Date Span", f"{s.get('date_span_days', 0)} Days")
        v6.metric("Estimated Revenue", f"${s.get('estimated_total_sales', 0):,.0f}")

        # Cleaning trigger
        st.markdown("---")
        st.markdown("#### 2. Preprocessing & Cleaning Engine")

        clean_btn = st.button("🧹 Execute Cleaning Pipeline", type="primary", use_container_width=False)
        if clean_btn or st.session_state.cleaned_df is None:
            cleaned, clean_stats = clean_and_prepare_data(raw_df)
            st.session_state.cleaned_df = cleaned
            st.session_state.cleaning_summary = clean_stats
            st.session_state.daily_df = prepare_daily_series(cleaned)
            # Log to SQLite
            log_dataset(
                filename=st.session_state.dataset_filename or "uploaded_data.csv",
                row_count=clean_stats["final_rows"],
                col_count=len(cleaned.columns),
                min_date=clean_stats["min_date"],
                max_date=clean_stats["max_date"],
                total_sales=clean_stats["total_sales"],
                total_quantity=clean_stats["total_quantity"],
                num_products=clean_stats["unique_products"],
                num_regions=clean_stats["unique_regions"]
            )
            st.success("✅ Data cleaned, chronologically sorted, and logged to SQLite!")

        if st.session_state.cleaning_summary is not None:
            cs = st.session_state.cleaning_summary
            st.markdown(f"""
            **Cleaning Audit Trail:**
            - **Raw Rows:** {cs['initial_rows']:,} ➔ **Cleaned Rows:** {cs['final_rows']:,}
            - **Deduplication:** {cs['duplicates_removed']} duplicate records dropped
            - **Invalid Dates:** {cs['invalid_dates_dropped']} unparseable records dropped
            - **Negative Sales Filter:** {cs['negative_sales_clipped']} negative values clipped to 0
            - **Missing Sales Imputation:** {cs['missing_sales_filled']} missing sales values populated
            """)

        # Preview Data Tabs
        st.markdown("#### 3. Dataset Previews")
        prev_tab1, prev_tab2 = st.tabs(["Cleaned Data (Model-Ready)", "Raw Uploaded Data"])
        with prev_tab1:
            if st.session_state.cleaned_df is not None:
                st.dataframe(st.session_state.cleaned_df.head(50), use_container_width=True)
            else:
                st.info("Execute cleaning to preview standardized data.")
        with prev_tab2:
            st.dataframe(raw_df.head(50), use_container_width=True)


# ==============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tabs[2]:
    st.markdown("### 🔍 Exploratory Data Analysis & Seasonality Diagnostics")
    st.write("Understand historical sales distributions, seasonal cycles, day-of-week demand velocity, and product/regional performance.")

    if st.session_state.cleaned_df is None:
        st.warning("Please upload and clean a dataset in Tab 2 to view EDA diagnostics.")
    else:
        clean_df = st.session_state.cleaned_df
        daily_df = st.session_state.daily_df

        # Filter row for EDA
        eda_f1, eda_f2 = st.columns(2)
        with eda_f1:
            product_list = ["All Products"] + sorted(clean_df["product"].unique().tolist())
            sel_eda_prod = st.selectbox("Filter Product", product_list, key="eda_prod")
        with eda_f2:
            region_list = ["All Regions"] + sorted(clean_df["region"].unique().tolist())
            sel_eda_reg = st.selectbox("Filter Region", region_list, key="eda_reg")

        # Filtered subset
        eda_df = clean_df.copy()
        if sel_eda_prod != "All Products":
            eda_df = eda_df[eda_df["product"] == sel_eda_prod]
        if sel_eda_reg != "All Regions":
            eda_df = eda_df[eda_df["region"] == sel_eda_reg]

        # Daily aggregated series for chart
        daily_eda = eda_df.groupby("date")["sales"].sum().reset_index()

        # 1. Sales over time with rolling moving average
        st.markdown("#### 1. Daily Sales Trajectory & Trend Decomposition")
        daily_eda["MA_7"] = daily_eda["sales"].rolling(7, min_periods=1).mean()
        daily_eda["MA_30"] = daily_eda["sales"].rolling(30, min_periods=1).mean()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=daily_eda["date"], y=daily_eda["sales"],
            mode="lines", name="Daily Actual Sales",
            line=dict(color="#93C5FD", width=1.5)
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily_eda["date"], y=daily_eda["MA_7"],
            mode="lines", name="7-Day Moving Avg",
            line=dict(color="#2563EB", width=2.5)
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily_eda["date"], y=daily_eda["MA_30"],
            mode="lines", name="30-Day Moving Avg (Trend)",
            line=dict(color="#DC2626", width=2.5, dash="dash")
        ))
        fig_ts.update_layout(
            xaxis_title="Date", yaxis_title="Sales Revenue ($)",
            template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # 2. Monthly and Product Performance
        c_eda1, c_eda2 = st.columns(2)

        with c_eda1:
            st.markdown("#### 2. Monthly Revenue Performance")
            eda_df["year_month"] = eda_df["date"].dt.strftime("%Y-%m")
            monthly_sales = eda_df.groupby("year_month")["sales"].sum().reset_index()
            fig_monthly = px.bar(
                monthly_sales, x="year_month", y="sales",
                color="sales", color_continuous_scale="Blues",
                labels={"year_month": "Month", "sales": "Sales ($)"},
                title="Monthly Aggregated Sales Revenue"
            )
            fig_monthly.update_layout(template="plotly_white")
            st.plotly_chart(fig_monthly, use_container_width=True)

        with c_eda2:
            st.markdown("#### 3. Product Sales Contribution")
            prod_summary = clean_df.groupby("product")["sales"].sum().reset_index().sort_values("sales", ascending=True)
            fig_prod = px.bar(
                prod_summary, y="product", x="sales",
                orientation="h", color="sales", color_continuous_scale="Viridis",
                labels={"product": "Product SKU", "sales": "Total Sales ($)"},
                title="Product Revenue Breakdown"
            )
            fig_prod.update_layout(template="plotly_white")
            st.plotly_chart(fig_prod, use_container_width=True)

        # 3. Regional Breakdown and Day of Week Seasonality
        c_eda3, c_eda4 = st.columns(2)

        with c_eda3:
            st.markdown("#### 4. Regional Territory Comparison")
            reg_summary = clean_df.groupby("region")["sales"].sum().reset_index()
            fig_reg = px.pie(
                reg_summary, values="sales", names="region",
                hole=0.45,
                title="Regional Share of Total Revenue",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_reg.update_layout(template="plotly_white")
            st.plotly_chart(fig_reg, use_container_width=True)

        with c_eda4:
            st.markdown("#### 5. Day-of-Week Seasonality (Cyclical Pattern)")
            clean_df["day_name"] = clean_df["date"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_sales = clean_df.groupby("day_name")["sales"].mean().reindex(day_order).reset_index()

            fig_dow = px.bar(
                dow_sales, x="day_name", y="sales",
                labels={"day_name": "Day of Week", "sales": "Average Sales ($)"},
                color="sales", color_continuous_scale="Purples",
                title="Mean Daily Sales by Day of Week"
            )
            fig_dow.update_layout(template="plotly_white")
            st.plotly_chart(fig_dow, use_container_width=True)

        # 4. Promotion Uplift Analysis
        if "promotion" in clean_df.columns:
            st.markdown("#### 6. Promotional Uplift Analysis")
            promo_stats = clean_df.groupby("promotion")["sales"].agg(["count", "mean", "median", "sum"]).reset_index()
            promo_stats["promotion_label"] = promo_stats["promotion"].map({0: "Standard Non-Promo Days", 1: "Promotional Flash Sale Days"})

            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.dataframe(
                    promo_stats[["promotion_label", "mean", "median", "sum"]].rename(
                        columns={"promotion_label": "Campaign", "mean": "Mean Sales", "median": "Median Sales", "sum": "Total Sales"}
                    ),
                    use_container_width=True
                )
            with p_col2:
                fig_promo = px.box(
                    clean_df, x="promotion", y="sales",
                    color="promotion",
                    labels={"promotion": "Promotion Flag (0=No, 1=Yes)", "sales": "Sales ($)"},
                    title="Sales Distribution: Promo vs Non-Promo",
                    color_discrete_sequence=["#64748B", "#10B981"]
                )
                fig_promo.update_layout(template="plotly_white")
                st.plotly_chart(fig_promo, use_container_width=True)


# ==============================================================================
# TAB 4: MODEL TRAINING
# ==============================================================================
with tabs[3]:
    st.markdown("### 🤖 Chronological Machine Learning Engine")
    st.write("Train multiple forecasting models using a strict temporal split to prevent data leakage.")

    if st.session_state.cleaned_df is None:
        st.warning("Please upload and clean a dataset in Tab 2 before training models.")
    else:
        clean_df = st.session_state.cleaned_df

        # Target filtering for training
        tr_c1, tr_c2, tr_c3 = st.columns(3)
        with tr_c1:
            tr_prod = st.selectbox("Select Target Product", ["All Products"] + sorted(clean_df["product"].unique().tolist()), key="tr_prod")
        with tr_c2:
            tr_reg = st.selectbox("Select Target Region", ["All Regions"] + sorted(clean_df["region"].unique().tolist()), key="tr_reg")
        with tr_c3:
            test_split_pct = st.slider("Chronological Test Set Size (%)", min_value=10, max_value=30, value=15, step=5)

        # Prepare target daily series
        prod_filter = "All" if tr_prod == "All Products" else tr_prod
        reg_filter = "All" if tr_reg == "All Regions" else tr_reg
        series_for_training = prepare_daily_series(clean_df, product=prod_filter, region=reg_filter)

        # Engineer features
        featured_series = create_features(series_for_training)
        train_df, test_df, split_info = split_chronological(featured_series, test_ratio=test_split_pct / 100.0)

        # Display temporal boundaries
        st.markdown("#### ⏳ Chronological Split Boundary (Zero Leakage)")
        sp1, sp2 = st.columns(2)
        with sp1:
            st.info(f"**Training Set (Older Observations):**  \n📅 **{split_info['train_start']}** to **{split_info['train_end']}** ({split_info['train_count']} days)")
        with sp2:
            st.success(f"**Holdout Test Set (Recent Future):**  \n📅 **{split_info['test_start']}** to **{split_info['test_end']}** ({split_info['test_count']} days)")

        st.caption("ℹ️ *Note: Random shuffling is strictly disabled for time series. The models are trained only on historical records and evaluated on subsequent unseen periods.*")

        st.markdown("---")
        st.markdown("#### Candidate Models Included:")
        m_list_col1, m_list_col2 = st.columns(2)
        with m_list_col1:
            st.markdown("""
            1. **Baseline Model:** 7-Day Moving Average naive persistence.
            2. **Linear Regression:** Baseline parametric regression on engineered lag & time features.
            3. **Random Forest Regressor:** Non-linear ensemble of 100 decision trees.
            """)
        with m_list_col2:
            st.markdown("""
            4. **Gradient Boosting Regressor:** Sequential boosting optimizing residual errors.
            5. **ARIMA (AutoRegressive Integrated Moving Average):** Classical econometric time series model capturing autocorrelation.
            """)

        # Training action button
        train_trigger = st.button("🚀 Train & Compare All 5 Models", type="primary", use_container_width=True)

        if train_trigger or (st.session_state.model_store is None and st.session_state.cleaned_df is not None):
            with st.spinner("Training models chronologically and evaluating on holdout test set..."):
                model_store, comp_df, best_model_name = train_and_evaluate_all_models(train_df, test_df)

                # Persist in session state
                st.session_state.model_store = model_store
                st.session_state.comparison_df = comp_df
                st.session_state.best_model_name = best_model_name
                st.session_state.split_info = split_info
                st.session_state.test_dates = test_df["date"]
                st.session_state.y_test = test_df["sales"].values

                # Save best model to disk
                save_forecasting_pipeline(
                    model_store=model_store,
                    best_model_name=best_model_name,
                    comparison_df=comp_df,
                    target_info={"product": prod_filter, "region": reg_filter}
                )

                # Log metrics to SQLite
                metrics_to_log = []
                for _, row in comp_df.iterrows():
                    metrics_to_log.append({
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
                log_model_metrics(
                    metrics_to_log,
                    dataset_name=st.session_state.dataset_filename or "active_dataset.csv",
                    target_level=f"{prod_filter} / {reg_filter}"
                )

                st.success(f"✅ Training completed! Best Model: **{best_model_name}**")

        if st.session_state.comparison_df is not None:
            st.markdown("---")
            st.markdown(f"#### 🏆 Model Tournament Leaderboard: Top Model = `{st.session_state.best_model_name}`")
            st.dataframe(
                st.session_state.comparison_df.style.highlight_min(subset=["RMSE", "MAE", "MAPE (%)"], color="#DCFCE7"),
                use_container_width=True
            )

            # Feature Importance for tree models
            rf_item = st.session_state.model_store.get("Random Forest")
            if rf_item and hasattr(rf_item.get("model"), "feature_importances_"):
                st.markdown("#### 🌲 Feature Importance (Random Forest)")
                rf_model = rf_item["model"]
                fi_df = pd.DataFrame({
                    "Feature": FEATURE_COLUMNS,
                    "Importance": rf_model.feature_importances_
                }).sort_values("Importance", ascending=True)

                fig_fi = px.bar(
                    fi_df, y="Feature", x="Importance", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                    title="Relative Feature Importance in Forecasting"
                )
                fig_fi.update_layout(template="plotly_white")
                st.plotly_chart(fig_fi, use_container_width=True)


# ==============================================================================
# TAB 5: MODEL EVALUATION
# ==============================================================================
with tabs[4]:
    st.markdown("### 📊 Model Evaluation & Diagnostic Visualizations")
    st.write("Examine test metrics, compare actual vs. predicted curves, and inspect residual distribution to verify model unbiasedness.")

    if st.session_state.model_store is None or st.session_state.comparison_df is None:
        st.warning("Please train models in Tab 4 first to inspect evaluation diagnostics.")
    else:
        comp_df = st.session_state.comparison_df
        best_name = st.session_state.best_model_name
        model_store = st.session_state.model_store
        test_dates = st.session_state.test_dates
        y_test = st.session_state.y_test

        # Best Model KPI Highlights
        best_metrics = model_store[best_name]["metrics"]
        bm1, bm2, bm3, bm4, bm5 = st.columns(5)
        bm1.metric("Top Model", best_name)
        bm2.metric("MAE", f"${best_metrics['mae']:,.2f}")
        bm3.metric("RMSE", f"${best_metrics['rmse']:,.2f}")
        bm4.metric("MAPE", f"{best_metrics['mape']:.2f}%")
        bm5.metric("R² Score", f"{best_metrics['r2']:.4f}")

        # Safe MAPE Note
        if best_metrics.get("has_zeros"):
            st.info(f"💡 **Evaluation Note on MAPE:** {best_metrics.get('mape_note')} WAPE ({best_metrics.get('wape')}%) is also provided for zero-resilient evaluation.")
        else:
            st.caption(f"💡 {best_metrics.get('mape_note')}")

        st.markdown("---")

        # 1. Actual vs Predicted Line Chart
        st.markdown("#### 1. Holdout Test Set: Actual vs Predicted Sales")
        predictions_map = {m: model_store[m]["predictions"] for m in model_store if "predictions" in model_store[m]}
        fig_eval = create_actual_vs_predicted_figure(test_dates, y_test, predictions_map, best_model=best_name)
        st.plotly_chart(fig_eval, use_container_width=True)

        # 2. Residual Diagnostics
        st.markdown("---")
        st.markdown("#### 2. Residual Error Analysis")
        res_col1, res_col2 = st.columns([1, 1])

        sel_res_model = st.selectbox("Inspect Residuals for Model", list(predictions_map.keys()), index=list(predictions_map.keys()).index(best_name))
        chosen_preds = predictions_map[sel_res_model]
        residuals_info = calculate_residuals(y_test, chosen_preds)

        with res_col1:
            fig_res_time = create_residuals_figure(test_dates, y_test, chosen_preds, sel_res_model)
            st.plotly_chart(fig_res_time, use_container_width=True)

        with res_col2:
            fig_res_hist = px.histogram(
                x=residuals_info["residuals"],
                nbins=25,
                title=f"Residual Error Distribution ({sel_res_model})",
                labels={"x": "Prediction Error ($)", "count": "Frequency"},
                color_discrete_sequence=["#DC2626"]
            )
            fig_res_hist.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Zero Error")
            fig_res_hist.update_layout(template="plotly_white")
            st.plotly_chart(fig_res_hist, use_container_width=True)

        # Residual Summary Cards
        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        r_c1.metric("Mean Error (Bias)", f"${residuals_info['mean_residual']:,.2f}")
        r_c2.metric("Std Error", f"${residuals_info['std_residual']:,.2f}")
        r_c3.metric("Min Error", f"${residuals_info['min_residual']:,.2f}")
        r_c4.metric("Max Error", f"${residuals_info['max_residual']:,.2f}")


# ==============================================================================
# TAB 6: SALES FORECAST
# ==============================================================================
with tabs[5]:
    st.markdown("### 🔮 Out-of-Sample Sales Forecast Generation")
    st.write("Generate future sales projections with 95% confidence prediction intervals across customized business horizons.")

    if st.session_state.cleaned_df is None or st.session_state.model_store is None:
        st.warning("Please clean data and train models in preceding tabs before generating forecasts.")
    else:
        clean_df = st.session_state.cleaned_df
        model_store = st.session_state.model_store
        best_name = st.session_state.best_model_name

        # Forecast Controls
        fc_c1, fc_c2, fc_c3, fc_c4 = st.columns(4)
        with fc_c1:
            forecast_horizon = st.selectbox("Forecasting Horizon", [7, 14, 30, 60, 90], index=2, format_func=lambda x: f"{x} Days")
        with fc_c2:
            fc_prod = st.selectbox("Product", ["All Products"] + sorted(clean_df["product"].unique().tolist()), key="fc_prod")
        with fc_c3:
            fc_reg = st.selectbox("Region", ["All Regions"] + sorted(clean_df["region"].unique().tolist()), key="fc_reg")
        with fc_c4:
            avail_models = list(model_store.keys())
            def_idx = avail_models.index(best_name) if best_name in avail_models else 0
            fc_model = st.selectbox("Forecasting Model", avail_models, index=def_idx)

        # Generate Forecast Action
        if st.button("✨ Generate Future Forecast", type="primary", use_container_width=True):
            prod_arg = "All" if fc_prod == "All Products" else fc_prod
            reg_arg = "All" if fc_reg == "All Regions" else fc_reg

            hist_series = prepare_daily_series(clean_df, product=prod_arg, region=reg_arg)
            f_df = generate_future_forecast(
                model_store=model_store,
                model_name=fc_model,
                historical_daily_df=hist_series,
                horizon_days=forecast_horizon
            )
            st.session_state.forecast_df = f_df

            # Log to SQLite
            log_forecast(
                forecast_df=f_df,
                dataset_name=st.session_state.dataset_filename or "active_data.csv",
                model_name=fc_model,
                product=fc_prod,
                region=fc_reg,
                horizon_days=forecast_horizon
            )
            st.success(f"✅ Generated {forecast_horizon}-day forecast using {fc_model}!")

        # Display Forecast if available
        if st.session_state.forecast_df is not None:
            f_df = st.session_state.forecast_df

            # KPI Summary of the Future Forecast
            total_proj_rev = f_df["forecast"].sum()
            avg_daily_proj = f_df["forecast"].mean()
            peak_val = f_df["forecast"].max()
            peak_date = f_df.loc[f_df["forecast"].idxmax(), "date"]

            f_kpi1, f_kpi2, f_kpi3, f_kpi4 = st.columns(4)
            f_kpi1.metric("Projected Total Revenue", f"${total_proj_rev:,.2f}")
            f_kpi2.metric("Avg Daily Revenue", f"${avg_daily_proj:,.2f}")
            f_kpi3.metric("Peak Projected Day", peak_date)
            f_kpi4.metric("Peak Day Revenue", f"${peak_val:,.2f}")

            st.markdown("---")

            # Combined Historical + Future Forecast Chart
            st.markdown("#### 📈 Historical Actuals vs. Future Forecast Curve")

            # Slice last 90 days of history for clear visualization context
            prod_arg = "All" if fc_prod == "All Products" else fc_prod
            reg_arg = "All" if fc_reg == "All Regions" else fc_reg
            recent_hist = prepare_daily_series(clean_df, product=prod_arg, region=reg_arg).tail(90)

            fig_fc = go.Figure()

            # 1. Historical Actuals (Blue line)
            fig_fc.add_trace(go.Scatter(
                x=recent_hist["date"],
                y=recent_hist["sales"],
                mode="lines",
                name="Historical Actual Sales",
                line=dict(color="#1E3A8A", width=2)
            ))

            # 2. Upper Prediction Bound (transparent for fill)
            fig_fc.add_trace(go.Scatter(
                x=f_df["date"],
                y=f_df["upper_bound"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="Upper Bound (95% CI)"
            ))

            # 3. Lower Prediction Bound with fill
            fig_fc.add_trace(go.Scatter(
                x=f_df["date"],
                y=f_df["lower_bound"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.15)",
                name="95% Confidence Interval"
            ))

            # 4. Projected Forecast (Green line)
            fig_fc.add_trace(go.Scatter(
                x=f_df["date"],
                y=f_df["forecast"],
                mode="lines+markers",
                name=f"Forecast ({f_df['model_used'].iloc[0]})",
                line=dict(color="#10B981", width=3, dash="dash"),
                marker=dict(size=5, color="#10B981")
            ))

            # Vertical separator line between history and forecast
            last_hist_date = str(recent_hist["date"].max().strftime("%Y-%m-%d"))
            fig_fc.add_vline(x=last_hist_date, line_dash="dot", line_color="#EF4444", annotation_text="Forecast Horizon Start")

            fig_fc.update_layout(
                title=f"Sales Forecast: Next {len(f_df)} Days ({fc_prod} | {fc_reg})",
                xaxis_title="Date",
                yaxis_title="Sales Revenue ($)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_fc, use_container_width=True)

            # Forecast Data Table
            st.markdown("#### 📋 Detailed Daily Forecast Schedule")
            st.dataframe(f_df, use_container_width=True)


# ==============================================================================
# TAB 7: REPORTS & DATABASE
# ==============================================================================
with tabs[6]:
    st.markdown("### 📑 Reports Export & SQLite Database Archive")
    st.write("Export clean forecast deliverables in CSV and Excel formats, generate executive Markdown briefings, and audit the local SQLite database.")

    rep_col1, rep_col2 = st.columns(2)

    with rep_col1:
        st.markdown("#### 📥 Download Deliverables")
        if st.session_state.forecast_df is not None:
            f_df = st.session_state.forecast_df

            # CSV Download
            csv_bytes = f_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download Forecast as CSV",
                data=csv_bytes,
                file_name=f"sales_forecast_{len(f_df)}days.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Excel Download
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                f_df.to_excel(writer, index=False, sheet_name="Forecast")
                if st.session_state.comparison_df is not None:
                    st.session_state.comparison_df.to_excel(writer, index=False, sheet_name="Model Metrics")
            st.download_button(
                label="📊 Download Forecast as Excel (.xlsx)",
                data=excel_buffer.getvalue(),
                file_name=f"sales_forecast_{len(f_df)}days.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.info("Generate a forecast in Tab 6 to enable CSV/Excel downloads.")

        if st.session_state.comparison_df is not None:
            metrics_csv = st.session_state.comparison_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📈 Download Model Comparison Metrics (CSV)",
                data=metrics_csv,
                file_name="model_evaluation_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )

    with rep_col2:
        st.markdown("#### 📝 Executive Forecasting Briefing")
        # Generate markdown report text
        best_name = st.session_state.best_model_name or "N/A"
        date_today = datetime.now().strftime("%Y-%m-%d")
        f_len = len(st.session_state.forecast_df) if st.session_state.forecast_df is not None else 0
        tot_rev = f"${st.session_state.forecast_df['forecast'].sum():,.2f}" if st.session_state.forecast_df is not None else "$0.00"

        report_md = f"""# AI Sales Forecasting Project - Executive Briefing
**Author:** Lokesh (BSc Data Science Final Year Project)  
**Date Generated:** {date_today}  
**Active Model:** {best_name}  

## 1. Executive Summary
- **Forecasting Horizon:** {f_len} Days
- **Total Projected Sales Revenue:** {tot_rev}
- **Primary Optimization Model:** {best_name} (Selected via minimum chronological RMSE & MAE)

## 2. Methodology & Compliance
- **Zero Lookahead Leakage:** Features (lags 1, 7, 14, 30 and rolling statistics) were strictly shifted ($t-1$) before evaluation.
- **Chronological Split:** Preserved temporal integrity by evaluating on unseen chronological holdout periods.
- **Safe Metric Calculation:** Zero values in actuals handled via safe MAPE and WAPE calculation.

## 3. Operational Recommendations
- **Inventory Safety Stock:** Maintain buffer stock equivalent to upper prediction interval for critical SKUs.
- **Procurement Timing:** Procure additional raw materials 14 days before anticipated weekend/seasonal surges.
- **Revenue Targets:** Use median forecast bounds as realistic baseline targets for sales quotas.
"""
        st.download_button(
            label="📄 Download Executive Briefing (.md)",
            data=report_md.encode("utf-8"),
            file_name="executive_sales_forecast_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### 🗄️ SQLite Database Records (Audit Trail)")

    db_tabs = st.tabs(["Dataset Upload Logs", "Model Evaluation Metrics History", "Saved Forecast Records"])

    with db_tabs[0]:
        ds_history = get_dataset_history(limit=15)
        if not ds_history.empty:
            st.dataframe(ds_history, use_container_width=True)
        else:
            st.info("No datasets logged yet.")

    with db_tabs[1]:
        m_history = get_saved_metrics(limit=25)
        if not m_history.empty:
            st.dataframe(m_history, use_container_width=True)
        else:
            st.info("No metrics logged yet.")

    with db_tabs[2]:
        f_history = get_saved_forecasts(limit=50)
        if not f_history.empty:
            st.dataframe(f_history, use_container_width=True)
        else:
            st.info("No forecast runs logged yet.")


# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px;">
    <strong>AI Sales Forecasting</strong> | Final Year BSc Data Science Project | Student: <strong>Lokesh</strong><br/>
    Technologies: Python 3.13 • Streamlit • Scikit-Learn • Statsmodels (ARIMA) • Plotly • SQLite • Pandas
</div>
""", unsafe_allow_html=True)
