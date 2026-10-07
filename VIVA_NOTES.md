# AI Sales Forecasting - Viva Preparation Notes

**Student Name:** Lokesh  
**Degree:** BSc Data Science (Final Year)  
**Project Title:** AI Sales Forecasting: Intelligent Sales Prediction & Business Planning  

---

## Quick Viva Introduction (30-Second Elevator Pitch)
> *"Respected Examiners, my final-year project is an end-to-end AI Sales Forecasting application. It takes raw historical retail transaction data, cleans and standardizes it, creates leak-free temporal and lag features, and compares five forecasting algorithms—from moving-average baselines and ARIMA to machine learning ensembles like Random Forest and Gradient Boosting. We evaluate models using chronological holdout testing to strictly prevent future-data leakage, assess performance with MAE, RMSE, and zero-safe MAPE/WAPE, and generate future sales projections with confidence intervals to help managers with inventory procurement, revenue targeting, and supply chain planning."*

---

## Core Questions and Viva Answers

### 1. What is Sales Forecasting?
**Answer:**  
Sales forecasting is the process of estimating future sales revenue or units sold over a specific upcoming time horizon (such as the next 7, 30, or 90 days) using historical sales patterns, promotional calendars, and demand trends. It is essential for inventory optimization, working capital management, and procurement planning.

---

### 2. Why is AI/ML used instead of traditional manual estimation?
**Answer:**  
Traditional estimation relies on simple moving averages, static percentages, or managerial intuition. These methods fail when dealing with non-linear relationships—such as weekend demand spikes, complex calendar seasonality, promotional discounts, and sudden holiday shifts. Machine learning algorithms can automatically learn multivariate, non-linear interactions across thousands of data points, producing consistent, data-backed, and mathematically measurable forecasts.

---

### 3. What is Time-Series Forecasting?
**Answer:**  
Time-series forecasting is the task of predicting future values of a variable based on its past historical observations arranged in sequential, chronological order. Unlike standard cross-sectional machine learning where samples are independent and identically distributed (i.i.d.), time-series data exhibits **temporal dependency**, autocorrelation, and sequential patterns where today's sales depend on yesterday's or last week's sales.

---

### 4. What is Exploratory Data Analysis (EDA)?
**Answer:**  
EDA is the initial process of analyzing, summarizing, and visualizing datasets to understand their underlying distributions, detect outliers or missing values, discover patterns and trends, and verify assumptions before building models. In our project, EDA reveals overall sales trajectories, day-of-week seasonality, product revenue shares, regional contributions, and promotional uplift.

---

### 5. What is Feature Engineering in this project?
**Answer:**  
Feature engineering is the process of transforming raw tabular and date attributes into informative predictive signals for machine learning models:
1. **Calendar Features:** Year, Month, Day, Day of Week, Quarter, and Weekend flag.
2. **Cyclical Encodings:** Sine and Cosine transformations for month ($1-12$) and day of week ($0-6$) so the model recognizes that Sunday (6) and Monday (0), or December (12) and January (1), are temporally adjacent.
3. **Lag Features:** Prior sales values at $t-1$, $t-7$, $t-14$, and $t-30$ to provide autoregressive context.
4. **Rolling Window Statistics:** 7-day, 14-day, and 30-day rolling means, medians, and standard deviations to capture moving velocity and volatility.

---

### 6. Why is a Chronological Train/Test Split used? Why must Random Split be avoided?
**Answer:**  
In time-series modeling, **temporal causality must always be preserved**.
- **Why random split fails:** A standard `train_test_split(shuffle=True)` randomly scrambles rows. If row $t$ is in the test set while row $t+1$ (the future) is placed into the training set, the model trains on future data to predict the past. This creates artificial "future data leakage" and unrealistically high accuracy that completely collapses in real-world deployment.
- **Why chronological split is correct:** We reserve older historical records (e.g. first 85% of days) strictly for training, and evaluate models exclusively on the recent consecutive future period (the last 15% of days). This mirrors real-world production where you predict tomorrow using only what you know up to today.

---

### 7. What is Data Leakage in time series, and how did you prevent it?
**Answer:**  
Data leakage occurs when information from outside the training dataset (specifically from the future or from the test set) is inadvertently used to train the model.  
**How we prevented it:**
1. **Chronological Splitting:** Training data strictly precedes test data.
2. **Lag Shifting:** When calculating rolling statistics (such as 7-day rolling mean), we applied `.shift(1)` first. This ensures the current day's target sales is never included in its own input feature.

---

### 8. What is MAE (Mean Absolute Error)?
**Answer:**  
$$\text{MAE} = \frac{1}{n} \sum |y - \hat{y}|$$
MAE is the average absolute difference between the actual sales and the predicted sales. It expresses error directly in the original business units (e.g., dollars). It treats all errors proportionally and is robust against extreme outliers.

---

### 9. What is RMSE (Root Mean Squared Error)?
**Answer:**  
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$$
RMSE is the square root of the average squared errors. Because errors are squared before averaging, RMSE penalizes large prediction errors much more severely than small ones. In retail, large forecast misses are catastrophic for inventory, making RMSE the primary metric for selecting our champion model.

---

### 10. What is MAPE, and why can it be problematic? How did you handle it?
**Answer:**  
$$\text{MAPE} = \frac{100\%}{n} \sum \left| \frac{y - \hat{y}}{y} \right|$$
MAPE measures percentage error relative to actual sales.
- **The Problem:** If actual sales $y = 0$ on any given day, the formula divides by zero, causing a division error or infinite output. Furthermore, when $y$ is very small, even a tiny error produces an inflated, misleading percentage.
- **Our Solution:** The application inspects actuals for zeros. If zeros exist, it computes safe MAPE on non-zero observations with an explicit warning note and computes **WAPE (Weighted Absolute Percentage Error)**:
$$\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$$
WAPE is zero-safe and industry-standard for retail demand.

---

### 11. What is the difference between Linear Regression and Random Forest?
**Answer:**  
| Aspect | Linear Regression | Random Forest Regressor |
|---|---|---|
| **Model Type** | Parametric, linear | Non-parametric, ensemble of decision trees |
| **Relationship** | Assumes linear relationship ($y = \beta X + \epsilon$) | Captures complex, non-linear interactions |
| **Outlier Sensitivity** | Sensitive to outliers | Robust against outliers |
| **Feature Interactions** | Must be manually engineered | Automatically captures feature interactions |
| **Interpretability** | Direct coefficients | Feature importance scores |

---

### 12. What is ARIMA?
**Answer:**  
ARIMA stands for **AutoRegressive Integrated Moving Average**. It is a classical econometric time-series technique parameterized by $(p, d, q)$:
- **$p$ (AutoRegressive):** Number of past lagged observations used to predict the current value.
- **$d$ (Integrated):** Number of differencing operations required to make a non-stationary time series stationary (removing trends).
- **$q$ (Moving Average):** Number of lagged forecast error terms included in the regression.

---

### 13. What is Seasonality vs. Trend?
**Answer:**  
- **Trend:** The long-term general direction in which sales are moving over an extended period (upward growth or downward contraction). In our data, there is a steady +12% annual upward trend.
- **Seasonality:** Repeating, predictable fluctuations that occur at fixed, known intervals (e.g. higher retail footfall every Saturday and Sunday, or annual festive shopping surges every November and December).

---

### 14. Why was the final model selected?
**Answer:**  
Model selection is based on objective empirical performance on the chronological holdout test set, not simply theoretical complexity. In our tournament:
- **Random Forest** achieved the lowest test RMSE ($50,711.38) and lowest test MAE ($34,909.02), with a test MAPE of $12.82\%$ and $R^2$ of $0.4692$.
- It outperformed Linear Regression ($51,931$ RMSE) and drastically outperformed the Moving Average Baseline ($84,407$ RMSE) and ARIMA ($88,512$ RMSE) because Random Forest captures non-linear interactions between calendar indicators, day-of-week seasonality, and past sales velocity.

---

### 15. How does Multi-Step Future Forecasting work?
**Answer:**  
To predict 30 days into the future where tomorrow's actual sales are unknown:
1. We compute calendar features for day 1 into the future.
2. We compute lag and rolling features from the end of the known history.
3. The model predicts day 1.
4. We append day 1's prediction back into our historical series.
5. For day 2, lag 1 becomes day 1's forecasted sales.
6. This recursive process repeats for all $H$ horizon days.
7. We calculate prediction intervals ($1.96 \times \text{RMSE}$) with a compounding uncertainty multiplier to communicate forecast risk to decision-makers.

---

### 16. What are the limitations of the current system?
**Answer:**  
1. **External Macro Shocks:** Pure statistical models cannot anticipate unexpected black swan disruptions (e.g. sudden natural disasters or pandemic closures) without external event regressors.
2. **Cold-Start SKUs:** Brand new products with fewer than 14 days of history lack sufficient lag signals for time-series modeling.
3. **Static Pricing Assumption:** The recursive forecast assumes future prices remain consistent with recent historical averages.

---

### 17. What is the future scope of this project?
**Answer:**  
1. **Hierarchical Forecasting:** Implement reconciliation algorithms (such as MinT) across store, regional, and national levels.
2. **Deep Learning Models:** Integrate Temporal Fusion Transformers (TFT) or DeepAR for probabilistic multi-horizon forecasting.
3. **Automated ERP Integration:** Direct API integration with inventory management systems to automatically generate supplier purchase orders when forecasted stock hits safety reorder thresholds.
