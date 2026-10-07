"""
Model Evaluation Module for AI Sales Forecasting
Computes MAE, RMSE, safe MAPE/WAPE, and R2 metrics, compares forecasting models,
and generates evaluation diagnostics.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes regression & time-series evaluation metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - Safe MAPE (Mean Absolute Percentage Error, handling zero/near-zero actuals)
    - WAPE (Weighted Absolute Percentage Error - robust against zeros)
    - R-squared (Coefficient of Determination)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wape": 0.0, "r2": 0.0, "mape_note": "No observations"}

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Safe MAPE handling
    zero_mask = np.abs(y_true) < 1e-4
    has_zeros = bool(np.any(zero_mask))
    sum_true = float(np.sum(np.abs(y_true)))

    # WAPE (Weighted Absolute Percentage Error): sum(|y - y_hat|) / sum(y) * 100
    wape = float((np.sum(np.abs(y_true - y_pred)) / sum_true) * 100.0) if sum_true > 1e-6 else 0.0

    if has_zeros:
        non_zero_true = y_true[~zero_mask]
        non_zero_pred = y_pred[~zero_mask]
        if len(non_zero_true) > 0:
            mape = float(np.mean(np.abs((non_zero_true - non_zero_pred) / non_zero_true)) * 100.0)
            mape_note = f"Computed on {len(non_zero_true)}/{len(y_true)} non-zero observations to prevent division by zero."
        else:
            mape = 0.0
            mape_note = "All actual values are 0. MAPE undefined."
    else:
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
        mape_note = "Standard MAPE (no zero values in actuals)."

    # R-squared
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "wape": round(wape, 2),
        "r2": round(r2, 4),
        "has_zeros": has_zeros,
        "mape_note": mape_note
    }


def compare_models(model_results: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, str]:
    """
    Takes a dictionary of model results where each entry contains 'y_true' and 'y_pred'.
    Computes comparative metrics, sorts by lowest RMSE, and returns (comparison_df, best_model_name).
    """
    records = []

    for name, res in model_results.items():
        y_true = res.get("y_true")
        y_pred = res.get("y_pred")
        metrics = calculate_metrics(y_true, y_pred)
        records.append({
            "Model": name,
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "MAPE (%)": metrics["mape"],
            "WAPE (%)": metrics["wape"],
            "R² Score": metrics["r2"],
            "MAPE Status": "Safe (non-zero)" if metrics["has_zeros"] else "Standard"
        })

    df_comp = pd.DataFrame(records)
    if df_comp.empty:
        return df_comp, "None"

    # Sort primarily by lowest RMSE, secondarily by lowest MAE
    df_comp = df_comp.sort_values(by=["RMSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
    best_model = df_comp.iloc[0]["Model"]

    return df_comp, best_model


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes residual diagnostics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    return {
        "residuals": residuals,
        "mean_residual": round(float(np.mean(residuals)), 2),
        "std_residual": round(float(np.std(residuals)), 2),
        "min_residual": round(float(np.min(residuals)), 2),
        "max_residual": round(float(np.max(residuals)), 2),
    }


def create_actual_vs_predicted_figure(
    dates: pd.Series,
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    best_model: Optional[str] = None
) -> go.Figure:
    """Generates an interactive Plotly figure comparing actual test data vs model predictions."""
    fig = go.Figure()

    # Actual test line
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=4)
    ))

    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    for idx, (m_name, m_pred) in enumerate(predictions_dict.items()):
        is_best = (m_name == best_model)
        color = colors[idx % len(colors)]
        dash_style = "solid" if is_best else "dot"
        width = 2.5 if is_best else 1.5
        label = f"{m_name} (Best)" if is_best else m_name

        fig.add_trace(go.Scatter(
            x=dates,
            y=m_pred,
            mode="lines",
            name=label,
            line=dict(color=color, width=width, dash=dash_style)
        ))

    fig.update_layout(
        title="Test Period: Actual vs. Predicted Sales Comparison",
        xaxis_title="Date",
        yaxis_title="Sales Revenue ($)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_residuals_figure(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    """Creates a residual plot over time and a zero-error reference line."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=residuals,
        mode="lines+markers",
        name="Residual (Actual - Predicted)",
        line=dict(color="#d62728", width=1.5),
        marker=dict(size=4)
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Zero Error Line")

    fig.update_layout(
        title=f"Residual Error Over Time ({model_name})",
        xaxis_title="Date",
        yaxis_title="Residual (Sales Error)",
        template="plotly_white"
    )
    return fig
