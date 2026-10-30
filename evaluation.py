"""
AirLens: Model Evaluation and Forecast Error Analysis Module
Computes standard regression metrics (MAE, RMSE, R², sMAPE) and granular error slices
(by station, hour-of-day, season, and forecast horizon).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes MAE, RMSE, R², and symmetric MAPE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "sMAPE": 0.0}

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))
    
    # Symmetric MAPE avoids explosion when actual values are close to 0
    denom = (np.abs(y_t) + np.abs(y_p)) / 2.0 + 1e-5
    smape = float(np.mean(np.abs(y_p - y_t) / denom) * 100.0)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "sMAPE": round(smape, 2)
    }


def analyze_errors_by_slice(df_eval: pd.DataFrame, true_col="y_true", pred_col="y_pred") -> dict:
    """
    Slices forecasting errors across multiple analytical dimensions:
    - By Monitoring Station
    - By Hour of Day (diurnal rush hour vs night)
    - By Season (Winter heating vs Summer)
    """
    df = df_eval.copy()
    df['abs_error'] = np.abs(df[true_col] - df[pred_col])
    df['signed_residual'] = df[pred_col] - df[true_col]

    # 1. Error by Station
    stn_err = df.groupby('station')['abs_error'].agg(['mean', 'median', 'std']).reset_index()
    stn_err.columns = ['station', 'MAE', 'Median_AE', 'Std_AE']
    stn_err = stn_err.sort_values(by='MAE')

    # 2. Error by Hour of Day
    hour_err = df.groupby('hour')['abs_error'].mean().reset_index()
    hour_err.columns = ['hour', 'MAE']

    # 3. Error by Season
    season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
    if 'season' in df.columns:
        df['season_name'] = df['season'].map(season_map)
        season_err = df.groupby('season_name')['abs_error'].mean().reset_index()
        season_err.columns = ['season', 'MAE']
    else:
        season_err = pd.DataFrame()

    return {
        "by_station": stn_err,
        "by_hour": hour_err,
        "by_season": season_err
    }
