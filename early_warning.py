"""
AirLens: Anomaly-Aware Early-Warning System
Generates structured health risk warnings and anomaly alerts by bridging
future multi-horizon predictions with AQI health thresholds and anomaly flags.
"""

import pandas as pd
import numpy as np
from src.config import AQI_BREAKPOINTS


def get_aqi_category_info(pm25_val: float) -> dict:
    """Returns the AQI category, color, severity score, and health recommendation."""
    for b in AQI_BREAKPOINTS:
        if b["min"] <= pm25_val < b["max"]:
            return b
    return AQI_BREAKPOINTS[-1]  # Hazardous fallback for > 1000 µg/m³


def generate_early_warnings(df_with_anomalies: pd.DataFrame,
                             forecast_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Constructs a unified table of early warnings and critical events.
    Combines:
    - Observed severe pollution and anomalies
    - Upcoming predicted threshold exceedances (future outlook)
    """
    warnings = []

    # 1. Historical & Current Monitored Warnings
    recent_records = df_with_anomalies.tail(500)  # Focus on recent timeline
    for _, row in recent_records.iterrows():
        pm25 = float(row.get('PM2.5', 0.0))
        cat_info = get_aqi_category_info(pm25)
        is_anom = bool(row.get('is_anomaly', False))
        
        # Trigger alert if severity >= 3 (Unhealthy for Sensitive) OR anomalous
        if cat_info['severity'] >= 3 or is_anom:
            if is_anom and cat_info['severity'] >= 3:
                warn_type = "Compound Warning (Severe Exceedance + Anomaly)"
                status = "Critical Active"
            elif is_anom:
                warn_type = "Diagnostic Anomaly Alert"
                status = "Under Review"
            else:
                warn_type = "Health Exceedance Alert"
                status = "Active"

            warnings.append({
                "timestamp": str(row['datetime']),
                "station": row['station'],
                "pollutant": "PM2.5",
                "concentration": round(pm25, 1),
                "data_source": "Observed Sensor Reading",
                "category": cat_info['category'],
                "severity_score": cat_info['severity'],
                "color": cat_info['color'],
                "warning_type": warn_type,
                "detection_method": row.get('detection_method', 'Threshold Exceedance'),
                "precaution": cat_info['action'],
                "status": status
            })

    # 2. Upcoming Predicted Forecast Warnings
    if forecast_df is not None and not forecast_df.empty:
        for _, f_row in forecast_df.iterrows():
            pred_pm25 = float(f_row['predicted_pm25'])
            cat_info = get_aqi_category_info(pred_pm25)
            
            if cat_info['severity'] >= 3:
                warnings.append({
                    "timestamp": str(f_row['target_datetime']),
                    "station": f_row['station'],
                    "pollutant": "PM2.5",
                    "concentration": round(pred_pm25, 1),
                    "data_source": f"Predicted ({f_row['horizon_hours']}h Horizon)",
                    "category": cat_info['category'],
                    "severity_score": cat_info['severity'],
                    "color": cat_info['color'],
                    "warning_type": "Upcoming Forecast Exceedance",
                    "detection_method": f"Transformer/LGBM Model (+{f_row['horizon_hours']}h Outlook)",
                    "precaution": f"Anticipated {cat_info['category']}: {cat_info['action']}",
                    "status": "Upcoming Forecast"
                })

    warn_df = pd.DataFrame(warnings)
    if not warn_df.empty:
        warn_df = warn_df.sort_values(by=["severity_score", "timestamp"], ascending=[False, False])
    return warn_df
