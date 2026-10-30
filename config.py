"""
AirLens: Configuration and Global Constants
Contains station coordinates, AQI thresholds, feature definitions, and model parameters.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "PRSA_Data_20130301-20170228"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

for d in [ARTIFACTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# 12 Beijing Monitoring Stations & Verified Geographic Coordinates
# Source: Beijing Municipal Environmental Monitoring Center / Zhang et al. (2017)
STATIONS_METADATA = {
    "Aotizhongxin": {"name": "Aotizhongxin (Olympic Sports)", "lat": 39.982, "lon": 116.397, "type": "Urban"},
    "Changping": {"name": "Changping Town", "lat": 40.217, "lon": 116.230, "type": "Suburban"},
    "Dingling": {"name": "Dingling (Ming Tombs)", "lat": 40.292, "lon": 116.220, "type": "Regional Background"},
    "Dongsi": {"name": "Dongsi (East District)", "lat": 39.929, "lon": 116.417, "type": "Urban Core"},
    "Guanyuan": {"name": "Guanyuan (Financial District)", "lat": 39.929, "lon": 116.339, "type": "Urban Core"},
    "Gucheng": {"name": "Gucheng (Industrial/West)", "lat": 39.914, "lon": 116.184, "type": "Urban/Industrial"},
    "Huairou": {"name": "Huairou Town", "lat": 40.328, "lon": 116.628, "type": "Northeastern Suburban"},
    "Nongzhanguan": {"name": "Nongzhanguan (Agricultural Exhibit)", "lat": 39.937, "lon": 116.461, "type": "Urban Core"},
    "Shunyi": {"name": "Shunyi Town", "lat": 40.127, "lon": 116.655, "type": "Northeastern Suburban"},
    "Tiantan": {"name": "Tiantan (Temple of Heaven)", "lat": 39.886, "lon": 116.407, "type": "Urban Core"},
    "Wanliu": {"name": "Wanliu (Haidian Academic)", "lat": 39.987, "lon": 116.287, "type": "Urban"},
    "Wanshouxigong": {"name": "Wanshouxigong (South District)", "lat": 39.878, "lon": 116.352, "type": "Urban Core"},
}

# Target and Predictor Features
TARGET_COL = "PM2.5"

POLLUTANT_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEOROLOGICAL_COLS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
CATEGORICAL_COLS = ["wd", "station"]

# China MEP & US EPA AQI Standard PM2.5 Breakpoints (in µg/m³)
# Used for early warnings, color encoding, and risk categorization
AQI_BREAKPOINTS = [
    {"category": "Good", "min": 0.0, "max": 35.0, "color": "#00E400", "severity": 1, "action": "Air quality is satisfactory; enjoy normal outdoor activities."},
    {"category": "Moderate", "min": 35.0, "max": 75.0, "color": "#FFFF00", "severity": 2, "action": "Acceptable air quality. Sensitive individuals should consider reducing prolonged heavy exertion."},
    {"category": "Unhealthy for Sensitive", "min": 75.0, "max": 115.0, "color": "#FF7E00", "severity": 3, "action": "Children, elderly, and people with respiratory diseases should limit prolonged outdoor exertion."},
    {"category": "Unhealthy", "min": 115.0, "max": 150.0, "color": "#FF0000", "severity": 4, "action": "General public should reduce outdoor exertion; sensitive groups should avoid outdoor activities."},
    {"category": "Very Unhealthy", "min": 150.0, "max": 250.0, "color": "#99004C", "severity": 5, "action": "Health alert: everyone should avoid outdoor physical exertion. Keep windows closed and run purifiers."},
    {"category": "Hazardous", "min": 250.0, "max": 1000.0, "color": "#7E0023", "severity": 6, "action": "Emergency condition: stay indoors with air purification. Significant aggravation of heart or lung disease."}
]

# Forecast Horizons (Hours)
FORECAST_HORIZONS = [1, 6, 12, 24, 48]

# Anomaly Detection Settings
ANOMALY_CONFIG = {
    "zscore_threshold": 3.0,
    "isolation_forest_contamination": 0.03,
    "residual_threshold_factor": 2.5,
    "rolling_window_hours": 24
}

# Prediction Interval Confidence (Uncertainty Estimation)
CONFIDENCE_LEVEL = 0.90  # 90% prediction interval
