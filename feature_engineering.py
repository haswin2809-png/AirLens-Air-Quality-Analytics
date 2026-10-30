"""
AirLens: Feature Engineering Module
Transforms cleaned hourly station data into predictive multivariate features:
- Cyclical temporal encodings (sin/cos of hour, month, day of week)
- Meteorological vector projections (Wind U and V components)
- Multi-horizon lag features (t-1, t-2, t-3, t-6, t-12, t-24)
- Rolling statistical features (rolling mean, rolling std, rate of change)
- Station numerical indexing & one-hot encoding
- Multi-horizon forecast targets (t+1, t+6, t+12, t+24, t+48)
"""

import numpy as np
import pandas as pd
from src.config import TARGET_COL, POLLUTANT_COLS, METEOROLOGICAL_COLS, FORECAST_HORIZONS

WIND_DEG_MAP = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
    'cv': 0, 'CV': 0
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs time-series features strictly per station to prevent inter-station leakage.
    """
    station_dfs = []

    # Map station name to unique integer index for embeddings
    station_names = sorted(df['station'].unique().tolist())
    station_to_id = {stn: idx for idx, stn in enumerate(station_names)}

    for stn_name in station_names:
        sub = df[df['station'] == stn_name].sort_values('datetime').copy()
        sub['station_id'] = station_to_id[stn_name]

        # 1. Cyclical Time Features
        sub['sin_hour'] = np.sin(2 * np.pi * sub['hour'] / 24.0)
        sub['cos_hour'] = np.cos(2 * np.pi * sub['hour'] / 24.0)
        sub['sin_month'] = np.sin(2 * np.pi * sub['month'] / 12.0)
        sub['cos_month'] = np.cos(2 * np.pi * sub['month'] / 12.0)
        sub['sin_dow'] = np.sin(2 * np.pi * sub['dayofweek'] / 7.0)
        sub['cos_dow'] = np.cos(2 * np.pi * sub['dayofweek'] / 7.0)

        # 2. Atmospheric Wind Vectors (U = East-West, V = North-South)
        deg = sub['wd'].map(WIND_DEG_MAP).fillna(0)
        rad = np.radians(deg)
        sub['wind_u'] = -sub['WSPM'] * np.sin(rad)
        sub['wind_v'] = -sub['WSPM'] * np.cos(rad)

        # 3. Lagged Target Features (PM2.5)
        for lag in [1, 2, 3, 6, 12, 24]:
            sub[f'PM2.5_lag_{lag}'] = sub['PM2.5'].shift(lag)

        # 4. Cross-Pollutant Lags
        for p in ['PM10', 'SO2', 'NO2', 'CO', 'O3']:
            sub[f'{p}_lag_1'] = sub[p].shift(1)
            sub[f'{p}_lag_3'] = sub[p].shift(3)

        # 5. Meteorological Lags
        for m in ['TEMP', 'PRES', 'DEWP', 'WSPM', 'RAIN']:
            sub[f'{m}_lag_1'] = sub[m].shift(1)

        # 6. Rolling Window Statistics
        sub['PM2.5_roll_mean_6h'] = sub['PM2.5'].shift(1).rolling(window=6, min_periods=1).mean()
        sub['PM2.5_roll_std_6h'] = sub['PM2.5'].shift(1).rolling(window=6, min_periods=1).std().fillna(0)
        sub['PM2.5_roll_mean_24h'] = sub['PM2.5'].shift(1).rolling(window=24, min_periods=1).mean()
        sub['PM2.5_roll_std_24h'] = sub['PM2.5'].shift(1).rolling(window=24, min_periods=1).std().fillna(0)

        # 7. Short-term Rate of Change (Velocity)
        sub['PM2.5_diff_1h'] = sub['PM2.5'].shift(1) - sub['PM2.5'].shift(2)

        # 8. Multi-Horizon Forecast Targets
        for h in FORECAST_HORIZONS:
            sub[f'target_h{h}'] = sub['PM2.5'].shift(-h)

        station_dfs.append(sub)

    full_featured = pd.concat(station_dfs, ignore_index=True)
    
    # Drop edge rows where essential 24-hr lags or target_h1 are NaN
    valid_mask = full_featured['PM2.5_lag_24'].notna() & full_featured['target_h1'].notna()
    return full_featured[valid_mask].reset_index(drop=True)


def get_feature_column_names() -> list[str]:
    """
    Returns the standard ordered list of predictor feature names.
    """
    base_features = [
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_dow', 'cos_dow',
        'is_weekend', 'season', 'wind_u', 'wind_v',
        'TEMP_lag_1', 'PRES_lag_1', 'DEWP_lag_1', 'WSPM_lag_1', 'RAIN_lag_1',
        'PM2.5_lag_1', 'PM2.5_lag_2', 'PM2.5_lag_3', 'PM2.5_lag_6', 'PM2.5_lag_12', 'PM2.5_lag_24',
        'PM10_lag_1', 'PM10_lag_3', 'SO2_lag_1', 'SO2_lag_3', 'NO2_lag_1', 'NO2_lag_3',
        'CO_lag_1', 'CO_lag_3', 'O3_lag_1', 'O3_lag_3',
        'PM2.5_roll_mean_6h', 'PM2.5_roll_std_6h', 'PM2.5_roll_mean_24h', 'PM2.5_roll_std_24h',
        'PM2.5_diff_1h', 'station_id'
    ]
    return base_features
