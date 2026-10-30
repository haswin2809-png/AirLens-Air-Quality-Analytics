"""
AirLens: Anomaly Detection Engine
Implements multi-method anomaly detection:
1. Local Rolling Z-Score (Sudden Spikes relative to 24h diurnal baseline)
2. Multivariate Isolation Forest (Unusual multi-pollutant ratio inversions)
3. Forecast Residual Divergence (Significant deviations between observed and predicted values)
4. Composite Anomaly Scoring and Categorization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.config import ANOMALY_CONFIG, POLLUTANT_COLS


class AirLensAnomalyDetector:
    def __init__(self, z_thresh=ANOMALY_CONFIG["zscore_threshold"],
                 contamination=ANOMALY_CONFIG["isolation_forest_contamination"]):
        self.z_thresh = z_thresh
        self.contamination = contamination
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.fitted = False

    def fit_isolation_forest(self, train_df: pd.DataFrame):
        """Fits Isolation Forest on multi-pollutant vector."""
        pollutant_data = train_df[POLLUTANT_COLS].dropna()
        self.iso_forest.fit(pollutant_data)
        self.fitted = True

    def detect_anomalies(self, df: pd.DataFrame, preds: np.ndarray = None) -> pd.DataFrame:
        """
        Runs comprehensive anomaly detection across the dataset.
        Returns a DataFrame of flagged anomalies with full metadata.
        """
        results_df = df.copy()

        # 1. Rolling Z-Score (Station-aware 24h baseline)
        results_df['zscore'] = 0.0
        for stn in results_df['station'].unique():
            mask = results_df['station'] == stn
            roll_mean = results_df.loc[mask, 'PM2.5'].rolling(24, min_periods=6).mean()
            roll_std = results_df.loc[mask, 'PM2.5'].rolling(24, min_periods=6).std().replace(0, 1.0)
            z = (results_df.loc[mask, 'PM2.5'] - roll_mean) / (roll_std + 1e-4)
            results_df.loc[mask, 'zscore'] = z.fillna(0.0)

        results_df['is_zscore_anomaly'] = results_df['zscore'].abs() > self.z_thresh

        # 2. Multivariate Isolation Forest
        if not self.fitted:
            self.fit_isolation_forest(results_df)

        iso_features = results_df[POLLUTANT_COLS].fillna(0)
        iso_preds = self.iso_forest.predict(iso_features)
        iso_scores = -self.iso_forest.score_samples(iso_features)
        
        results_df['iso_score'] = iso_scores
        results_df['is_iso_anomaly'] = iso_preds == -1

        # 3. Forecast Residual Anomaly (if predictions provided)
        results_df['is_residual_anomaly'] = False
        results_df['residual'] = 0.0
        if preds is not None and len(preds) == len(results_df):
            residuals = np.abs(results_df['PM2.5'].values - preds)
            results_df['residual'] = residuals
            res_thresh = np.mean(residuals) + (ANOMALY_CONFIG["residual_threshold_factor"] * np.std(residuals))
            results_df['is_residual_anomaly'] = residuals > res_thresh

        # 4. Composite Scoring & Primary Method Tagging
        # We compute an anomaly score on a 0 - 100 scale
        norm_z = np.clip(np.abs(results_df['zscore']) / 4.0, 0, 1) * 40
        norm_iso = np.clip((results_df['iso_score'] - 0.5) / 0.3, 0, 1) * 35
        norm_res = np.clip(results_df['residual'] / 80.0, 0, 1) * 25
        
        results_df['composite_anomaly_score'] = np.round(norm_z + norm_iso + norm_res, 1)
        
        # Determine primary method label
        def label_anomaly_method(row):
            methods = []
            if row['is_zscore_anomaly']:
                methods.append("Rolling Z-Score (Sudden Spike)")
            if row['is_iso_anomaly']:
                methods.append("Isolation Forest (Multi-Pollutant Inversion)")
            if row['is_residual_anomaly']:
                methods.append("Residual Divergence (Forecast Deviation)")
            
            if not methods:
                return "Normal Baseline"
            return " + ".join(methods)

        results_df['detection_method'] = results_df.apply(label_anomaly_method, axis=1)
        results_df['is_anomaly'] = (
            results_df['is_zscore_anomaly'] | 
            results_df['is_iso_anomaly'] | 
            results_df['is_residual_anomaly']
        )

        return results_df
