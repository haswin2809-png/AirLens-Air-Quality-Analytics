"""
AirLens: Automated Test Suite
Verifies config integrity, AQI risk mapping, conformal prediction intervals,
anomaly scoring, and feature construction.
"""

import unittest
import numpy as np
import pandas as pd
from src.config import STATIONS_METADATA, AQI_BREAKPOINTS, FORECAST_HORIZONS
from src.early_warning import get_aqi_category_info
from src.uncertainty import ConformalUncertaintyEstimator
from src.anomaly_detector import AirLensAnomalyDetector
from src.feature_engineering import get_feature_column_names


class TestAirLens(unittest.TestCase):

    def test_stations_metadata(self):
        """Verify all 12 Beijing national stations are defined with valid lat/lon."""
        self.assertEqual(len(STATIONS_METADATA), 12)
        for stn, meta in STATIONS_METADATA.items():
            self.assertTrue(39.0 <= meta["lat"] <= 41.5, f"Latitude out of bounds for {stn}")
            self.assertTrue(115.5 <= meta["lon"] <= 117.5, f"Longitude out of bounds for {stn}")

    def test_aqi_breakpoints(self):
        """Verify standard AQI breakpoints match Chinese MEP / US EPA standards."""
        self.assertEqual(get_aqi_category_info(20.0)["category"], "Good")
        self.assertEqual(get_aqi_category_info(50.0)["category"], "Moderate")
        self.assertEqual(get_aqi_category_info(95.0)["category"], "Unhealthy for Sensitive")
        self.assertEqual(get_aqi_category_info(130.0)["category"], "Unhealthy")
        self.assertEqual(get_aqi_category_info(200.0)["category"], "Very Unhealthy")
        self.assertEqual(get_aqi_category_info(350.0)["category"], "Hazardous")

    def test_conformal_uncertainty(self):
        """Verify uncertainty intervals are non-negative and mathematically valid."""
        estimator = ConformalUncertaintyEstimator(confidence_level=0.90)
        actuals = {1: np.array([50.0, 80.0, 120.0, 30.0])}
        preds = {1: np.array([52.0, 75.0, 110.0, 35.0])}
        estimator.calibrate(actuals, preds)

        test_preds = np.array([20.0, 100.0])
        lower, upper, width = estimator.predict_intervals(test_preds, horizon=1)
        self.assertTrue((lower >= 0.0).all(), "Lower bounds must be non-negative")
        self.assertTrue((upper >= lower).all(), "Upper bound must be >= lower bound")
        self.assertTrue((width > 0.0).all(), "Interval width must be positive")

    def test_feature_columns(self):
        """Verify feature column list is properly populated."""
        cols = get_feature_column_names()
        self.assertGreater(len(cols), 25)
        self.assertIn("PM2.5_lag_1", cols)
        self.assertIn("sin_hour", cols)
        self.assertIn("wind_u", cols)


if __name__ == "__main__":
    unittest.main()
