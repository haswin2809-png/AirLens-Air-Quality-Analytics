"""
AirLens: Forecast Uncertainty Estimation Module
Computes calibrated prediction intervals (lower & upper bounds) using Conformal Residual Analysis.
Evaluates prediction interval coverage probability (PICP) and mean prediction interval width (MPIW).
"""

import numpy as np
import pandas as pd
from src.config import CONFIDENCE_LEVEL, FORECAST_HORIZONS


class ConformalUncertaintyEstimator:
    """
    Empirical Conformal Prediction Interval Estimator.
    Computes distribution-free calibrated error bounds for multi-horizon forecasts.
    """
    def __init__(self, confidence_level=CONFIDENCE_LEVEL):
        self.confidence_level = confidence_level
        self.quantile = (1.0 + confidence_level) / 2.0  # e.g., 0.95 for 90% interval
        self.horizon_margins = {}
        self.horizon_metrics = {}

    def calibrate(self, val_actuals: dict[int, np.ndarray], val_preds: dict[int, np.ndarray]):
        """
        Calibrates the nonconformity score on the validation split.
        """
        for h in FORECAST_HORIZONS:
            if h in val_actuals and h in val_preds:
                residuals = np.abs(val_actuals[h] - val_preds[h])
                # Compute empirical quantile of residuals
                margin = float(np.percentile(residuals, self.quantile * 100))
                self.horizon_margins[h] = max(margin, 5.0)  # At least 5 µg/m³ baseline margin

    def predict_intervals(self, preds: np.ndarray, horizon: int, rolling_volatility: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (lower_bound, upper_bound, interval_width) for predictions at a given horizon.
        Optionally scales margin by local atmospheric volatility.
        """
        base_margin = self.horizon_margins.get(horizon, 18.0)
        
        if rolling_volatility is not None:
            # Scale dynamically: higher atmospheric turbulence -> wider uncertainty
            norm_vol = np.clip(rolling_volatility / (np.median(rolling_volatility) + 1e-5), 0.7, 1.8)
            dyn_margin = base_margin * norm_vol
        else:
            dyn_margin = base_margin

        lower = np.maximum(0.0, preds - dyn_margin)
        upper = preds + dyn_margin
        width = upper - lower

        return lower, upper, width

    def evaluate_coverage(self, actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
        """
        Computes Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).
        """
        covered = (actuals >= lower) & (actuals <= upper)
        picp = float(np.mean(covered) * 100)
        mpiw = float(np.mean(upper - lower))
        
        return {
            "target_confidence_pct": round(self.confidence_level * 100, 1),
            "empirical_coverage_pct": round(picp, 2),
            "mean_interval_width": round(mpiw, 2),
            "is_calibrated": picp >= (self.confidence_level * 100 - 5.0)
        }
