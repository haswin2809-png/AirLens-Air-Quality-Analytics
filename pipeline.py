"""
AirLens: End-to-End Pipeline Orchestrator
Executes data loading, preprocessing, feature engineering, multi-model training,
uncertainty calibration, anomaly detection, early warning generation, and SHAP explainability.
Saves all artifacts for high-speed dashboard visual analytics.
"""

import time
import os
import joblib
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.config import ARTIFACTS_DIR, REPORTS_DIR, FORECAST_HORIZONS
from src.data_loader import load_raw_dataset, preprocess_data, split_train_val_test
from src.feature_engineering import build_features, get_feature_column_names
from src.models import (
    PersistenceModel, RidgeModel, RandomForestModel, LightGBMModel,
    PyTorchLSTM, StationAwareTransformer, AirQualityDataset, train_deep_model
)
from src.uncertainty import ConformalUncertaintyEstimator
from src.anomaly_detector import AirLensAnomalyDetector
from src.early_warning import generate_early_warnings
from src.explainability import AirLensExplainer
from src.evaluation import compute_metrics, analyze_errors_by_slice


def run_pipeline():
    print("=" * 70)
    print("  AIRLENS: END-TO-END PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Load and Preprocess Dataset
    print("\n[Phase 1/6] Loading raw dataset from 12 monitoring stations...")
    raw_df = load_raw_dataset()
    print(f"Loaded {len(raw_df):,} raw hourly observations.")

    print("\n[Phase 2/6] Cleaning, imputing gaps, and screening outliers...")
    clean_df, quality_report = preprocess_data(raw_df)
    print(f"Cleaned dataset: {len(clean_df):,} rows. Retention rate: {quality_report['data_retention_rate_pct']}%")

    # 2. Feature Engineering
    print("\n[Phase 3/6] Generating temporal, meteorological, and lag features...")
    featured_df = build_features(clean_df)
    feature_cols = get_feature_column_names()
    print(f"Constructed {len(feature_cols)} predictor features across {len(featured_df):,} valid records.")

    # 3. Chronological Train / Val / Test Splitting
    print("\n[Phase 4/6] Chronological train / validation / test partitioning...")
    train_df, val_df, test_df = split_train_val_test(featured_df)
    print(f"Train set: {len(train_df):,} rows (2013-03 to 2016-02)")
    print(f"Val set:   {len(val_df):,} rows (2016-03 to 2016-08)")
    print(f"Test set:  {len(test_df):,} rows (2016-09 to 2017-02: Winter peak)")

    # Sample representative subsets for efficient laptop execution & rapid iteration
    # Take stratified sample of train (e.g. 35,000 rows) and val/test (10,000 rows each)
    np.random.seed(42)
    train_sample = train_df.sample(n=min(35000, len(train_df)), random_state=42).sort_values('datetime')
    val_sample = val_df.sample(n=min(8000, len(val_df)), random_state=42).sort_values('datetime')
    # For test, keep a continuous chronological slice of the recent winter season for realistic visualization
    test_slice = test_df.tail(6000).copy()

    X_train = train_sample[feature_cols].values
    y_train_h1 = train_sample['target_h1'].values
    y_train_multi = train_sample[[f'target_h{h}' for h in FORECAST_HORIZONS]].values

    X_val = val_sample[feature_cols].values
    y_val_h1 = val_sample['target_h1'].values
    y_val_multi = val_sample[[f'target_h{h}' for h in FORECAST_HORIZONS]].values

    X_test = test_slice[feature_cols].values
    y_test_h1 = test_slice['target_h1'].values
    y_test_multi = test_slice[[f'target_h{h}' for h in FORECAST_HORIZONS]].values

    # 4. Multi-Model Training and Evaluation
    print("\n[Phase 5/6] Training & Evaluating 6 Forecasting Architectures...")
    model_results = []
    trained_models = {}
    test_preds_by_model = {}

    # A. Persistence Baseline
    t0 = time.time()
    persist = PersistenceModel().fit(X_train, y_train_h1)
    persist_time = time.time() - t0
    persist_preds = persist.predict(test_slice)
    persist_metrics = compute_metrics(y_test_h1, persist_preds)
    persist_metrics.update({"Model": persist.name, "Train_Time_s": round(persist_time, 2)})
    model_results.append(persist_metrics)
    test_preds_by_model[persist.name] = persist_preds

    # B. Ridge Regression
    t0 = time.time()
    ridge = RidgeModel(alpha=1.0).fit(X_train, y_train_h1)
    ridge_time = time.time() - t0
    ridge_preds = ridge.predict(X_test)
    ridge_metrics = compute_metrics(y_test_h1, ridge_preds)
    ridge_metrics.update({"Model": ridge.name, "Train_Time_s": round(ridge_time, 2)})
    model_results.append(ridge_metrics)
    test_preds_by_model[ridge.name] = ridge_preds

    # C. Random Forest
    print(" - Training Random Forest Regressor...")
    t0 = time.time()
    rf = RandomForestModel(n_estimators=50, max_depth=12).fit(X_train, y_train_h1)
    rf_time = time.time() - t0
    rf_preds = rf.predict(X_test)
    rf_metrics = compute_metrics(y_test_h1, rf_preds)
    rf_metrics.update({"Model": rf.name, "Train_Time_s": round(rf_time, 2)})
    model_results.append(rf_metrics)
    test_preds_by_model[rf.name] = rf_preds

    # D. LightGBM (Multi-Horizon)
    print(" - Training LightGBM Gradient Boosted Forecaster...")
    lgbm_models = {}
    lgbm_horizon_preds = {}
    lgbm_val_preds = {}
    t0 = time.time()
    for h in FORECAST_HORIZONS:
        lgbm = LightGBMModel(n_estimators=100, learning_rate=0.08)
        lgbm.fit(X_train, train_sample[f'target_h{h}'].values)
        lgbm_models[h] = lgbm
        lgbm_val_preds[h] = lgbm.predict(X_val)
        lgbm_horizon_preds[h] = lgbm.predict(X_test)
    lgbm_total_time = time.time() - t0
    
    lgbm_metrics = compute_metrics(y_test_h1, lgbm_horizon_preds[1])
    lgbm_metrics.update({"Model": "LightGBM", "Train_Time_s": round(lgbm_total_time, 2)})
    model_results.append(lgbm_metrics)
    test_preds_by_model["LightGBM"] = lgbm_horizon_preds[1]
    trained_models["LightGBM_h1"] = lgbm_models[1]

    # E. PyTorch LSTM
    print(" - Training PyTorch LSTM Sequence Forecaster...")
    train_dataset = AirQualityDataset(X_train, train_sample['station_id'].values, y_train_multi)
    val_dataset = AirQualityDataset(X_val, val_sample['station_id'].values, y_val_multi)
    test_dataset = AirQualityDataset(X_test, test_slice['station_id'].values)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    lstm_net = PyTorchLSTM(num_features=len(feature_cols))
    lstm_net, lstm_time = train_deep_model(lstm_net, train_loader, val_loader, epochs=5)
    
    lstm_net.eval()
    with torch.no_grad():
        lstm_preds_list = []
        for bx, bstn in test_loader:
            lstm_preds_list.append(lstm_net(bx, bstn).numpy())
        lstm_preds_all = np.vstack(lstm_preds_list)
    
    lstm_preds_h1 = np.clip(lstm_preds_all[:, 0], 0, 1000)
    lstm_metrics = compute_metrics(y_test_h1, lstm_preds_h1)
    lstm_metrics.update({"Model": "LSTM (PyTorch)", "Train_Time_s": round(lstm_time, 2)})
    model_results.append(lstm_metrics)
    test_preds_by_model["LSTM (PyTorch)"] = lstm_preds_h1

    # F. Station-Aware Transformer
    print(" - Training Station-Aware Transformer (Spatial Attention)...")
    transformer_net = StationAwareTransformer(num_features=len(feature_cols))
    transformer_net, trans_time = train_deep_model(transformer_net, train_loader, val_loader, epochs=6)
    
    transformer_net.eval()
    with torch.no_grad():
        trans_preds_list = []
        for bx, bstn in test_loader:
            trans_preds_list.append(transformer_net(bx, bstn).numpy())
        trans_preds_all = np.vstack(trans_preds_list)
    
    trans_preds_h1 = np.clip(trans_preds_all[:, 0], 0, 1000)
    trans_metrics = compute_metrics(y_test_h1, trans_preds_h1)
    trans_metrics.update({"Model": "Station-Aware Transformer", "Train_Time_s": round(trans_time, 2)})
    model_results.append(trans_metrics)
    test_preds_by_model["Station-Aware Transformer"] = trans_preds_h1

    # Compile Model Evaluation Table
    eval_df = pd.DataFrame(model_results)
    print("\n--- MODEL PERFORMANCE COMPARISON (TEST SPLIT) ---")
    print(eval_df.to_string(index=False))

    # 5. Uncertainty Estimation (Conformal Intervals)
    print("\n[Phase 6/6] Computing Calibrated Prediction Intervals & Anomaly Engine...")
    uncertainty_estimator = ConformalUncertaintyEstimator(confidence_level=0.90)
    val_actuals = {h: val_sample[f'target_h{h}'].values for h in FORECAST_HORIZONS}
    uncertainty_estimator.calibrate(val_actuals, lgbm_val_preds)
    
    # Compute test intervals for LightGBM
    lower_h1, upper_h1, width_h1 = uncertainty_estimator.predict_intervals(
        lgbm_horizon_preds[1], horizon=1, rolling_volatility=test_slice['PM2.5_roll_std_6h'].values
    )
    coverage_eval = uncertainty_estimator.evaluate_coverage(y_test_h1, lower_h1, upper_h1)
    print(f"90% Prediction Interval Empirical Coverage: {coverage_eval['empirical_coverage_pct']}% (Mean Width: {coverage_eval['mean_interval_width']:.1f} µg/m³)")

    # 6. Anomaly Detection
    detector = AirLensAnomalyDetector()
    detector.fit_isolation_forest(train_sample)
    test_slice_anomalies = detector.detect_anomalies(test_slice, preds=lgbm_horizon_preds[1])
    num_anoms = int(test_slice_anomalies['is_anomaly'].sum())
    print(f"Detected {num_anoms} statistical/environmental anomalies in test period.")

    # 7. Early Warning Generation
    test_slice_anomalies['y_pred'] = lgbm_horizon_preds[1]
    test_slice_anomalies['pred_lower'] = lower_h1
    test_slice_anomalies['pred_upper'] = upper_h1
    
    # Create sample forecast future dataframe
    latest_stn = test_slice_anomalies.iloc[-1]
    sample_forecasts = []
    base_time = pd.to_datetime(latest_stn['datetime'])
    for h in FORECAST_HORIZONS:
        sample_forecasts.append({
            "target_datetime": base_time + pd.Timedelta(hours=h),
            "station": latest_stn['station'],
            "horizon_hours": h,
            "predicted_pm25": float(lgbm_horizon_preds[h][-1])
        })
    forecast_outlook_df = pd.DataFrame(sample_forecasts)

    warnings_df = generate_early_warnings(test_slice_anomalies, forecast_outlook_df)
    print(f"Generated {len(warnings_df)} structured early warnings & risk cards.")

    # 8. SHAP Explainability Layer
    print("\nComputing SHAP attributions and diagnostic explanations...")
    explainer = AirLensExplainer(lgbm_models[1], feature_names=feature_cols)
    shap_sample = test_slice[feature_cols].iloc[:300]
    shap_values, global_importance = explainer.explain_dataset(shap_sample)
    
    # Compute local explanation on high-pollution sample
    high_idx = int(np.argmax(y_test_h1[:100]))
    local_explanation = explainer.explain_instance(test_slice[feature_cols].iloc[high_idx])
    print(f"Sample explanation: {local_explanation['natural_language_summary']}")

    # 9. Error Analysis by Slices
    test_eval_slice = test_slice_anomalies.copy()
    test_eval_slice['y_true'] = test_eval_slice['target_h1']
    test_eval_slice['y_pred'] = lgbm_horizon_preds[1]
    error_slices = analyze_errors_by_slice(test_eval_slice)

    # 10. Multi-Horizon Error Decay
    horizon_metrics_list = []
    for h in FORECAST_HORIZONS:
        h_metrics = compute_metrics(test_slice[f'target_h{h}'].values, lgbm_horizon_preds[h])
        h_metrics['horizon_hours'] = h
        horizon_metrics_list.append(h_metrics)
    horizon_decay_df = pd.DataFrame(horizon_metrics_list)

    # 11. Save All Artifacts
    print("\nSaving precomputed artifacts to disk for instant dashboard analytics...")
    artifacts = {
        "eval_df": eval_df,
        "test_slice": test_slice_anomalies,
        "test_preds_by_model": test_preds_by_model,
        "lgbm_horizon_preds": lgbm_horizon_preds,
        "global_importance": global_importance,
        "shap_values": shap_values,
        "shap_features": feature_cols,
        "coverage_eval": coverage_eval,
        "warnings_df": warnings_df,
        "forecast_outlook_df": forecast_outlook_df,
        "error_slices": error_slices,
        "horizon_decay_df": horizon_decay_df,
        "feature_cols": feature_cols,
        "clean_summary_sample": clean_df.sample(n=min(50000, len(clean_df)), random_state=42)
    }

    joblib.dump(artifacts, ARTIFACTS_DIR / "airlens_artifacts.pkl")
    joblib.dump(lgbm_models[1], ARTIFACTS_DIR / "lgbm_model_h1.pkl")
    torch.save(transformer_net.state_dict(), ARTIFACTS_DIR / "transformer_net.pt")
    
    print("\n" + "=" * 70)
    print("  AIRLENS PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Artifacts saved in: {ARTIFACTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
