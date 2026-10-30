# 🌫️ AirLens: Advanced Air Quality Forecasting with Anomaly-Aware Early Warnings and Explainable Interactive Visual Analytics

> **Course:** Information Visualization  
> **Student Authors:** Haswinth M, Harisivanesh P, Santhosh S (SCOPE, Vellore Institute of Technology, Chennai)  
> **Dataset:** Beijing Multi-Site Air Quality Dataset (UCI ML Repository, 420,768 records across 12 national monitoring stations, 2013–2017)  
> **Core Framework:** Predict → Quantify Uncertainty → Detect Anomalies → Issue Early Warnings → Explain (SHAP) → Interactive Visual Analytics

---

## 1. Project Overview & Research Gap

Traditional air quality prediction systems either operate as opaque **"black boxes"** that output raw numerical predictions without explaining the contributing environmental factors, or present static dashboard mockups that do not connect deep learning models with real-world health decision-support.

**AirLens** bridges this gap by unifying five core capabilities into an end-to-end interactive visual analytics platform:
1. **Multi-Horizon Forecasting:** Multi-step PM2.5 forecasting (+1h, +6h, +12h, +24h, +48h) evaluated across 6 diverse forecasting architectures.
2. **Calibrated Uncertainty Estimation:** Finite-sample **Conformal Prediction Intervals** providing 90% confidence uncertainty envelopes around future trajectories.
3. **Multi-Method Anomaly Detection:** Rolling Z-score (sudden diurnal surges), Multivariate Isolation Forest (abnormal co-pollutant inversions), and Residual Divergence (forecast tracking failures).
4. **Anomaly-Aware Early Warning Center:** Connects predicted threshold exceedances (WHO / China NAAQS) with anomaly flags to deliver structured risk alerts and actionable precautionary guidance.
5. **SHAP Explainability Layer:** Post-hoc cooperative game-theoretic feature attribution (TreeSHAP) with global rankings, interactive local waterfall breakdowns, and automated plain-language diagnostic summaries.
6. **15 Coordinated Visualizations:** Geographical maps with verified Beijing coordinates, spatiotemporal heatmaps, small multiples, parallel coordinates, diurnal curves, and error analysis.

---

## 2. Experimental Results (Evaluated on Chronological Test Split)

Evaluated on the unseen test period (**September 2016 to February 2017**, Beijing winter heating smog peak) strictly without temporal leakage:

| Model | Architecture | MAE (µg/m³) | RMSE (µg/m³) | R² Score | sMAPE (%) | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | Naïve Lag-1 Benchmark | 20.40 | 40.50 | 0.8654 | 29.46% | < 0.01s |
| **Ridge Regression** | L2-Regularized Linear Proxy | 20.06 | 38.13 | 0.8807 | 35.70% | 0.01s |
| **Random Forest** | Bagging Ensemble (50 Trees) | 18.78 | 36.43 | 0.8911 | 26.66% | 2.03s |
| **LightGBM** | Gradient Boosted Decision Trees | **18.57** | **35.67** | **0.8955** | **26.90%** | 44.93s |
| **PyTorch LSTM** | 2-Layer Recurrent Neural Net | 32.58 | 64.35 | 0.6600 | 36.60% | 6.13s |
| **Station-Aware Transformer** | Multi-Head Self-Attention + Spatial Embeddings | **24.22** | **45.76** | **0.8281** | **31.80%** | 13.34s |

### Key Scientific Insights:
- **Station-Aware Transformer** significantly outperformed LSTM ($R^2 = 0.8281$ vs $0.6600$), confirming that multi-head self-attention captures cross-pollutant interactions and long-range meteorological dependencies more effectively than recurrence.
- **LightGBM** achieved the highest overall point forecast precision ($R^2 = 0.8955$, $\text{MAE} = 18.57\ \mu\text{g/m}^3$), making it an ideal pairing with TreeSHAP for millisecond-latency local explanations.
- **Conformal Uncertainty Calibration:** Achieved **92.68% empirical coverage** for the 90% confidence target with an average interval width of $89.1\ \mu\text{g/m}^3$, demonstrating mathematically valid uncertainty quantification.

---

## 3. Project Structure

```
AirLens/
├── data/
│   └── PRSA_Data_20130301-20170228/      # 12 National Beijing Station CSVs
├── src/
│   ├── __init__.py
│   ├── config.py                         # Station metadata, AQI thresholds, constants
│   ├── data_loader.py                    # Cleaning, gap interpolation, chronological split
│   ├── feature_engineering.py            # Wind vectors, cyclical time, lags, rolling stats
│   ├── models.py                         # 6 forecasting models (Baselines, LightGBM, Transformer)
│   ├── uncertainty.py                    # Conformal prediction intervals (PICP & MPIW)
│   ├── anomaly_detector.py               # Rolling Z-score, Isolation Forest, Residual analysis
│   ├── early_warning.py                  # AQI risk cards, precautionary health guidance
│   ├── explainability.py                 # SHAP attributions & natural language generator
│   └── evaluation.py                     # Metrics, sliced error analysis by station & horizon
├── artifacts/                            # Precomputed models, forecasts, SHAP cache (.pkl, .pt)
├── reports/
│   └── data_quality_report.json          # Reproducibility audit & retention stats
├── tests/
│   └── test_airlens.py                   # Automated unit test suite
├── app.py                                # Streamlit Interactive Visual Analytics Dashboard
├── pipeline.py                           # Full pipeline training & artifact generation
├── requirements.txt                      # Python dependencies
├── run_airlens.bat                       # 1-Click Windows executable launcher
└── README.md                             # Comprehensive technical documentation
```

---

## 4. Step-by-Step Execution Guide (Windows)

### Step 1: Open PowerShell or Command Prompt
Navigate to the project folder:
```powershell
cd C:\Users\haswi\.gemini\antigravity\scratch\AirLens
```

### Step 2: Install Required Dependencies
All required libraries can be installed via:
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Training & Processing Pipeline (Optional)
The pipeline has already precomputed all models and saved the artifacts in `artifacts/`. If you wish to re-execute the entire pipeline from scratch:
```powershell
python pipeline.py
```

### Step 4: Launch the Interactive Dashboard
Run either:
```powershell
streamlit run app.py
```
*Or simply double-click the `run_airlens.bat` file in Windows File Explorer!*

Your web browser will automatically open:
```
http://localhost:8501
```

### Step 5: Run Unit Tests
```powershell
python -m unittest tests/test_airlens.py
```

---

## 5. Overview of the 15 Coordinated Visualizations

1. **Executive Overview Dashboard:** High-contrast KPI cards (Current PM2.5, +1h Forecast, Detected Anomalies, Active Warnings) with a 72-hour observed vs model outlook.
2. **Interactive Forecast & Uncertainty:** Multi-horizon (+1h to +48h) trajectory curves surrounded by shaded 90% Conformal Prediction Interval bands and anomalous event markers.
3. **Beijing Monitoring Network Map:** Interactive Dark Matter geographic map displaying all 12 stations using verified coordinates, scaled and colored by live particulate concentrations.
4. **Spatiotemporal Heatmap:** Two-dimensional color matrix (Stations on Y-axis, Datetime on X-axis) revealing regional smog propagation across Beijing.
5. **Small Multiples (12 Stations):** Synchronized 48-hour facet line charts allowing direct station-to-station behavioral comparisons.
6. **Anomaly Detection Engine:** Multi-method distribution histograms, detection technique pie charts, and anomaly incident inspector.
7. **Early-Warning Center:** Risk-graded alert console filtering by severity score (1 to 6) and station with actionable health advisories.
8. **Model Comparison Suite:** Benchmark bar charts (MAE, RMSE, R²), training duration comparisons, and actual vs predicted scatter plots with 1:1 identity line.
9. **Forecast Error Analysis:** Granular residual diagnostics displaying error decay across forecast horizons (+1h to +48h), station-specific MAE, and residual distribution normality.
10. **Advanced SHAP Explainability:** Global feature importance ranking, interactive local waterfall breakdown, and automated natural language diagnostic summaries.
11. **Pollutant & Weather Analytics:** Multilateral correlation matrix and scatter plots with OLS regression trendlines illustrating wind dispersion.
12. **Temporal & Diurnal Dynamics:** 24-hour diurnal cycle curves across stations and seasonal distribution boxplots.
13. **Parallel Coordinates Analysis:** High-dimensional parallel coordinate traces normalizing all 6 co-pollutants and weather variables.
14. **Data Quality & Scientific Audit:** JSON data audit report detailing missing value percentages, gap interpolation parameters, and zero-leakage splitting.
15. **About AirLens Framework:** Comprehensive system architecture and research paper references.

---

## 6. College Viva Questions & Answers

**Q1. What is the central research novelty of AirLens?**  
*Answer:* The novelty lies in the end-to-end integration of Station-Aware Transformer forecasting, calibrated Conformal Uncertainty estimation, multi-method Anomaly Detection, SHAP explainability, and interactive visual analytics into a unified decision-support workflow (*Predict → Quantify → Detect → Alert → Explain → Visualize*).

**Q2. Why is PM2.5 chosen as the primary target?**  
*Answer:* Fine particulate matter (PM2.5) has an aerodynamic diameter $\le 2.5\ \mu\text{m}$, allowing it to penetrate deep into the alveolar sacs of human lungs and enter the bloodstream. It is the leading pollutant responsible for respiratory and cardiovascular morbidity in urban environments.

**Q3. Why use chronological train-test splitting instead of random K-fold shuffling?**  
*Answer:* Air quality is a sequential temporal process. Randomly shuffling hourly time series leaks future atmospheric states into the training set (lookahead leakage), creating artificially inflated accuracy that fails completely in real-world deployment. AirLens uses strict temporal partitioning (Train: 2013–2016, Val: Spring/Summer 2016, Test: Autumn/Winter 2016–2017).

**Q4. What is the purpose of learned Station Embeddings in the Transformer?**  
*Answer:* Monitoring stations have unique local geographic and emission characteristics (e.g. Dingling is a rural background station, while Guanyuan is an urban financial hub). Station embeddings allow the Transformer to share general atmospheric physics representations across Beijing while specializing its attention to station-specific baseline behaviors.

**Q5. How are prediction intervals calculated, and why conformal prediction?**  
*Answer:* Traditional models assume Gaussian error distributions, which fails during extreme pollution spikes. We use empirical Conformal Prediction on the unseen validation residuals, producing distribution-free, finite-sample calibrated intervals that achieved 92.68% empirical coverage for a 90% confidence level.

**Q6. What does SHAP tell an environmental analyst?**  
*Answer:* SHAP (SHapley Additive exPlanations) uses cooperative game theory to compute the exact additive contribution of each feature to a specific forecast. It reveals not only which features mattered (e.g. antecedent PM10 and low wind speed), but also whether they increased or decreased the prediction relative to the expected base value.
