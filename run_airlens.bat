@echo off
title AirLens - Advanced Air Quality Forecasting Dashboard
color 0B
echo ======================================================================
echo             AIRLENS: EXPLAINABLE AIR QUALITY FORECASTING
echo          AND ANOMALY-AWARE INTERACTIVE VISUAL ANALYTICS
echo ======================================================================
echo.
echo Launching AirLens Streamlit Web Dashboard...
echo Your web browser will open automatically at http://localhost:8501
echo Press Ctrl+C in this command window anytime to stop the server.
echo.

python -m streamlit run app.py

pause
