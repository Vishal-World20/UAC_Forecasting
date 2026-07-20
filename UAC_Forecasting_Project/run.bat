@echo off
ECHO ============================================================
ECHO  HHS UAC Predictive Forecasting Project - Setup and Launch
ECHO ============================================================

ECHO.
ECHO [1/4] Installing required Python packages...
pip install -r requirements.txt

ECHO.
ECHO [2/4] Running data preprocessing pipeline...
python src\data_preprocessing.py

ECHO.
ECHO [3/4] Training and comparing forecasting models...
python src\train_models.py

ECHO.
ECHO [4/4] Launching Streamlit dashboard...
ECHO      The dashboard will open in your default web browser.
ECHO      Press CTRL+C in this window to stop the app.
ECHO.
streamlit run app\streamlit_app.py

PAUSE
