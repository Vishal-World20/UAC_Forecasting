#!/bin/bash
echo "============================================================"
echo " HHS UAC Predictive Forecasting Project - Setup and Launch"
echo "============================================================"

echo ""
echo "[1/4] Installing required Python packages..."
pip install -r requirements.txt

echo ""
echo "[2/4] Running data preprocessing pipeline..."
python3 src/data_preprocessing.py

echo ""
echo "[3/4] Training and comparing forecasting models..."
python3 src/train_models.py

echo ""
echo "[4/4] Launching Streamlit dashboard..."
echo "      Press CTRL+C to stop the app."
echo ""
streamlit run app/streamlit_app.py
