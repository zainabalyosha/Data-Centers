# Climate-Aware Data Center Heat Risk Demo

**Purpose:** Concept demo for a system that fuses weather/climate data and data-center telemetry to predict heat-driven extreme events and auto-generate pre-emptive action plans (load shift, BESS charge, cooling tweaks).

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501).

## Files
- `weather_mock.csv` – Mocked US-wide weather data (2019–2024), heat only.
- `dc_mock.csv` – Mocked per-DC daily load & cooling telemetry.
- `model.pkl` – Trained RandomForestClassifier.
- `feature_order.csv` – Feature ordering used by the model.
- `response_rules.py` – Simple rule engine for action plans.
- `app.py` – Streamlit front-end (3 pages).
- `train_model.py` – Example training script you can modify.
- `metrics.json` – Contains ROC AUC for the demo model.

## Deployment
**Streamlit Cloud (easiest)**  
1. Push these files to a public GitHub repo.  
2. Go to share.streamlit.io → “New app” → select repo/branch/app.py.  
3. (Optional) Add secrets for real APIs.

**Vercel (for a React front-end alternative)**  
- Replace Streamlit with a React UI and host on Vercel; use a small FastAPI backend (not included here).

## Notes
- This is a **concept demo**. The data are synthetic and the model is illustrative.
- For real systems: plug NOAA/NWS APIs, ERA5/CMIP6 NetCDF, DC telemetry, grid data (EIA/CAISO), and implement robust MLOps + alerting.
