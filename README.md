# Climate-Aware Data Center Heat Risk Demo

**Purpose:** Concept demo for a system that fuses weather/climate data and data-center telemetry to predict heat-driven extreme events and auto-generate pre-emptive action plans (load shift, BESS charge, cooling tweaks).


## Files
- `weather_mock.csv` – Mocked US-wide weather data (2019–2024), heat only.
- `dc_mock.csv` – Mocked per-DC daily load & cooling telemetry.
- `model.pkl` – Trained RandomForestClassifier.
- `feature_order.csv` – Feature ordering used by the model.
- `response_rules.py` – Simple rule engine for action plans.
- `app.py` – Streamlit front-end (3 pages).
- `train_model.py` – Example training script you can modify.
- `metrics.json` – Contains ROC AUC for the demo model.


## Notes
- This is a **concept demo**. The data are synthetic and the model is illustrative.
