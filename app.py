import streamlit as st
import pandas as pd
import numpy as np
import joblib
from response_rules import make_plan

st.set_page_config(page_title="Heat Risk & DC Action Demo", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Data Fusion", "Risk Forecast", "Action Plan"])

# Load data
weather = pd.read_csv("weather_mock.csv", parse_dates=["date"])
dc      = pd.read_csv("dc_mock.csv", parse_dates=["date"])
features = pd.read_csv("feature_order.csv", header=None)[0].tolist()
model  = joblib.load("model.pkl")

if page == "Data Fusion":
    st.title("Data Fusion: Combining Multiple Sources")
    st.markdown("""
    **Mock Sources in this Demo**  
    - NOAA/NWS daily weather (temperature & humidity)  
    - CMIP/ERA5 climate features (represented by rolling max temp & humidity index)  
    - Data center telemetry (load & cooling power)  
    - Synergy Research hyperscale counts (cited in your paper, static info)  

    Below is a sample of the merged dataset (first 200 rows).
    """)
    merged = dc.merge(weather, on="date")
    st.dataframe(merged.head(200))

elif page == "Risk Forecast":
    st.title("Extreme Heat Risk Forecast")
    st.write("Select a date and a data center to score heat risk.")

    # Pick date
    all_dates = weather["date"].dt.date.unique()
    pick_date = st.date_input("Date", value=all_dates[-1], min_value=all_dates[0], max_value=all_dates[-1])

    # Pick data center
    dcs = dc["dc_id"].unique()
    pick_dc = st.selectbox("Data Center ID", sorted(dcs))

    # Prepare features
    row_weather = weather[weather["date"].dt.date == pick_date]
    row_dc      = dc[(dc["date"].dt.date == pick_date) & (dc["dc_id"]==pick_dc)]

    if row_weather.empty or row_dc.empty:
        st.warning("No data for selection (demo limitation). Try another date / DC.")
    else:
        # In real app we'd use location-specific merge, for demo assume same date join
        wf = row_weather.iloc[0]
        df_dc = row_dc.iloc[0]
        X = pd.DataFrame([{
            "max_temp_F": wf["max_temp_F"],
            "rolling_max7": wf["rolling_max7"],
            "humidity_idx": wf["humidity_idx"],
            "load_MW": df_dc["load_MW"],
            "cooling_kW": df_dc["cooling_kW"]
        }])
        X = X[features]
        risk = model.predict_proba(X)[:,1][0]
        st.metric("Predicted Heat Risk", f"{risk:.2f}")

        st.subheader("Feature Snapshot")
        st.write(X.T.rename(columns={0:"value"}))
        st.caption("Interpretation: Higher rolling_max7 and humidity_idx usually increase risk.")

elif page == "Action Plan":
    st.title("Automated Response Plan (Heat Event)")
    st.write("Adjust the sliders to simulate a risk score & current operating state.")
    risk = st.slider("Risk Score (0 to 1)", 0.0, 1.0, 0.78, 0.01)
    load = st.number_input("Current Load (MW)", 0.0, 200.0, 25.0, 0.1)
    cool = st.number_input("Cooling Power (kW)", 0.0, 20000.0, 3000.0, 100.0)

    plan = make_plan(risk, load, cool)
    st.subheader("Suggested Actions")
    for item in plan:
        st.write("- ", item)

    st.caption("Note: Demo ruleset only. Real deployment would integrate with BMS/DCIM APIs for automated execution.")
