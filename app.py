import streamlit as st
import pandas as pd
import numpy as np
import os, joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from response_rules import make_plan  # keep your existing file

st.set_page_config(page_title="Heat Risk & DC Action Demo", layout="wide")

page = st.sidebar.radio("Page", ["Data Fusion", "Risk Forecast", "Action Plan"])

@st.cache_resource(show_spinner=False)
def load_data():
    weather = pd.read_csv("weather_mock.csv", parse_dates=["date"])
    dc      = pd.read_csv("dc_mock.csv", parse_dates=["date"])
    merged  = pd.read_csv("merged_training_sample.csv")
    return weather, dc, merged

@st.cache_resource(show_spinner=False)
def get_model_and_features(merged_df):
    # Try to load an existing pickle; if it fails, retrain
    if os.path.exists("model.pkl"):
        try:
            model = joblib.load("model.pkl")
            feats = pd.read_csv("feature_order.csv", header=None)[0].tolist()
            return model, feats
        except Exception:
            pass  # fall through

    features = ["max_temp_F","rolling_max7","humidity_idx","load_MW","cooling_kW"]
    X = merged_df[features]
    y = merged_df["extreme_heat_event"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=250, class_weight="balanced", random_state=42
    ).fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    st.caption(f"Demo model AUC: {auc:.3f}")

    joblib.dump(model, "model.pkl")
    pd.Series(features).to_csv("feature_order.csv", index=False)
    return model, features

weather, dc, merged_df = load_data()
model, features = get_model_and_features(merged_df)

if page == "Data Fusion":
    st.title("Data Fusion: Combining Multiple Sources")
    st.markdown("""
**Mock Sources in this Demo**  
- NOAA/NWS daily weather (temperature & humidity)  
- CMIP/ERA5 climate-like features (rolling max temp & humidity index)  
- Data center telemetry (load & cooling power)  
- Synergy Research hyperscale counts (static)

Sample of merged dataset:
""")
    st.dataframe(merged_df.head(200))

elif page == "Risk Forecast":
    st.title("Extreme Heat Risk Forecast")
    all_dates = weather["date"].dt.date.unique()
    pick_date = st.date_input("Date", value=all_dates[-1],
                              min_value=all_dates[0], max_value=all_dates[-1])
    dcs = dc["dc_id"].unique()
    pick_dc = st.selectbox("Data Center ID", sorted(dcs))

    row_weather = weather[weather["date"].dt.date == pick_date]
    row_dc      = dc[(dc["date"].dt.date == pick_date) & (dc["dc_id"]==pick_dc)]

    if row_weather.empty or row_dc.empty:
        st.warning("No data for that selection (demo limitation). Try another date / DC.")
    else:
        wf = row_weather.iloc[0]
        df_dc = row_dc.iloc[0]
        X = pd.DataFrame([{
            "max_temp_F": wf["max_temp_F"],
            "rolling_max7": wf["rolling_max7"],
            "humidity_idx": wf["humidity_idx"],
            "load_MW": df_dc["load_MW"],
            "cooling_kW": df_dc["cooling_kW"]
        }])[features]
        risk = model.predict_proba(X)[:,1][0]
        st.metric("Predicted Heat Risk", f"{risk:.2f}")
        st.subheader("Feature Snapshot")
        st.write(X.T.rename(columns={0:"value"}))

elif page == "Action Plan":
    st.title("Automated Response Plan (Heat Event)")
    risk = st.slider("Risk Score (0 to 1)", 0.0, 1.0, 0.78, 0.01)
    load = st.number_input("Current Load (MW)", 0.0, 200.0, 25.0, 0.1)
    cool = st.number_input("Cooling Power (kW)", 0.0, 20000.0, 3000.0, 100.0)

    plan = make_plan(risk, load, cool)
    st.subheader("Suggested Actions")
    for item in plan:
        st.write("- ", item)

    st.caption("Demo ruleset only. Real deployment would integrate with BMS/DCIM APIs for automated execution.")
