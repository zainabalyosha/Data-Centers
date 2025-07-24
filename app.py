import os, time, joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Optional live-refresh helper
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None  # graceful fallback

from response_rules import make_plan  # your existing rules file

# ---------------------------- CONFIG ----------------------------
st.set_page_config(page_title="Climate-Aware DC Ops Demo", layout="wide")

PAGES = [
    "🏠 Dashboard",
    "📍 Data Centers",
    "🔮 Scenario Simulator",
    "🛠️ Action Engine",
    "📈 Climate & Risk Outlook",
    "📊 Risk Forecast",
    "🧪 Data Fusion (raw)",
]

page = st.sidebar.radio("Page", PAGES)

# ------------------------- DATA / MODEL -------------------------
@st.cache_resource(show_spinner=False)
def load_data():
    weather = pd.read_csv("weather_mock.csv", parse_dates=["date"])
    dc      = pd.read_csv("dc_mock.csv", parse_dates=["date"])
    merged  = pd.read_csv("merged_training_sample.csv")
    return weather, dc, merged

@st.cache_resource(show_spinner=False)
def get_model_and_features(merged_df):
    if os.path.exists("model.pkl"):
        try:
            model = joblib.load("model.pkl")
            feats = pd.read_csv("feature_order.csv", header=None)[0].tolist()
            return model, feats
        except Exception:
            pass

    features = ["max_temp_F","rolling_max7","humidity_idx","load_MW","cooling_kW"]
    X = merged_df[features]
    y = merged_df["extreme_heat_event"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=250, class_weight="balanced", random_state=42
    ).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
    st.caption(f"Demo model AUC: {auc:.3f}")

    joblib.dump(model, "model.pkl")
    pd.Series(features).to_csv("feature_order.csv", index=False)
    return model, features

weather, dc, merged_df = load_data()
model, FEATURES = get_model_and_features(merged_df)

# Build a DC inventory table (city/state + typical load/cooling)
@st.cache_resource(show_spinner=False)
def build_inventory(dc_df):
    inv = (
        dc_df.groupby("dc_id")
            .agg(city=("city", "first"),
                 state=("state","first"),
                 avg_load_MW=("load_MW","mean"),
                 avg_cooling_kW=("cooling_kW","mean"))
            .reset_index()
            .sort_values("dc_id")
    )
    return inv.round({"avg_load_MW":2, "avg_cooling_kW":1})

dc_inventory = build_inventory(dc.merge(weather[["date","city","state"]], on=["date","city","state"]))

# Helper: score risk for ALL DCs on a chosen date
def score_day(date_obj):
    wday = weather[weather["date"].dt.date == date_obj]
    dday = dc[dc["date"].dt.date == date_obj]
    if wday.empty or dday.empty:
        return pd.DataFrame()
    # merge each dc row with the single day's weather for that DC's city
    # (our mock weather is per city/state, so join on city/state)
    day_all = dday.merge(
        wday[["city","state","max_temp_F","rolling_max7","humidity_idx"]],
        on=["city","state"], how="left"
    )
    X = day_all[FEATURES]
    risks = model.predict_proba(X)[:,1]
    day_all["risk"] = risks
    return day_all

# Forecast helper: risk series for a DC across a range
def risk_series_for_dc(dc_id, start_date=None, end_date=None):
    dcdc = dc[dc["dc_id"]==dc_id]
    if start_date: dcdc = dcdc[dcdc["date"]>=pd.Timestamp(start_date)]
    if end_date:   dcdc = dcdc[dcdc["date"]<=pd.Timestamp(end_date)]

    wsub = weather[weather["city"].isin(dcdc["city"].unique())]

    df = dcdc.merge(
        wsub[["date","max_temp_F","rolling_max7","humidity_idx","city","state"]],
        on=["date","city","state"], how="left"
    )
    X = df[FEATURES]
    df["risk"] = model.predict_proba(X)[:,1]
    return df[["date","risk","max_temp_F","rolling_max7","humidity_idx"]].sort_values("date")

# ---------------------------- PAGES -----------------------------

if page == "🏠 Dashboard":
    st.title("Real-time-ish Dashboard")

    # auto refresh every 5 sec (optional)
    if st_autorefresh:
        st_autorefresh(interval=5000, key="refresh")

    today = weather["date"].max().date()
    scored = score_day(today)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("# of Data Centers", len(dc_inventory))
    with col2:
        if not scored.empty:
            st.metric("High-Risk DCs Today (risk ≥ 0.7)",
                      int((scored["risk"]>=0.7).sum()))
        else:
            st.metric("High-Risk DCs Today", "—")
    with col3:
        st.metric("Avg Max Temp Today (°F)",
                  f"{weather[weather['date'].dt.date==today]['max_temp_F'].mean():.1f}")
    with col4:
        st.metric("Max Rolling-7 Temp Today (°F)",
                  f"{weather[weather['date'].dt.date==today]['rolling_max7'].max():.1f}")

    st.write("*(Values refresh every ~5 seconds.)*")

    if not scored.empty:
        st.subheader("Today’s Risk by Data Center")
        st.dataframe(scored[["dc_id","city","state","risk","load_MW","cooling_kW"]]
                     .sort_values("risk", ascending=False).round(3))
    else:
        st.info("No data for today in the mock dataset (select another date on Risk Forecast page).")

elif page == "📍 Data Centers":
    st.title("Data Center Inventory")
    st.write("Static list of mocked DCs, their locations, and average loads.")
    st.dataframe(dc_inventory)

    st.subheader("Filter")
    state_pick = st.multiselect("State", sorted(dc_inventory["state"].unique()))
    if state_pick:
        st.dataframe(dc_inventory[dc_inventory["state"].isin(state_pick)])

elif page == "🔮 Scenario Simulator":
    st.title("Scenario Simulator")

    st.markdown("""
Pick a scenario and see what might happen to operations/equipment.

**Scenarios included (demo logic):**
- **Extreme Heat**: High ambient temp → cooling stress, higher load, potential throttling  
- **Grid Shortfall**: Utility curtailment → must shed load / run BESS / gensets  
- **Cooling Failure**: CRAH/CRAC failure → temperature rise, emergency shutdown of non-critical racks  
- **Full Outage**: Power loss → UPS/generator runtime, graceful shutdown sequences
    """)
    scen = st.selectbox("Scenario", ["Extreme Heat","Grid Shortfall","Cooling Failure","Full Outage"])
    dc_id_pick = st.selectbox("Data Center", sorted(dc_inventory["dc_id"]))
    severity = st.slider("Severity (0=minor, 1=catastrophic)", 0.0, 1.0, 0.6, 0.05)

    # Simple narrative outputs
    st.subheader("Predicted Operational Impacts")
    impacts = []
    if scen == "Extreme Heat":
        impacts = [
            "Cooling power spikes; PUE rises.",
            "Fans/CRAH units pushed near limits; risk of hot spots.",
            "Battery rooms & UPS may derate in high temps.",
            "Higher risk of component thermal throttling."
        ]
    elif scen == "Grid Shortfall":
        impacts = [
            "Load shedding required to meet utility curtailment.",
            "Shift non-critical tasks to other regions.",
            "Run BESS/generators; fuel/logistics planning needed.",
            "Risk of SLA breaches if curtailment persists."
        ]
    elif scen == "Cooling Failure":
        impacts = [
            "Room temp climbs rapidly; thermal runaway risk.",
            "Immediate workload migration or throttling needed.",
            "Possible emergency shutdown of affected racks.",
            "Post-event inspection/restart procedures required."
        ]
    elif scen == "Full Outage":
        impacts = [
            "Transfer to UPS → generator sequence.",
            "Strict runtime limits (fuel, battery) drive priorities.",
            "Graceful shutdown of non-essential systems.",
            "Data integrity & restart sequencing critical."
        ]

    # Adjust by severity
    impacts = [f"{imp} (severity factor: {severity:.2f})" for imp in impacts]
    for imp in impacts:
        st.write("- ", imp)

    st.subheader("Suggested Mitigations (demo)")
    plan = make_plan(risk=max(severity,0.7) if scen=="Extreme Heat" else severity,
                     load_mw=25, cooling_kw=3000)
    for p in plan:
        st.write("•", p)

elif page == "🛠️ Action Engine":
    st.title("Automated Response Plan")
    st.write("Choose parameters and see what the engine would 'auto-execute'.")

    risk = st.slider("Risk Score (0–1)", 0.0, 1.0, 0.78, 0.01)
    load = st.number_input("Current Load (MW)", 0.0, 200.0, 25.0, 0.1)
    cool = st.number_input("Cooling Power (kW)", 0.0, 20000.0, 3000.0, 100.0)

    auto = st.checkbox("Pretend to auto-execute actions?")
    plan = make_plan(risk, load, cool)

    st.subheader("Recommended Actions")
    for step in plan:
        st.write("- ", step)

    if auto:
        st.success("✅ Actions sent to (mock) DCIM/BMS API. (In real life: call your orchestration endpoints here.)")

elif page == "📈 Climate & Risk Outlook":
    st.title("Climate & Risk Outlook")

    st.markdown("**Pick a data center** to see projected risk & temp trends from 2019–2024 (mock).")
    dc_pick = st.selectbox("Data Center", sorted(dc_inventory["dc_id"]))
    df_risk = risk_series_for_dc(dc_pick)

    if df_risk.empty:
        st.info("No data for that DC.")
    else:
        import plotly.express as px
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(df_risk, x="date", y="risk", title=f"Risk Score Over Time - {dc_pick}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(df_risk, x="date", y="max_temp_F", title="Max Temp (°F) Over Time")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Monthly Avg Max Temp (historical)")
        monthly = (weather
                   .assign(month=lambda x: x["date"].dt.to_period("M"))
                   .groupby("month")["max_temp_F"]
                   .mean()
                   .reset_index())
        monthly["month"] = monthly["month"].astype(str)
        fig3 = px.line(monthly, x="month", y="max_temp_F",
                       labels={"max_temp_F":"Avg Max Temp (°F)"},
                       title="Monthly Avg Max Temp (All Cities)")
        st.plotly_chart(fig3, use_container_width=True)

elif page == "📊 Risk Forecast":
    st.title("Extreme Heat Risk Forecast (Single Day / DC)")
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
        }])[FEATURES]
        risk = model.predict_proba(X)[:,1][0]
        st.metric("Predicted Heat Risk", f"{risk:.2f}")

        st.subheader("Feature Snapshot")
        st.write(X.T.rename(columns={0:"value"}))

elif page == "🧪 Data Fusion (raw)":
    st.title("Data Fusion: Combining Multiple Sources")
    st.markdown("""
**Mock Sources in this Demo**  
- NOAA/NWS daily weather (temperature & humidity)  
- CMIP/ERA5 climate-like features (rolling max temp & humidity index)  
- Data center telemetry (load & cooling power)  
- Synergy Research hyperscale counts (static)

Below is a sample of the merged dataset.
""")
    st.dataframe(merged_df.head(200))
