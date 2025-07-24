import os, json, time, requests, joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Optional live refresh
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None  # fallback

from response_rules import make_plan  # your existing rules file

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# DATA LOADING & MODEL
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_data():
    weather = pd.read_csv("weather_mock.csv", parse_dates=["date"])
    dc      = pd.read_csv("dc_mock.csv", parse_dates=["date"])
    merged  = pd.read_csv("merged_training_sample.csv")
    return weather, dc, merged

# ------- NOAA real data for one city (Sacramento, CA) -------------
NOAA_ENDPOINT = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
STATION_ID = "GHCND:USW00023232"   # Sacramento Exec Airport (example)

@st.cache_resource(show_spinner=True)
def fetch_noaa_daily_max(start="2019-01-01", end="2024-12-31"):
    token = st.secrets.get("NOAA_TOKEN", "")
    if not token:
        return None  # no token set, skip
    params = {
        "datasetid": "GHCND",
        "datatypeid": "TMAX",
        "stationid": STATION_ID,
        "startdate": start,
        "enddate": end,
        "limit": 1000,
        "units": "standard"  # Fahrenheit
    }
    headers = {"token": token}
    results = []
    offset = 1
    while True:
        params["offset"] = offset
        r = requests.get(NOAA_ENDPOINT, headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            break
        data = r.json().get("results", [])
        if not data:
            break
        results.extend(data)
        offset += len(data)
        if len(data) < params["limit"]:
            break

    if not results:
        return None

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns={"value":"max_temp_F_noaa"})[["date","max_temp_F_noaa"]]
    return df

@st.cache_resource(show_spinner=False)
def get_model_and_features(merged_df):
    # try loading cached model
    if os.path.exists("model.pkl"):
        try:
            model = joblib.load("model.pkl")
            feats = pd.read_csv("feature_order.csv", header=None)[0].tolist()
            return model, feats
        except Exception:
            pass
    # otherwise retrain
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

# Load raw data
weather, dc, merged_df = load_data()

# Try to blend real NOAA data for Sacramento
noaa_df = fetch_noaa_daily_max()
if noaa_df is not None:
    noaa_df["date"] = pd.to_datetime(noaa_df["date"])
    mask = (weather["city"]=="Sacramento") & (weather["state"]=="CA")
    weather = weather.merge(noaa_df, on="date", how="left")
    weather.loc[mask & weather["max_temp_F_noaa"].notna(), "max_temp_F"] = \
        weather.loc[mask & weather["max_temp_F_noaa"].notna(), "max_temp_F_noaa"]
    # recompute features for Sacramento rows
    weather["rolling_max7"] = weather.groupby("city")["max_temp_F"].transform(lambda s: s.rolling(7, min_periods=1).max())
    weather["humidity_idx"] = weather["humidity_pct"] * weather["max_temp_F"] / 100
    # rebuild merged
    merged_df = dc.merge(weather, on=["date","city","state"])

# Train / load model
model, FEATURES = get_model_and_features(merged_df)

# ------------------------------------------------------------------
# INVENTORY + MAP SUPPORT
# ------------------------------------------------------------------
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

# City lat/lon (rough centroids)
CITY_LATLON = {
    ("Phoenix","AZ"): (33.4484, -112.0740),
    ("Sacramento","CA"): (38.5816, -121.4944),
    ("Los Angeles","CA"): (34.0522, -118.2437),
    ("Houston","TX"): (29.7604, -95.3698),
    ("Dallas","TX"): (32.7767, -96.7970),
    ("Miami","FL"): (25.7617, -80.1918),
    ("Atlanta","GA"): (33.7490, -84.3880),
    ("Chicago","IL"): (41.8781, -87.6298),
    ("New York","NY"): (40.7128, -74.0060),
    ("Seattle","WA"): (47.6062, -122.3321),
    ("Denver","CO"): (39.7392, -104.9903),
    ("Las Vegas","NV"): (36.1699, -115.1398),
    ("Boston","MA"): (42.3601, -71.0589),
    ("Minneapolis","MN"): (44.9778, -93.2650),
    ("Portland","OR"): (45.5051, -122.6750),
    ("Salt Lake City","UT"): (40.7608, -111.8910),
    ("Nashville","TN"): (36.1627, -86.7816),
    ("Raleigh","NC"): (35.7796, -78.6382),
    ("Philadelphia","PA"): (39.9526, -75.1652),
    ("Kansas City","MO"): (39.0997, -94.5786),
}

def add_latlon(inv_df):
    inv_df = inv_df.copy()
    inv_df["lat"] = inv_df.apply(lambda r: CITY_LATLON.get((r.city, r.state), (np.nan,np.nan))[0], axis=1)
    inv_df["lon"] = inv_df.apply(lambda r: CITY_LATLON.get((r.city, r.state), (np.nan,np.nan))[1], axis=1)
    return inv_df

dc_inventory = build_inventory(dc.merge(weather[["date","city","state"]], on=["date","city","state"]))
dc_inventory = add_latlon(dc_inventory)

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def score_day(date_obj):
    """Score all DCs on a given date."""
    wday = weather[weather["date"].dt.date == date_obj]
    dday = dc[dc["date"].dt.date == date_obj]
    if wday.empty or dday.empty:
        return pd.DataFrame()
    day_all = dday.merge(
        wday[["city","state","max_temp_F","rolling_max7","humidity_idx"]],
        on=["city","state"], how="left"
    )
    X = day_all[FEATURES]
    day_all["risk"] = model.predict_proba(X)[:,1]
    return day_all

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

def send_fake_webhook(url, payload):
    if not url:
        return True, "SIMULATED"
    try:
        r = requests.post(url, json=payload, timeout=5)
        return (r.status_code < 400), f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

# ------------------------------------------------------------------
# PAGES
# ------------------------------------------------------------------

if page == "🏠 Dashboard":
    st.title("Real-time-ish Dashboard")

    # auto refresh every 5 seconds
    if st_autorefresh:
        st_autorefresh(interval=5000, key="refresh")

    today = weather["date"].max().date()
    scored = score_day(today)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("# Data Centers", len(dc_inventory))
    with c2:
        st.metric("High-Risk DCs Today (≥0.7)", int((scored["risk"]>=0.7).sum()) if not scored.empty else "—")
    with c3:
        avg_temp = weather[weather["date"].dt.date==today]["max_temp_F"].mean()
        st.metric("Avg Max Temp Today (°F)", f"{avg_temp:.1f}" if not np.isnan(avg_temp) else "—")
    with c4:
        max_roll = weather[weather["date"].dt.date==today]["rolling_max7"].max()
        st.metric("Max Rolling-7 Temp Today (°F)", f"{max_roll:.1f}" if not np.isnan(max_roll) else "—")

    st.write("*(Values refresh every ~5 seconds.)*")
    if not scored.empty:
        st.subheader("Today’s Risk by Data Center")
        st.dataframe(scored[["dc_id","city","state","risk","load_MW","cooling_kW"]]
                     .sort_values("risk", ascending=False).round(3))
    else:
        st.info("No rows for 'today' in the mock set.")

elif page == "📍 Data Centers":
    import pydeck as pdk

    st.title("Data Center Inventory")
    st.write("Static list of mocked DCs, their locations, and average loads.")
    st.dataframe(dc_inventory)

    st.subheader("Map")
    map_df = dc_inventory.dropna(subset=["lat","lon"])
    initial_view = pdk.ViewState(
        latitude=map_df["lat"].mean(),
        longitude=map_df["lon"].mean(),
        zoom=3.5,
        pitch=0
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_radius=70000,
        get_fill_color=[255, 0, 0, 130],
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {"html": "<b>{dc_id}</b><br/>{city}, {state}<br/>Load: {avg_load_MW} MW", "style": {"color": "white"}}
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=initial_view,
        layers=[layer],
        tooltip=tooltip,
    )
    st.pydeck_chart(deck)

    st.subheader("Filter")
    state_pick = st.multiselect("State", sorted(dc_inventory["state"].unique()))
    if state_pick:
        st.dataframe(dc_inventory[dc_inventory["state"].isin(state_pick)])

elif page == "🔮 Scenario Simulator":
    st.title("Scenario Simulator")
    st.markdown("""
Pick a scenario and see what might happen to operations/equipment.

**Scenarios (demo logic):**
- **Extreme Heat**
- **Grid Shortfall**
- **Cooling Failure**
- **Full Outage**
""")
    scen = st.selectbox("Scenario", ["Extreme Heat","Grid Shortfall","Cooling Failure","Full Outage"])
    dc_id_pick = st.selectbox("Data Center", sorted(dc_inventory["dc_id"]))
    severity = st.slider("Severity (0 = minor, 1 = catastrophic)", 0.0, 1.0, 0.6, 0.05)

    st.subheader("Predicted Operational Impacts")
    impacts = []
    if scen == "Extreme Heat":
        impacts = [
            "Cooling power spikes; PUE rises.",
            "Fans/CRAH units near limits; risk of hot spots.",
            "Battery rooms/UPS may derate in high temps.",
            "Component thermal throttling likely."
        ]
    elif scen == "Grid Shortfall":
        impacts = [
            "Mandatory load shedding to meet utility curtailment.",
            "Shift non-critical tasks to other regions.",
            "Run BESS/generators; manage fuel logistics.",
            "Potential SLA breaches if curtailment persists."
        ]
    elif scen == "Cooling Failure":
        impacts = [
            "Room temp climbs; thermal runaway risk.",
            "Immediate workload migration/throttling needed.",
            "Emergency shutdown of affected racks possible.",
            "Inspection/restart procedures post-event."
        ]
    elif scen == "Full Outage":
        impacts = [
            "UPS→generator sequence kicks in.",
            "Runtime limits (fuel/battery) drive priorities.",
            "Graceful shutdown of non-essential systems.",
            "Data integrity & restart sequencing critical."
        ]
    impacts = [f"{imp} (severity {severity:.2f})" for imp in impacts]
    for imp in impacts:
        st.write("- ", imp)

    st.subheader("Suggested Mitigations (demo)")
    plan = make_plan(risk=max(severity,0.7) if scen=="Extreme Heat" else severity,
                     load_mw=25, cooling_kw=3000)
    for p in plan:
        st.write("•", p)

elif page == "🛠️ Action Engine":
    st.title("Automated Response Plan")

    risk = st.slider("Risk Score (0–1)", 0.0, 1.0, 0.78, 0.01)
    load = st.number_input("Current Load (MW)", 0.0, 200.0, 25.0, 0.1)
    cool = st.number_input("Cooling Power (kW)", 0.0, 20000.0, 3000.0, 100.0)

    plan = make_plan(risk, load, cool)

    st.subheader("Recommended Actions")
    for step in plan:
        st.write("- ", step)

    st.markdown("---")
    st.subheader("Simulated Auto-Execution")
    auto = st.checkbox("Send to a (fake) webhook?")
    url = st.text_input("Webhook URL (leave empty to simulate)", value="")
    if auto:
        payload = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "risk": risk,
            "load_MW": load,
            "cooling_kW": cool,
            "actions": plan
        }
        ok, msg = send_fake_webhook(url, payload)
        if ok:
            st.success(f"✅ Actions dispatched. Response: {msg}")
            st.code(json.dumps(payload, indent=2), language="json")
        else:
            st.error(f"❌ Webhook failed: {msg}")

elif page == "📈 Climate & Risk Outlook":
    st.title("Climate & Risk Outlook")

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
