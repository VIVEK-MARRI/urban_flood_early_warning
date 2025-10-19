import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import requests
import os
import numpy as np
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict_flood"
PRED_LOG_PATH = "./data/prediction_log.csv"
LIVE_DATA_PATH = "./data/live_data.csv"
LAST_CALL_FILE = "./data/last_api_call.txt"
ALERT_STATE_FILE = "./data/last_alert_state.txt"

# Telangana major cities
TELANGANA_CITIES = {
    "Hyderabad": (17.3850, 78.4867),
    "Warangal": (17.9789, 79.5915),
    "Nizamabad": (18.6720, 78.0941),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514),
    "Mahbubnagar": (16.7435, 78.0081),
    "Suryapet": (17.1500, 79.6167),
    "Adilabad": (19.6667, 78.5333)
}

st.set_page_config(page_title="Urban Flood Early Warning Dashboard", layout="wide")
st.title("🌧️ Urban Flood Early Warning System Dashboard")

# -----------------------------
# Auto-refresh every 10 seconds
# -----------------------------
st_autorefresh(interval=10 * 1000, limit=None, key="flood_dashboard")

# -----------------------------
# Helper functions
# -----------------------------
def safe_load_csv(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            st.warning(f"⚠️ Failed to read {path}. Starting fresh.")
            return pd.DataFrame()
    return pd.DataFrame()

def get_live_prediction(weather_data):
    try:
        response = requests.post(API_URL, json=weather_data, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException:
        st.error("⚠️ API not reachable. Retrying next cycle...")
        return None

def already_processed(latest_row):
    if not os.path.exists(LAST_CALL_FILE):
        return False
    try:
        with open(LAST_CALL_FILE, "r") as f:
            last_row_hash = f.read().strip()
        current_hash = str(hash(frozenset(latest_row.items())))
        return last_row_hash == current_hash
    except Exception:
        return False

def save_last_processed(latest_row):
    try:
        os.makedirs(os.path.dirname(LAST_CALL_FILE), exist_ok=True)
        with open(LAST_CALL_FILE, "w") as f:
            f.write(str(hash(frozenset(latest_row.items()))))
    except Exception:
        pass

def play_alert_sound():
    sound_url = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"
    st.markdown(
        f"""<audio autoplay><source src="{sound_url}" type="audio/ogg"></audio>""",
        unsafe_allow_html=True,
    )

def trigger_high_risk_alert():
    st.toast("🚨 **High Flood Risk Detected!** Stay Alert!", icon="⚠️")
    play_alert_sound()

def check_alert_repeat(current_status):
    if not os.path.exists(ALERT_STATE_FILE):
        with open(ALERT_STATE_FILE, "w") as f:
            f.write(str(current_status))
        return False
    with open(ALERT_STATE_FILE, "r") as f:
        last_state = f.read().strip()
    if last_state == str(current_status):
        return True
    else:
        with open(ALERT_STATE_FILE, "w") as f:
            f.write(str(current_status))
        return False

# -----------------------------
# Load data
# -----------------------------
df_log = safe_load_csv(PRED_LOG_PATH)

# Fix timestamp column: convert all to datetime, drop invalid
if not df_log.empty and 'timestamp' in df_log.columns:
    df_log['timestamp'] = pd.to_datetime(df_log['timestamp'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df_log = df_log.dropna(subset=['timestamp'])
    df_log = df_log.sort_values('timestamp')
    df_log = df_log.drop_duplicates(subset=['timestamp', 'city'], keep='last')

df_live = safe_load_csv(LIVE_DATA_PATH)
latest_weather = df_live.iloc[-1].to_dict() if not df_live.empty else None

# -----------------------------
# Fetch & log live prediction
# -----------------------------
if latest_weather and not already_processed(latest_weather):
    live_pred = get_live_prediction(latest_weather)
    if live_pred:
        city = np.random.choice(list(TELANGANA_CITIES.keys()))
        base_lat, base_lon = TELANGANA_CITIES[city]
        lat = base_lat + np.random.uniform(-0.02, 0.02)
        lon = base_lon + np.random.uniform(-0.02, 0.02)

        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": live_pred.get('flood_risk', 0),
            "probability": live_pred.get('flood_probability', 0.0),
            "lat": lat,
            "lon": lon,
            "city": city
        }
        for k, v in latest_weather.items():
            new_row[k] = v

        df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)
        df_log.to_csv(PRED_LOG_PATH, index=False)
        save_last_processed(latest_weather)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("📊 Controls")
latest_only = st.sidebar.checkbox("Show latest prediction only", value=True)
show_high_risk_only = st.sidebar.checkbox("Show only HIGH RISK events", value=False)

df_display = df_log[df_log['prediction'] == 1] if show_high_risk_only else df_log

# -----------------------------
# Dashboard
# -----------------------------
if not df_display.empty:
    latest = df_display.iloc[-1]
    st.subheader("🚨 Latest Flood Prediction")

    # Alert logic
    if latest['prediction'] == 1:
        st.error(f"⚠️ HIGH FLOOD RISK ALERT! Probability: {latest['probability']*100:.1f}%")
        if not check_alert_repeat(True):
            trigger_high_risk_alert()
    else:
        st.success(f"✅ Safe. Probability: {latest['probability']*100:.1f}%")
        check_alert_repeat(False)

    col1, col2, col3 = st.columns([1, 1, 2])

    # Gauge chart
    with col1:
        gauge_color = "red" if latest['prediction'] == 1 else "green"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest['probability'] * 100,
            title={'text': "Flood Probability (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': gauge_color},
                   'steps': [
                       {'range': [0, 50], 'color': 'lightgreen'},
                       {'range': [50, 75], 'color': 'yellow'},
                       {'range': [75, 100], 'color': 'lightcoral'}]}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Status
    with col2:
        risk_text = "⚠️ HIGH RISK" if latest['prediction'] == 1 else "✅ Safe"
        risk_color = "red" if latest['prediction'] == 1 else "green"
        st.markdown(f"<h2 style='color:{risk_color}'>{risk_text}</h2>", unsafe_allow_html=True)
        st.write(f"⏱️ Last updated: {latest['timestamp']}")

    # Weather features
    with col3:
        key_features = ['rain_1h', 'rain_3h', 'humidity', 'temp', 'wind_speed']
        available_features = [f for f in key_features if f in df_display.columns]
        if available_features:
            feature_values = latest[available_features].to_frame(name='Value').reset_index().rename(columns={'index': 'Feature'})
            feature_values['Color'] = feature_values['Feature'].apply(lambda x: 'red' if latest['prediction'] == 1 else 'green')
            fig_bar = px.bar(feature_values, x='Feature', y='Value', color='Color', color_discrete_map="identity")
            st.plotly_chart(fig_bar, use_container_width=True)

    # Telangana Map
    st.subheader("🗺️ Flood Risk Map")
    df_display_tg = df_display[
        df_display['lat'].between(16.0, 19.7) & df_display['lon'].between(78.0, 81.7)
    ].copy()
    if not df_display_tg.empty:
        df_display_tg['risk_status'] = df_display_tg['prediction'].apply(lambda x: "HIGH RISK" if x == 1 else "Safe")
        fig_map = px.scatter_mapbox(
            df_display_tg,
            lat="lat",
            lon="lon",
            color="risk_status",
            size="probability",
            hover_name="city",
            size_max=20,
            zoom=6,
            center={"lat": 17.8, "lon": 79.5},
            mapbox_style="open-street-map",
            hover_data=["timestamp", "probability", "risk_status"],
            color_discrete_map={"HIGH RISK": "red", "Safe": "green"}
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # Probability trend
    st.subheader("📈 Flood Probability Trend Over Time")
    fig_prob = px.line(df_display, x='timestamp', y='probability', markers=True)
    fig_prob.add_hline(y=0.5, line_dash="dot", line_color="red",
                       annotation_text="Threshold", annotation_position="top right")
    st.plotly_chart(fig_prob, use_container_width=True)

    # Heatmap
    st.subheader("🌡️ Recent Weather Feature Heatmap")
    heatmap_features = [f for f in key_features if f in df_display.columns]
    if heatmap_features:
        fig_heatmap = px.imshow(df_display[heatmap_features].tail(50).T,
                                labels=dict(x="Time", y="Feature", color="Value"),
                                x=df_display['timestamp'].tail(50).astype(str),
                                y=heatmap_features,
                                aspect="auto",
                                color_continuous_scale='Viridis')
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Log table
    st.subheader("📋 Prediction Log")
    st.dataframe(df_display.tail(10) if latest_only else df_display)

else:
    st.info("No predictions yet. Waiting for DAG or API updates...")
