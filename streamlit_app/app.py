import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import requests, os, numpy as np, json
import hashlib # For stable hashing
from datetime import datetime
import psycopg2
from psycopg2 import extras

# -----------------------------
# Config
# -----------------------------
API_URL = os.environ.get("API_URL", "http://flood_api:8000/predict_flood")
DB_HOST = os.environ.get("DB_HOST", "postgres")
DB_NAME = os.environ.get("DB_NAME", "airflow")
DB_USER = os.environ.get("DB_USER", "airflow")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_TABLE = "flood_predictions_log"

LIVE_DATA_PATH = "./data/live_data.csv"
LAST_CALL_FILE = "./data/last_api_call.txt"
ALERT_STATE_FILE = "./data/last_alert_state.txt"
FEATURES_PATH = "./models/feature_columns.json"

TELANGANA_CITIES = {
    "Hyderabad": (17.3850, 78.4867),
    "Warangal": (17.9789, 79.5915),
    "Nizamabad": (18.6720, 78.0941),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514),
    "Mahbubnagar": (16.7435, 78.0081),
    "Suryapet": (17.1500, 79.6167),
    "Adilabad": (19.6667, 78.5333),
    "Ramagundam": (18.7750, 79.4500),
    "Secunderabad": (17.4399, 78.4983),
    "Kothagudem": (17.9392, 80.3150),
    "Sangareddy": (17.8444, 78.0833),
    "Siddipet": (18.1064, 78.8528),
    "Jagtial": (18.7900, 78.9200),
    "Mancherial": (18.8789, 79.4678),
    "Peddapalli": (18.6250, 79.4000),
    "Vikarabad": (17.3333, 77.9167),
    "Bhongir": (17.5144, 78.8953),
    "Kamareddy": (18.3100, 78.3400),
}

FLOOD_THRESHOLD = 0.75

# -----------------------------
# Page setup
# -----------------------------
# -----------------------------
# Page setup & Custom CSS
# -----------------------------
st.set_page_config(page_title="Urban Flood Dashboard", layout="wide", page_icon="🌧️")
st_autorefresh(interval=10 * 1000, key="dashboard_refresh")

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 14px;
        color: #AAAAAA;
    }
    .risk-high { color: #FF4B4B; }
    .risk-safe { color: #00CC96; }
</style>
""", unsafe_allow_html=True)

st.title("🌧 Urban Flood Early Warning System")

# -----------------------------
# Database Query Function
# -----------------------------
@st.cache_data(ttl=5)
def fetch_logs_from_db():
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        query = f"SELECT * FROM {DB_TABLE} ORDER BY timestamp DESC LIMIT 100;"
        df_log = pd.read_sql(query, conn)
        df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], errors="coerce")
        df_log = df_log.sort_values("timestamp", ascending=True).reset_index(drop=True)
        return df_log
    except psycopg2.OperationalError as e:
        st.error(f"🛑 Database Connection Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"🛑 Database Query Error: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_feature_columns():
    if os.path.exists(FEATURES_PATH):
        try:
            with open(FEATURES_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return [
        "temp", "humidity", "pressure", "wind_speed", "rain_1h", "rain_3h",
        "rain_6h", "rain_24h", "rain_intensity", "temp_delta", "humidity_delta",
        "hour", "day", "month", "year",
    ]

def safe_load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except:
            st.warning(f"⚠ Failed to read {path}")
    return pd.DataFrame()

def get_live_prediction(weather_data):
    feature_columns = get_feature_columns()
    payload = {col: weather_data.get(col, 0.0) for col in feature_columns}
    try:
        serializable_payload = {
            k: (float(v) if isinstance(v, np.generic) else v) for k, v in payload.items()
        }
        r = requests.post(API_URL, json=serializable_payload, timeout=8)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API Error: {r.status_code} - {r.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("🛑 API Connection Failed. Is FastAPI running?")
        return None
    except Exception as e:
        st.error(f"Unexpected error during API call: {e}")
        return None

def already_processed(latest_row):
    if not os.path.exists(LAST_CALL_FILE):
        return False
    try:
        comparable = {k: v for k, v in latest_row.items() if k not in ["minute", "second", "timestamp"]}
        # Use stable SHA-256 hash
        row_str = json.dumps(comparable, sort_keys=True)
        current_hash = hashlib.sha256(row_str.encode('utf-8')).hexdigest()
        
        with open(LAST_CALL_FILE) as f:
            return f.read().strip() == current_hash
    except:
        return False

def save_last_processed(latest_row):
    os.makedirs(os.path.dirname(LAST_CALL_FILE), exist_ok=True)
    comparable = {k: v for k, v in latest_row.items() if k not in ["minute", "second", "timestamp"]}
    # Use stable SHA-256 hash
    row_str = json.dumps(comparable, sort_keys=True)
    current_hash = hashlib.sha256(row_str.encode('utf-8')).hexdigest()
    
    with open(LAST_CALL_FILE, "w") as f:
        f.write(current_hash)

def play_alert_sound():
    st.markdown(
        """<audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
        </audio>""",
        unsafe_allow_html=True,
    )

def trigger_high_risk_alert():
    st.toast("🚨 High Flood Risk Detected!", icon="⚠️")
    play_alert_sound()

def check_alert_repeat(current_status):
    if not os.path.exists(ALERT_STATE_FILE):
        with open(ALERT_STATE_FILE, "w") as f:
            f.write(str(current_status))
        return False
    with open(ALERT_STATE_FILE) as f:
        last_state = f.read().strip()
    if last_state == str(current_status):
        return True
    with open(ALERT_STATE_FILE, "w") as f:
        f.write(str(current_status))
    return False

def risk_color(prediction):
    return "red" if prediction == 1 else "green"

# -----------------------------
# Main Dashboard Logic
# -----------------------------
df_live = safe_load_csv(LIVE_DATA_PATH)
latest_weather = df_live.iloc[-1].to_dict() if not df_live.empty else None
df_log = fetch_logs_from_db()

latest_explanation = {}
if latest_weather:
    live_pred = get_live_prediction(latest_weather)
    if live_pred and "flood_risk" in live_pred:
        latest_explanation = live_pred.get("explanation", {})

latest = df_log.iloc[-1] if not df_log.empty else None

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("📊 Controls")
latest_only = st.sidebar.checkbox("Show latest only", True)
show_high_risk_only = st.sidebar.checkbox("Show only HIGH RISK", False)
city_filter = st.sidebar.selectbox("Filter by City", ["All"] + list(TELANGANA_CITIES.keys()))

df_display = df_log.copy()
if show_high_risk_only:
    df_display = df_display[df_display["prediction"] == 1]
if city_filter != "All":
    df_display = df_display[df_display["city"] == city_filter]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📈 Dashboard", "ℹ About Project"])

with tab1:
    if latest is not None:
        latest_prediction = latest.get("prediction", 0)
        latest_probability = latest.get("probability", 0.0)
        
        # Calculate summary metrics (Averaged over all log data)
        avg_rain_24h = df_log['rain_24h'].mean() if 'rain_24h' in df_log.columns and not df_log.empty else 0.0

        # --- 1. KEY METRICS ROW (Custom Cards) ---
        col_A, col_B, col_C, col_D = st.columns(4)

        with col_A:
            risk_class = "risk-high" if latest_prediction == 1 else "risk-safe"
            risk_text = "HIGH RISK" if latest_prediction == 1 else "SAFE"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Status</div>
                <div class="metric-value {risk_class}">{risk_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_B:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Flood Probability</div>
                <div class="metric-value">{latest_probability * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_C:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cities Monitored</div>
                <div class="metric-value">{len(TELANGANA_CITIES)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_D:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg 24h Rain</div>
                <div class="metric-value">{avg_rain_24h:.2f} mm</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- 2. Telangana Flood Map (Prominent) ---
        st.subheader("🗺 Real-Time Flood Risk Map")

        df_base_map = pd.DataFrame(
            [{"city": c, "lat": loc[0], "lon": loc[1], "size_default": 5} for c, loc in TELANGANA_CITIES.items()]
        )

        df_latest_status = df_log.groupby("city").last().reset_index()
        # Use simple gradient for size and color
        df_map = df_base_map.merge(
            df_latest_status[["city", "prediction", "probability"]],
            on="city",
            how="left"
        ).fillna({"prediction": 0, "probability": 0})

        # Dynamic Marker Logic
        df_map["marker_size"] = 10 + (df_map["probability"] * 25) # 10 to 35px
        df_map["risk_label"] = df_map["probability"].apply(lambda x: f"{x*100:.1f}% Risk")

        fig_map = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lon",
            color="probability",
            size="marker_size",
            hover_name="city",
            hover_data={"probability": ":.2f", "prediction": False, "marker_size": False},
            color_continuous_scale=["#00CC96", "#FFD700", "#FF4B4B"], # Green -> Yellow -> Red
            range_color=[0, 1],
            zoom=6.5,
            center={"lat": 17.8, "lon": 79.5},
            mapbox_style="carto-darkmatter",
            height=600,
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.markdown("---")

        # --- Probability Trend ---
        st.subheader("📈 Probability Trend")
        if not df_display.empty:
            fig_prob = px.line(df_display, x="timestamp", y="probability", markers=True, color="city")
            fig_prob.add_hline(y=0.75, line_dash="dot", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("No data to show trend.")

        # --- Logs ---
        st.subheader("📋 Prediction Log (PostgreSQL)")
        df_table = df_display.sort_values("timestamp", ascending=False)
        st.dataframe(df_table.head(10) if latest_only else df_table.style.format({"probability": "{:.1%}"}))
        st.download_button("📥 Download Logs", df_table.to_csv(index=False), "db_prediction_log.csv")

    else:
        st.info("No predictions yet. Please ensure the MLOps pipeline (Airflow/DB) is running.")

# -----------------------------
# About Tab

# -----------------------------

with tab2:
    st.markdown("""
    ## 🌊 Urban Flood Early Warning System (UFEWS)

    The **Urban Flood Early Warning System (UFEWS)** is a **production-grade, AI-driven flood prediction platform** designed to provide **real-time monitoring, risk forecasting, and early alerts** for urban regions.
    By combining **machine learning**, **MLOps automation**, and **live meteorological data**, the system empowers city authorities and disaster management teams to make proactive and data-driven decisions.

    ### 💡 Key Capabilities
    - 🌦 **Live Weather Integration:** Continuously ingests real-time meteorological data from APIs or IoT sensors.
    - 🤖 **ML-Powered Predictions:** Utilizes an optimized Random Forest model served through **FastAPI** for low-latency scoring.
    - 🪶 **Automated Orchestration:** **Apache Airflow** automates end-to-end workflows—data ingestion, prediction, and retraining.
    - 📊 **Interactive Monitoring Dashboard:** Built with **Streamlit** for real-time visualization, analytics, and alerting.
    - 🗺 **Geospatial Intelligence:** Dynamic **Plotly Mapbox** visualization of flood-prone zones (demo region: *Telangana, India*).
    - 🧠 **MLOps Integration:** Continuous model tracking, registry, and versioning powered by **MLflow**.
    - 🚨 **Smart Alert Mechanism:** Multi-level risk alerts with both **visual** and **audio** notifications.
    - 🕒 **Historical Trend Analytics:** Time-series tracking of flood probabilities for situational awareness.

    ### 🧱 Architecture & Tech Stack
    **Tech Stack:**
    `Python` • `Streamlit` • `FastAPI` • `Apache Airflow` • `MLflow` • `PostgreSQL` • `Docker` • `Pandas` • `Scikit-learn` • `Plotly`

    **Core Architecture Highlights:**
    - **FastAPI Microservice:** Real-time model inference and SHAP-based explainability.
    - **PostgreSQL Database:** Centralized data store ensuring auditability and transactional consistency.
    - **Airflow Scheduler:** Automates pipeline execution—data fetch, model prediction, and retraining.
    - **MLflow Tracking Server:** Monitors experiments, parameters, and production model versions.
    - **Streamlit Frontend:** Live dashboard for predictions, insights, and visualization.

    ### 🎯 Objective
    To build a **scalable, automated, and resilient early warning system** capable of providing actionable
    flood insights—reducing urban risk and supporting **sustainable smart city planning**.

    ### 👨‍💻 Developed by
    **Vivek Marri**
    *MLOps Engineer & Data Scientist*
    [GitHub](https://github.com/VIVEK-MARRI) • [Portfolio](https://vivekmarri.com)
    """, unsafe_allow_html=True)

    st.markdown(
        """
        ### 📝 License
        This project is open-source and available under the [MIT License](https://github.com/VIVEK-MARRI/Urban-Flood-Early-Warning-System/blob/main/LICENSE).

        ### 📨 Contact
        For any inquiries or suggestions, please reach out to [Vivek Marri](https://github.com/VIVEK-MARRI).
        """,
        unsafe_allow_html=True,
    )
