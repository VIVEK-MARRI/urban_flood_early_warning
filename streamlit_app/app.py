import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import requests, os, numpy as np, json
from datetime import datetime
import psycopg2
from psycopg2 import extras

# -----------------------------
# Config
# -----------------------------
API_URL = "http://flood_api:8000/predict_flood"
DB_HOST = os.environ.get("DB_HOST", "postgres")
DB_NAME = os.environ.get("DB_NAME", "airflow")
DB_USER = os.environ.get("DB_USER", "airflow")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "airflow")
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
st.set_page_config(page_title="Urban Flood Dashboard", layout="wide")
st.title("🌧 Urban Flood Early Warning System")
st_autorefresh(interval=10 * 1000, key="dashboard_refresh")

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
        with open(LAST_CALL_FILE) as f:
            return f.read().strip() == str(hash(frozenset(comparable.items())))
    except:
        return False

def save_last_processed(latest_row):
    os.makedirs(os.path.dirname(LAST_CALL_FILE), exist_ok=True)
    comparable = {k: v for k, v in latest_row.items() if k not in ["minute", "second", "timestamp"]}
    with open(LAST_CALL_FILE, "w") as f:
        f.write(str(hash(frozenset(comparable.items()))))

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
        
        # --- 1. HEADER METRICS ROW (NEW) ---
        
        # Calculate summary metrics (Averaged over all log data)
        # Using a small epsilon to prevent division by zero on empty dataframes
        num_cities = len(TELANGANA_CITIES)
        avg_risk_prob = df_log['probability'].mean() if not df_log.empty else 0.0
        avg_rain_24h = df_log['rain_24h'].mean() if 'rain_24h' in df_log.columns and not df_log.empty else 0.0

        col_A, col_B, col_C, col_D = st.columns(4)

        with col_A:
            st.metric(
                label="Current Risk",
                value=f"{latest_probability * 100:.1f}%",
                delta=f"{'HIGH' if latest_prediction == 1 else 'SAFE'}"
            )

        with col_B:
            st.metric(
                label="Cities Monitored",
                value=num_cities
            )

        with col_C:
            st.metric(
                label="Avg 24h Rain (mm)",
                value=f"{avg_rain_24h:.2f}"
            )

        with col_D:
            st.metric(
                label="Avg Risk Probability",
                value=f"{avg_risk_prob * 100:.1f}%"
            )

        st.markdown("---") # Visual separator

        # --- Alert Banner ---
        if latest_prediction == 1:
            st.markdown(
                f"<div style='background-color:#cc0000; padding:10px; border-radius:5px; text-align:center;'>"
                f"🚨 HIGH FLOOD RISK! Probability: {latest_probability*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            if not check_alert_repeat(True):
                trigger_high_risk_alert()
        else:
            st.markdown(
                f"<div style='background-color:#28a745; color:white; padding:12px; border-radius:8px; text-align:center; font-weight:bold;'>"
                f"✅ SAFE. Probability: {latest_probability*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            check_alert_repeat(False)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=latest_probability * 100,
                    title={"text": "Flood Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color(latest_prediction)},
                        "steps": [
                            {"range": [0, 50], "color": "lightgreen"},
                            {"range": [50, 75], "color": "yellow"},
                            {"range": [75, 100], "color": "lightcoral"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            color = risk_color(latest_prediction)
            st.markdown(f"<h2 style='color:{color}'>{'⚠ HIGH RISK' if latest_prediction==1 else '✅ SAFE'}</h2>", unsafe_allow_html=True)
            st.write(f"⏱ Last Prediction Time: {latest.get('timestamp', 'N/A')}")

        with col3:
            key_features = ["rain_24h", "humidity", "temp", "pressure"]
            features_data = {f: latest.get(f) for f in key_features if f in latest}
            if features_data:
                feat_df = pd.DataFrame.from_dict(features_data, orient="index", columns=["Value"]).reset_index().rename(columns={"index": "Feature"})
                feat_df["Color"] = risk_color(latest_prediction)
                fig_bar = px.bar(feat_df, x="Feature", y="Value", color="Color", color_discrete_map="identity")
                st.plotly_chart(fig_bar, use_container_width=True)
       

        # --- Telangana Flood Map ---
        st.markdown("---")
        st.subheader("🗺 Telangana Flood Map")

        df_base_map = pd.DataFrame(
            [{"city": c, "lat": loc[0], "lon": loc[1], "size_default": 5} for c, loc in TELANGANA_CITIES.items()]
        )

        df_latest_status = df_log.groupby("city").last().reset_index()
        df_latest_status["risk_status_pred"] = df_latest_status["prediction"].map({1: "HIGH RISK", 0: "Safe"})
        df_latest_status["size_scaled"] = df_latest_status["probability"] * 20 + 5

        df_map = df_base_map.merge(
            df_latest_status[["city", "risk_status_pred", "size_scaled", "probability"]],
            on="city",
            how="left"
        )

        df_map['risk_level'] = df_map['risk_status_pred'].fillna('No Data')
        df_map['marker_size'] = df_map['size_scaled'].fillna(df_base_map['size_default'])
        df_map['probability_score'] = df_map['probability'].fillna(0)

        color_map = {"HIGH RISK": "red", "Safe": "green", "No Data": "gray"}

        fig_map = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lon",
            color="risk_level",
            size="marker_size",
            hover_data={"city": True, "probability_score": True, "risk_level": True},
            zoom=6,
            center={"lat": 17.8, "lon": 79.5},
            mapbox_style="open-street-map",
            color_discrete_map=color_map,
        )
        fig_map.update_traces(marker=dict(size=df_map["marker_size"] * 1.5, opacity=0.9))
        st.plotly_chart(fig_map, use_container_width=True)

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
