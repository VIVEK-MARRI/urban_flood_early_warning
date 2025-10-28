import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import requests, os, numpy as np, json
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict_flood"
PRED_LOG_PATH = "./data/prediction_log.csv"
LIVE_DATA_PATH = "./data/live_data.csv"
LAST_CALL_FILE = "./data/last_api_call.txt"
ALERT_STATE_FILE = "./data/last_alert_state.txt"
FEATURES_PATH = "./models/feature_columns.json"

TELANGANA_CITIES = {
    "Hyderabad": (17.3850, 78.4867),
    "Warangal": (17.9789, 79.5915),
    "Nizamabad": (18.6720, 78.0941),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514)
}

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Urban Flood Dashboard", layout="wide")
st.title("🌧 Urban Flood Early Warning System")
st_autorefresh(interval=10 * 1000, key="dashboard_refresh") # refresh every 10 sec

# -----------------------------
# Helper functions
# -----------------------------
def safe_load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except:
            st.warning(f"⚠ Failed to read {path}")
    return pd.DataFrame()

def get_feature_columns():
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            return json.load(f)
    return []

def get_live_prediction(weather_data):
    # Load feature columns from model
    feature_columns = get_feature_columns()

    # Ensure all model-required features exist
    for col in feature_columns:
        if col not in weather_data:
            weather_data[col] = 0.0

    # Remove any unknown columns
    weather_data = {k: weather_data[k] for k in weather_data if k in feature_columns}

    try:
        r = requests.post(API_URL, json=weather_data, timeout=8)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def already_processed(latest_row):
    if not os.path.exists(LAST_CALL_FILE):
        return False
    try:
        comparable_data = {k: v for k, v in latest_row.items() if k not in ['minute', 'second', 'timestamp']}
        with open(LAST_CALL_FILE) as f:
            return f.read().strip() == str(hash(frozenset(comparable_data.items())))
    except:
        return False

def save_last_processed(latest_row):
    os.makedirs(os.path.dirname(LAST_CALL_FILE), exist_ok=True)
    comparable_data = {k: v for k, v in latest_row.items() if k not in ['minute', 'second', 'timestamp']}
    with open(LAST_CALL_FILE, "w") as f:
        f.write(str(hash(frozenset(comparable_data.items()))))

def play_alert_sound():
    st.markdown(
        """<audio autoplay><source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg"></audio>""",
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

# NOTE: Removed reset_data_files function as requested

# -----------------------------
# Load logs and ENFORCE DATA TYPES (CRITICAL FIXES)
# -----------------------------
df_log = safe_load_csv(PRED_LOG_PATH)
if not df_log.empty and 'timestamp' in df_log.columns:
    df_log['timestamp'] = pd.to_datetime(df_log['timestamp'], errors='coerce')
    df_log = df_log.dropna(subset=['timestamp'])
    df_log = df_log.sort_values('timestamp').drop_duplicates(subset=['timestamp', 'city'], keep='last')
    
    # CRITICAL FIX: ENSURE LAT/LON AND PROBABILITY ARE NUMERIC
    try:
        df_log['lat'] = pd.to_numeric(df_log['lat'], errors='coerce')
        df_log['lon'] = pd.to_numeric(df_log['lon'], errors='coerce')
        df_log['probability'] = pd.to_numeric(df_log['probability'], errors='coerce')
        df_log['prediction'] = pd.to_numeric(df_log['prediction'], errors='coerce')
        df_log = df_log.dropna(subset=['lat', 'lon', 'probability', 'prediction'])
    except KeyError:
        st.error("Data integrity issue: Missing critical columns (lat/lon/probability/prediction).")
        df_log = pd.DataFrame()


df_live = safe_load_csv(LIVE_DATA_PATH)
latest_weather = df_live.iloc[-1].to_dict() if not df_live.empty else None

# -----------------------------
# Fetch live prediction
# -----------------------------
latest_explanation = {}
if latest_weather and not already_processed(latest_weather):
    live_pred = get_live_prediction(latest_weather)
    if live_pred:
        city = np.random.choice(list(TELANGANA_CITIES.keys()))
        lat, lon = TELANGANA_CITIES[city]
        lat += np.random.uniform(-0.02, 0.02)
        lon += np.random.uniform(-0.02, 0.02)

        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": live_pred.get("flood_risk", 0),
            "probability": live_pred.get("flood_probability", 0.0),
            "lat": lat,
            "lon": lon,
            "city": city,
            "explanation": json.dumps(live_pred.get("explanation", {})) # Store explanation as JSON string
        }
        latest_explanation = live_pred.get("explanation", {}) # Keep the dict for immediate display
        
        new_row.update(latest_weather)
        df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)
        df_log.to_csv(PRED_LOG_PATH, index=False)
        save_last_processed(latest_weather)

# If no new prediction was made, try to retrieve the last explanation from log
elif not df_log.empty and 'explanation' in df_log.columns:
    try:
        # Load the explanation from the last log entry
        last_explanation_str = df_log.iloc[-1].get('explanation', '{}')
        latest_explanation = json.loads(last_explanation_str)
    except:
        latest_explanation = {}


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("📊 Controls")
latest_only = st.sidebar.checkbox("Show latest only", True)
show_high_risk_only = st.sidebar.checkbox("Show only HIGH RISK", False)
city_filter = st.sidebar.selectbox("Filter by City", ["All"] + list(TELANGANA_CITIES.keys()))

# Filter the display dataframe based on user selections
df_display = df_log[df_log['prediction'] == 1] if show_high_risk_only else df_log
if city_filter != "All":
    df_display = df_display[df_display['city'] == city_filter]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📈 Dashboard", "ℹ About Project"])

with tab1:
    if not df_log.empty:
        latest = df_log.iloc[-1] 

        # --- Alert Banner ---
        if latest['prediction'] == 1:
            st.markdown(
                f"<div style='background-color:#cc0000; padding:10px; border-radius:5px; text-align:center;'>"
                f"🚨 HIGH FLOOD RISK! Probability: {latest['probability']*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            if not check_alert_repeat(True):
                trigger_high_risk_alert()
        else:
            st.markdown(
                f"<div style='background-color:#28a745; color:white; padding:12px; border-radius:8px; text-align:center; font-weight:bold;'>"
                f"✅ SAFE. Probability: {latest['probability']*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            check_alert_repeat(False)

        # --- Dashboard Layout ---
        col1, col2, col3 = st.columns([1, 1, 2])
        
        # --- Gauge (Col 1) ---
        with col1:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=latest["probability"] * 100,
                    title={"text": "Flood Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color(latest["prediction"])},
                        "steps": [
                            {"range": [0, 50], "color": "lightgreen"},
                            {"range": [50, 75], "color": "yellow"},
                            {"range": [75, 100], "color": "lightcoral"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # --- Status / Key Features (Col 2 & 3) ---
        with col2:
            color = risk_color(latest["prediction"])
            st.markdown(
                f"<h2 style='color:{color}'>{'⚠ HIGH RISK' if latest['prediction']==1 else '✅ SAFE'}</h2>",
                unsafe_allow_html=True,
            )
            st.write(f"⏱ Updated: {latest['timestamp']}")
        
        with col3:
            key_features = ["rain_1h", "humidity", "temp", "wind_speed"]
            features_avail = [f for f in key_features if f in df_log.columns]
            if features_avail:
                feat_df = latest[features_avail].to_frame("Value").reset_index().rename(columns={"index": "Feature"})
                feat_df["Color"] = risk_color(latest["prediction"])
                fig_bar = px.bar(feat_df, x="Feature", y="Value", color="Color", color_discrete_map="identity")
                st.plotly_chart(fig_bar, use_container_width=True)
                
        st.markdown("---")
        
        # --- V2 SHAP EXPLANATION (New Section) ---
        st.subheader("💡 Prediction Explainability (SHAP)")
        
        if latest_explanation:
            # Convert SHAP dictionary to DataFrame for plotting
            df_explanation = pd.DataFrame(
                list(latest_explanation.items()), columns=['Feature', 'SHAP Value']
            ).sort_values(by='SHAP Value', ascending=False)
            
            # Create a column for positive/negative influence
            df_explanation['Influence'] = np.where(df_explanation['SHAP Value'] > 0, 'Risk Increase', 'Risk Decrease')
            
            # Filter for the top 10 most influential features (by magnitude)
            df_explanation['Abs_SHAP'] = df_explanation['SHAP Value'].abs()
            df_explanation = df_explanation.nlargest(10, 'Abs_SHAP')
            
            fig_explain = px.bar(
                df_explanation.sort_values(by='SHAP Value', ascending=True),
                x='SHAP Value',
                y='Feature',
                orientation='h',
                color='Influence',
                color_discrete_map={'Risk Increase': 'red', 'Risk Decrease': 'blue'},
                title='Top 10 Factors Driving Flood Probability Score'
            )
            fig_explain.update_layout(yaxis={'autorange': "reversed"})
            st.plotly_chart(fig_explain, use_container_width=True)
        else:
            st.info("Waiting for the first prediction with SHAP explanation.")
        
        st.markdown("---")
        
        # --- Map (FIXED FOR CLEAN DISPLAY) ---
        st.subheader("🗺 Telangana Flood Map") 
        
        # 1. Create a Base Map DataFrame with all city locations (The "clean map")
        df_base_map = pd.DataFrame([
            {'city': c, 'lat': loc[0], 'lon': loc[1], 'size': 1, 'color': 'Default', 'status': 'No Data'}
            for c, loc in TELANGANA_CITIES.items()
        ])
        
        # 2. Overlay the latest prediction data only
        df_latest_status = df_log.groupby('city').last().reset_index()
        
        if not df_latest_status.empty:
            df_latest_status["risk_status"] = df_latest_status["prediction"].map({1: "HIGH RISK", 0: "Safe"})
            df_latest_status["size"] = df_latest_status["probability"] * 20 + 5 # Scale size for visualization

            # Merge the latest status onto the base map
            df_map = df_base_map[['city', 'lat', 'lon']].merge(
                df_latest_status[['city', 'lat', 'lon', 'risk_status', 'probability', 'size']],
                on=['city'],
                suffixes=('_base', '_pred'),
                how='left'
            ).fillna({'risk_status': 'No Data', 'probability': 0, 'size': 5})
            
            # Use predicted lat/lon if available, otherwise use base lat/lon
            df_map['lat'] = df_map['lat_pred'].fillna(df_map['lat_base'])
            df_map['lon'] = df_map['lon_pred'].fillna(df_map['lon_base'])
            
            # Map colors for the plot
            color_map = {"HIGH RISK": "red", "Safe": "green", "No Data": "gray"}
            
            fig_map = px.scatter_mapbox(
                df_map,
                lat="lat",
                lon="lon",
                color="risk_status",
                size="size", # Use the scaled size
                hover_data={"city": True, "probability": True, "risk_status": True, "lat": False, "lon": False, "size": False},
                zoom=6,
                center={"lat": 17.8, "lon": 79.5},
                mapbox_style="open-street-map",
                color_discrete_map=color_map,
            )
            # Make the map markers visible
            fig_map.update_traces(marker=dict(size=df_map["size"] * 1.5, opacity=0.9), selector=dict(mode='markers'))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            # Display a basic map if no prediction data exists
            fig_map_base = px.scatter_mapbox(
                df_base_map,
                lat="lat",
                lon="lon",
                hover_name="city",
                zoom=6,
                center={"lat": 17.8, "lon": 79.5},
                mapbox_style="open-street-map"
            )
            st.plotly_chart(fig_map_base, use_container_width=True)


        # --- Probability Trend ---
        st.subheader("📈 Probability Trend")
        if not df_display.empty:
            fig_prob = px.line(df_display, x="timestamp", y="probability", markers=True)
            fig_prob.add_hline(y=0.7, line_dash="dot", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
             st.info("No data to show trend. Run the Airflow DAG to generate predictions.")

        # --- Log Table ---
        st.subheader("📋 Prediction Log")
        st.dataframe(df_display.tail(10) if latest_only else df_display.sort_values('timestamp', ascending=False).style.format({"probability": "{:.1%}"}))
        st.download_button("📥 Download Logs", df_display.to_csv(index=False), "prediction_log.csv")
    else:
        st.info("No predictions yet. Please ensure the FastAPI service and Airflow DAG are running.")

with tab2:
    st.markdown("""
    ## 🌊 Urban Flood Early Warning System

    The **Urban Flood Early Warning System (UFEWS)** is a production-ready platform 
    designed to deliver **real-time flood risk monitoring and early alerts** for urban areas. 
    Powered by machine learning, MLOps pipelines, and live weather data, the system enables 
    proactive disaster management and urban planning.

    ---
    ### 🔍 Core Capabilities
    - 🌦 **Live Weather Integration:** Real-time meteorological data via APIs or IoT sensors 
    - 🤖 **ML-Powered Predictions:** Optimized Random Forest model served via FastAPI 
    - 🪶 **Automated Orchestration:** Airflow DAGs for continuous ingestion & prediction
    - 📊 **Interactive Monitoring Dashboard:** Streamlit-based visualization & analytics 
    - 🗺 **Geospatial Visualization:** Live map tracking of flood-prone areas (Telangana demo) 
    - 🧠 **MLOps Integration:** Experiment tracking & model versioning with MLflow 
    - 🚨 **Smart Alert System:** Adaptive visual and audio alerts 
    - 🕒 **Historical Insights:** Time-series probability tracking for long-term risk assessment 

    ---
    ### 🧩 Tech Stack
    Python • Streamlit • FastAPI • Apache Airflow • MLflow • Docker • Pandas • Scikit-learn 

    ---
    ### 🎯 Objective
    Build a scalable, automated, and data-driven flood early warning system to assist 
    **urban authorities, disaster management teams, and citizens** in making timely, informed decisions. 

    ---
    ### 👨‍💻 Developed By
    **Vivek Marri** *(End-to-end MLOps-driven implementation for real-world resilience analytics)* Connect with me: 
    - 🔗 [LinkedIn](https://www.linkedin.com/in/vivek-marri-49419a274/) 
    - 💻 [GitHub](https://github.com/VIVEK-MARRI) 
    - 📄 [Portfolio](https://vivekmarri.com) 
    """, unsafe_allow_html=True)