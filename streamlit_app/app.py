import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/prediction_log.csv"

st.set_page_config(
    page_title="Urban Flood Early Warning Dashboard",
    layout="wide"
)

st.title("🌧️ Urban Flood Early Warning System Dashboard")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("📊 Controls")
latest_only = st.sidebar.checkbox("Show latest prediction only", value=True)

# Auto-refresh every 10 seconds
st_autorefresh(interval=10 * 1000, limit=None, key="flood_dashboard")

# -----------------------------
# Latest Prediction & Key Metrics
# -----------------------------
latest = df.iloc[-1]

st.subheader("🚨 Latest Flood Prediction")
col1, col2, col3 = st.columns([1, 1, 2])

# Gauge for Flood Probability
with col1:
    gauge_color = "red" if latest['prediction'] == 1 else "green"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['probability'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Flood Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': gauge_color},
               'steps': [
                   {'range': [0, 50], 'color': "lightgreen"},
                   {'range': [50, 75], 'color': "yellow"},
                   {'range': [75, 100], 'color': "lightcoral"}]}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

# Flood Risk Metric
with col2:
    risk_text = "⚠️ HIGH RISK" if latest['prediction'] == 1 else "✅ Safe"
    risk_color = "red" if latest['prediction'] == 1 else "green"
    st.markdown(f"<h2 style='color:{risk_color};'>{risk_text}</h2>", unsafe_allow_html=True)
    st.write(f"⏱️ Last updated: {latest['timestamp']}")

# Key Weather Features (Bar chart)
with col3:
    key_features = ['rain_1h', 'rain_3h', 'humidity', 'temp', 'wind_speed']
    feature_values = latest[key_features].to_frame(name='Value').reset_index().rename(columns={'index':'Feature'})
    feature_values['Color'] = feature_values['Feature'].apply(
        lambda x: 'red' if latest['prediction']==1 else 'green'
    )
    fig_bar = px.bar(feature_values, x='Feature', y='Value', color='Color', color_discrete_map="identity")
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Flood Probability Trend
# -----------------------------
st.subheader("📈 Flood Probability Trend Over Time")
fig_prob = px.line(df, x='timestamp', y='probability', markers=True,
                   title="Flood Probability Trend")
fig_prob.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Threshold", annotation_position="top right")
st.plotly_chart(fig_prob, use_container_width=True)

# -----------------------------
# Recent Weather Features Heatmap
# -----------------------------
st.subheader("🌡️ Recent Weather Features Heatmap")
feature_cols = ['rain_1h','rain_3h','humidity','temp','wind_speed']
fig_heatmap = px.imshow(df[feature_cols].tail(50).T,
                        labels=dict(x="Time", y="Feature", color="Value"),
                        x=df['timestamp'].tail(50).astype(str),
                        y=feature_cols,
                        aspect="auto",
                        color_continuous_scale='Viridis')
st.plotly_chart(fig_heatmap, use_container_width=True)

# -----------------------------
# Raw Data Display
# -----------------------------
st.subheader("📋 Latest Predictions Data")
st.dataframe(df.tail(10) if latest_only else df)

# -----------------------------
# Footer
# -----------------------------
st.info("Dashboard refreshes automatically every 10 seconds. Scroll up to see live updates.")
