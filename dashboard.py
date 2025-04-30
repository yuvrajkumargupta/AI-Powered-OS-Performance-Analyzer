import streamlit as st
import pandas as pd
import psutil
import time
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Configuration
st.set_page_config(
    page_title="AI-Powered OS Performance Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #1E3C72, #2A5298);
        color: white;
        margin-bottom: 20px;
    }
    .anomaly-alert {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,0,0,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255,0,0,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,0,0,0); }
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("⚡ AI-Powered OS Performance Analyzer")
st.markdown("""
    <div style="margin-bottom:30px">
    Real-time system monitoring with AI-powered anomaly detection and predictive analytics
    </div>
""", unsafe_allow_html=True)

# Data Collection
@st.cache_data(ttl=1)
def get_system_metrics():
    return {
        'timestamp': datetime.now(),
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'process_count': len(psutil.pids()),
        'cpu_temp': psutil.sensors_temperatures().get('coretemp', [{}])[0].current if hasattr(psutil, 'sensors_temperatures') else None
    }

# Initialize session state for data storage
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = pd.DataFrame(columns=[
        'timestamp', 'cpu_usage', 'memory_usage', 
        'disk_usage', 'process_count', 'cpu_temp'
    ])

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    refresh_rate = st.slider("Refresh rate (seconds)", 1, 10, 2)
    history_length = st.slider("History length (minutes)", 1, 60, 15)
    enable_ai = st.toggle("Enable AI Analysis", True)

# Live Data Collection
placeholder = st.empty()
while True:
    new_data = get_system_metrics()
    st.session_state.performance_data = pd.concat([
        st.session_state.performance_data,
        pd.DataFrame([new_data])
    ], ignore_index=True)
    
    # Keep only recent data
    cutoff_time = datetime.now() - pd.Timedelta(minutes=history_length)
    st.session_state.performance_data = st.session_state.performance_data[
        st.session_state.performance_data['timestamp'] > cutoff_time
    ]
    
    # AI Analysis
    if enable_ai and len(st.session_state.performance_data) > 10:
        # Anomaly Detection
        model = IsolationForest(contamination=0.1)
        features = st.session_state.performance_data[['cpu_usage', 'memory_usage']]
        anomalies = model.fit_predict(features)
        st.session_state.performance_data['anomaly'] = anomalies == -1
        
        # Simple Prediction
        st.session_state.performance_data['cpu_prediction'] = (
            st.session_state.performance_data['cpu_usage']
            .rolling(5).mean()
            .shift(1)
        )
    
    # Dashboard Update
    with placeholder.container():
        # Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        latest = st.session_state.performance_data.iloc[-1]
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>CPU Usage</h3>
                    <h2>{latest['cpu_usage']:.1f}%</h2>
                    {"🚨 ANOMALY DETECTED" if 'anomaly' in latest and latest['anomaly'] else ""}
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Memory Usage</h3>
                    <h2>{latest['memory_usage']:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Disk Usage</h3>
                    <h2>{latest['disk_usage']:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Processes</h3>
                    <h2>{latest['process_count']}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["CPU Analysis", "Memory Analysis", "System Overview"])
        
        with tab1:
            fig = px.line(
                st.session_state.performance_data,
                x='timestamp',
                y=['cpu_usage', 'cpu_prediction'] if 'cpu_prediction' in st.session_state.performance_data else 'cpu_usage',
                title="CPU Usage Over Time",
                labels={'value': 'Usage (%)'}
            )
            if 'anomaly' in st.session_state.performance_data:
                anomalies = st.session_state.performance_data[st.session_state.performance_data['anomaly']]
                fig.add_scatter(
                    x=anomalies['timestamp'],
                    y=anomalies['cpu_usage'],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Anomaly'
                )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.line(
                st.session_state.performance_data,
                x='timestamp',
                y='memory_usage',
                title="Memory Usage Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.dataframe(
                st.session_state.performance_data.sort_values('timestamp', ascending=False),
                use_container_width=True,
                height=300
            )
    
    time.sleep(refresh_rate)