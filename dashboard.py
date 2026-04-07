import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime

# Import environment client (from local path)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from client import GreenhouseEnv
from models import GreenhouseAction

# ─── Configuration ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🌿 Greenhouse Climate Control Monitor",
    page_icon="🌿",
    layout="wide",
)

# Constants
BASE_URL = "http://127.0.0.1:8000"
TASKS = ["maintain_temperature", "optimize_growth", "weather_resilience"]

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "curr_step" not in st.session_state:
    st.session_state.curr_step = 0
if "done" not in st.session_state:
    st.session_state.done = False
if "env" not in st.session_state:
    st.session_state.env = None
if "last_obs" not in st.session_state:
    st.session_state.last_obs = None

# ─── Main Sidebar ────────────────────────────────────────────────────────────

st.sidebar.title("🌿 Greenhouse Control")
st.sidebar.markdown("---")

server_url = st.sidebar.text_input("Server URL", BASE_URL)
task_id = st.sidebar.selectbox("Select Task", TASKS)

col1, col2 = st.sidebar.columns(2)
if col1.button("🔄 Reset Environment", width="stretch"):
    try:
        if st.session_state.env:
            st.session_state.env.close()
        
        env = GreenhouseEnv(base_url=server_url).sync()
        result = env.reset(task_id=task_id)
        
        st.session_state.env = env
        st.session_state.last_obs = result.observation
        st.session_state.history = [result.observation.dict()]
        st.session_state.curr_step = 0
        st.session_state.done = False
        st.success(f"Task '{task_id}' initialized!")
    except Exception as e:
        st.error(f"Failed to reset: {e}")

if col2.button("🚫 Close Env", width="stretch"):
    if st.session_state.env:
        st.session_state.env.close()
        st.session_state.env = None
        st.info("Environment closed.")

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Controls")

h_p = st.sidebar.slider("Heater Power", 0.0, 1.0, 0.3)
v_r = st.sidebar.slider("Ventilation Rate", 0.0, 1.0, 0.1)
h_l = st.sidebar.slider("Humidifier Level", 0.0, 1.0, 0.2)
a_l = st.sidebar.slider("Artificial Light", 0.0, 1.0, 0.0)

if st.sidebar.button("▶️ Step (Manual)", type="primary", width="stretch", disabled=st.session_state.done or st.session_state.env is None):
    action = GreenhouseAction(
        heater_power=h_p,
        ventilation_rate=v_r,
        humidifier_level=h_l,
        artificial_lighting=a_l
    )
    try:
        result = st.session_state.env.step(action)
        st.session_state.last_obs = result.observation
        st.session_state.history.append(result.observation.dict())
        st.session_state.curr_step += 1
        st.session_state.done = result.done
    except Exception as e:
        st.error(f"Step failed: {e}")

# ─── Dashboard Title ──────────────────────────────────────────────────────────

st.title("🌿 Greenhouse Climate Control Monitor")

if st.session_state.last_obs:
    obs = st.session_state.last_obs
    
    # Header Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Temperature", f"{obs.temperature}°C", f"{obs.outside_temperature}°C (out)")
    m2.metric("Humidity", f"{obs.humidity}%", f"{obs.outside_humidity}% (out)")
    m3.metric("Plant Health", f"{obs.plant_health * 100:.1f}%")
    m4.metric("Growth Progress", f"{obs.growth_progress * 100:.2f}%")
    m5.metric("Step Reward", f"{obs.reward:.3f}")

    # Status Info
    with st.expander("Status Message & Metadata", expanded=True):
        st.write(f"**Step:** {obs.step_number} / {obs.max_steps}  |  **Task:** {obs.task_id}")
        st.info(obs.status_message)
        if st.session_state.done:
            st.warning(f"🏁 Episode Complete! Final Score: {obs.metadata.get('grader_score', 'N/A')}")
    
    # ─── Visualizations ───────────────────────────────────────────────────────
    
    # Prepare data for plotting
    df = pd.DataFrame(st.session_state.history)
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Time Series plots
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Climate Trends (Step)", "Resource Usage & Health")
        )
        
        # Row 1: Climate
        fig.add_trace(go.Scatter(y=df['temperature'], name="Temp (°C)", line=dict(color='red', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(y=df['humidity'], name="Humid (%)", line=dict(color='blue', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(y=df['outside_temperature'], name="Outside T", line=dict(color='orange', width=1, dash='dot')), row=1, col=1)
        
        # Row 2: Progress & Energy
        fig.add_trace(go.Scatter(y=df['plant_health'], name="Health", fill='tozeroy', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(y=df['growth_progress'], name="Growth", line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(y=df['energy_consumed_step'] / 4.0, name="Action Energy", line=dict(color='gray', dash='dot')), row=2, col=1)
        
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    with c2:
        # Gauge charts for current state
        # Health Gauge
        fig_health = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = obs.plant_health * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Current Plant Health", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'red'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'lightgreen'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        fig_health.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_health, width="stretch")

        # Reward Gauge
        fig_reward = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = obs.reward,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Current Step Reward", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "gold"},
                'steps': [
                    {'range': [0, 0.5], 'color': 'lightgray'},
                    {'range': [0.5, 0.8], 'color': 'white'},
                    {'range': [0.8, 1.0], 'color': 'cyan'}]}))
        fig_reward.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_reward, width="stretch")

    # CO2 and light
    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("💨 CO₂ Dynamics")
        fig_co2 = go.Figure()
        fig_co2.add_trace(go.Scatter(y=df['co2_level'], fill='tonexty', name="CO2 ppm", line=dict(color='brown')))
        fig_co2.add_hline(y=800, line_dash="dash", annotation_text="Optimal Min", line_color="green")
        fig_co2.add_hline(y=1200, line_dash="dash", annotation_text="Optimal Max", line_color="green")
        fig_co2.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_co2, width="stretch")

    with colB:
        st.subheader("☀️ Light Intensity")
        fig_light = go.Figure()
        fig_light.add_trace(go.Bar(y=df['light_intensity'], name="Photon Flux", marker_color='orange'))
        fig_light.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_light, width="stretch")

else:
    st.info("👈 Connect to the server and reset the environment to start monitoring.")
    st.image("https://images.unsplash.com/photo-1585320806297-9794b3e4eeae?auto=format&fit=crop&q=80&w=1000", caption="Smart Greenhouse System")

# Footer
st.markdown("---")
st.caption("RL Greenhouse Climate Control Dashboard — Built with Streamlit, Plotly and OpenEnv Spec")
