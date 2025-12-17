"""
Interactive Streamlit Dashboard for PhoenixDT

Provides real-time monitoring, control, and visualization of the digital twin
with 3D motor visualization, anomaly detection, and causal analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import asyncio
import time
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from phoenixdt.core.config import Config
from phoenixdt.core.digital_twin import DigitalTwin
from phoenixdt.simulation.motor_simulator import MotorSimulator
from phoenixdt.ml.causal_inference import CausalInference
from phoenixdt.interfaces.opcua_server import OpcUaServer


class PhoenixDTDashboard:
    """Main dashboard application"""

    def __init__(self):
        self.config = Config()
        self.digital_twin = None
        self.causal_inference = CausalInference()
        self.is_running = False

        # Session state initialization
        if "digital_twin" not in st.session_state:
            st.session_state.digital_twin = None
        if "history_data" not in st.session_state:
            st.session_state.history_data = pd.DataFrame()
        if "anomaly_data" not in st.session_state:
            st.session_state.anomaly_data = []
        if "control_mode" not in st.session_state:
            st.session_state.control_mode = "rl"

    def run(self):
        """Run the dashboard"""
        st.set_page_config(
            page_title="PhoenixDT - Digital Twin Dashboard",
            page_icon="🔥",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🔥 PhoenixDT - Failure-Aware Digital Twin")
        st.markdown(
            "*Real-time industrial system monitoring with AI-powered anomaly detection and self-healing control*"
        )

        # Sidebar controls
        self._render_sidebar()

        # Main content
        if st.session_state.digital_twin is None:
            self._render_welcome_page()
        else:
            self._render_main_dashboard()

    def _render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("🎛️ Control Panel")

            # System controls
            st.subheader("System Control")

            if st.button("🚀 Start Digital Twin", type="primary"):
                self._start_digital_twin()

            if st.button("⏹️ Stop Digital Twin"):
                self._stop_digital_twin()

            if st.button("🔄 Reset System"):
                self._reset_system()

            st.divider()

            # Control mode
            st.subheader("Control Mode")
            control_mode = st.selectbox(
                "Select Control Mode",
                ["rl", "pid", "manual"],
                index=["rl", "pid", "manual"].index(st.session_state.control_mode),
            )

            if control_mode != st.session_state.control_mode:
                st.session_state.control_mode = control_mode
                if st.session_state.digital_twin:
                    st.session_state.digital_twin.set_control_mode(control_mode)

            # Manual control (only visible in manual mode)
            if control_mode == "manual":
                st.subheader("Manual Control")
                voltage_a = st.slider("Phase A Voltage (V)", -500, 500, 0)
                voltage_b = st.slider("Phase B Voltage (V)", -500, 500, 0)
                voltage_c = st.slider("Phase C Voltage (V)", -500, 500, 0)

                if st.button("Apply Manual Control"):
                    manual_voltage = np.array([voltage_a, voltage_b, voltage_c])
                    if st.session_state.digital_twin:
                        st.session_state.digital_twin.set_manual_control(manual_voltage)

            st.divider()

            # Fault injection
            st.subheader("Fault Injection")
            fault_type = st.selectbox(
                "Fault Type",
                [
                    "bearing_wear",
                    "lubrication_loss",
                    "overload",
                    "voltage_unbalance",
                    "cooling_failure",
                ],
            )

            severity = st.slider("Severity", 0.0, 1.0, 0.5)

            if st.button("⚠️ Inject Fault"):
                self._inject_fault(fault_type, severity)

            st.divider()

            # System status
            st.subheader("System Status")
            if st.session_state.digital_twin:
                current_state = st.session_state.digital_twin.get_current_state()
                if current_state:
                    st.metric("Status", "🟢 Running")
                    st.metric("Control Mode", st.session_state.control_mode.upper())
                    st.metric("Simulation Time", f"{current_state.timestamp:.2f}s")
                    st.metric(
                        "Health Score",
                        f"{current_state.health_metrics.get('overall_health', 0):.2%}",
                    )
                else:
                    st.metric("Status", "🟡 Starting...")
            else:
                st.metric("Status", "🔴 Stopped")

    def _render_welcome_page(self):
        """Render welcome/setup page"""
        st.markdown("""
        ## Welcome to PhoenixDT! 🔥
        
        PhoenixDT is a cutting-edge industrial digital twin platform that combines:
        
        🧠 **AI-Powered Anomaly Detection** - Real-time failure prediction with uncertainty quantification  
        🎛️ **Self-Healing Control** - Reinforcement learning for adaptive system control  
        🔍 **Causal Inference** - Explainable AI for root cause analysis  
        🏭 **Industrial Integration** - OPC-UA interface for real-world deployment  
        
        ### Getting Started:
        
        1. Click **"🚀 Start Digital Twin"** in the sidebar to initialize the system
        2. Monitor real-time system performance and health metrics
        3. Experiment with different control modes and fault scenarios
        4. Analyze anomalies with causal inference explanations
        
        ### Features:
        
        - **Real-time Monitoring**: Live visualization of motor parameters, bearing health, and system performance
        - **3D Motor Visualization**: Interactive 3D model of the motor system with animated components
        - **Anomaly Detection**: Multi-algorithm ensemble with uncertainty quantification
        - **Causal Analysis**: Root cause analysis with explainable AI
        - **Control Systems**: Compare RL, PID, and manual control strategies
        - **Fault Injection**: Test system resilience with synthetic failure scenarios
        
        Ready to experience the future of industrial AI? Click **Start Digital Twin** to begin!
        """)

        # System architecture diagram
        st.markdown("### System Architecture")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **🧠 AI/ML Layer**
            - Failure Synthesis (VAE)
            - Anomaly Detection
            - Causal Inference
            - RL Controller
            """)

        with col2:
            st.markdown("""
            **⚙️ Simulation Layer**
            - Physics-Based Motor Model
            - Bearing Degradation
            - Thermal Dynamics
            - Vibration Analysis
            """)

        with col3:
            st.markdown("""
            **🌐 Interface Layer**
            - OPC-UA Server
            - REST API
            - Real-time Dashboard
            - Prometheus Metrics
            """)

    def _render_main_dashboard(self):
        """Render main dashboard with all components"""
        # Get current state
        current_state = st.session_state.digital_twin.get_current_state()

        if not current_state:
            st.warning("Digital twin is starting up...")
            return

        # Key metrics row
        self._render_key_metrics(current_state)

        # Tabbed interface
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📊 Real-time Monitoring",
                "🎮 3D Visualization",
                "⚠️ Anomaly Detection",
                "🔍 Causal Analysis",
                "📈 Performance Analytics",
            ]
        )

        with tab1:
            self._render_realtime_monitoring(current_state)

        with tab2:
            self._render_3d_visualization(current_state)

        with tab3:
            self._render_anomaly_detection(current_state)

        with tab4:
            self._render_causal_analysis(current_state)

        with tab5:
            self._render_performance_analytics()

    def _render_key_metrics(self, current_state):
        """Render key performance metrics"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            health_score = current_state.health_metrics.get("overall_health", 0)
            st.metric(
                "🏥 System Health",
                f"{health_score:.1%}",
                delta=f"{health_score - 0.9:.1%}"
                if health_score >= 0.9
                else f"{health_score - 0.9:.1%}",
            )

        with col2:
            efficiency = current_state.physical_state.get("efficiency", 0)
            st.metric(
                "⚡ Efficiency", f"{efficiency:.1%}", delta=f"{efficiency - 0.85:.1%}"
            )

        with col3:
            speed = current_state.physical_state.get("speed_rpm", 0)
            st.metric(
                "🔄 Motor Speed", f"{speed:.0f} RPM", delta=f"{speed - 1800:.0f} RPM"
            )

        with col4:
            anomaly_count = len(current_state.anomalies)
        st.metric(
            "⚠️ Active Anomalies",
            anomaly_count,
            delta=f"+{anomaly_count}" if anomaly_count > 0 else "0",
        )

    def _render_realtime_monitoring(self, current_state):
        """Render real-time monitoring charts"""
        col1, col2 = st.columns(2)

        with col1:
            # Motor parameters
            st.subheader("📊 Motor Parameters")

            # Create time series data
            if len(st.session_state.history_data) > 0:
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "Speed (RPM)",
                        "Torque (Nm)",
                        "Current (A)",
                        "Power (W)",
                    ),
                    vertical_spacing=0.1,
                )

                # Get recent data
                recent_data = st.session_state.history_data.tail(100)

                # Speed
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=recent_data["speed_rpm"],
                        name="Speed",
                        line=dict(color="blue"),
                    ),
                    row=1,
                    col=1,
                )

                # Torque
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=recent_data["torque_nm"],
                        name="Torque",
                        line=dict(color="green"),
                    ),
                    row=1,
                    col=2,
                )

                # Current
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=recent_data["current_a"],
                        name="Current",
                        line=dict(color="red"),
                    ),
                    row=2,
                    col=1,
                )

                # Power
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=recent_data["power_w"],
                        name="Power",
                        line=dict(color="purple"),
                    ),
                    row=2,
                    col=2,
                )

                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Bearing health
            st.subheader("🏥 Bearing Health")

            # Create gauge charts for bearing metrics
            bearing_wear = current_state.physical_state.get("bearing_wear", 0)
            bearing_temp = current_state.physical_state.get("bearing_temp", 25)
            bearing_vibration = current_state.physical_state.get("vibration_mm_s", 0)

            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}],
                ],
                subplot_titles=(
                    "Wear Level",
                    "Temperature (°C)",
                    "Vibration (mm/s)",
                    "Lubrication",
                ),
            )

            # Wear level gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=bearing_wear,
                    domain={"x": [0, 0.5], "y": [0, 0.5]},
                    title={"text": "Wear Level"},
                    gauge={
                        "axis": {"range": [None, 1]},
                        "bar": {"color": "red"},
                        "steps": [
                            {"range": [0, 0.5], "color": "lightgray"},
                            {"range": [0.5, 0.8], "color": "yellow"},
                            {"range": [0.8, 1], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 0.8,
                        },
                    },
                ),
                row=1,
                col=1,
            )

            # Temperature gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=bearing_temp,
                    domain={"x": [0.5, 1], "y": [0, 0.5]},
                    title={"text": "Temperature"},
                    gauge={
                        "axis": {"range": [None, 150]},
                        "bar": {"color": "orange"},
                        "steps": [
                            {"range": [0, 60], "color": "lightgray"},
                            {"range": [60, 100], "color": "yellow"},
                            {"range": [100, 150], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 120,
                        },
                    },
                ),
                row=1,
                col=2,
            )

            # Vibration gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=bearing_vibration,
                    domain={"x": [0, 0.5], "y": [0.5, 1]},
                    title={"text": "Vibration"},
                    gauge={
                        "axis": {"range": [None, 20]},
                        "bar": {"color": "blue"},
                        "steps": [
                            {"range": [0, 5], "color": "lightgray"},
                            {"range": [5, 10], "color": "yellow"},
                            {"range": [10, 20], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 15,
                        },
                    },
                ),
                row=2,
                col=1,
            )

            # Lubrication gauge
            lubrication = st.session_state.digital_twin.simulator.bearing_state.lubrication_quality
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=lubrication,
                    domain={"x": [0.5, 1], "y": [0.5, 1]},
                    title={"text": "Lubrication"},
                    gauge={
                        "axis": {"range": [None, 1]},
                        "bar": {"color": "green"},
                        "steps": [
                            {"range": [0, 0.3], "color": "red"},
                            {"range": [0.3, 0.7], "color": "yellow"},
                            {"range": [0.7, 1], "color": "lightgreen"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 0.3,
                        },
                    },
                ),
                row=2,
                col=2,
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def _render_3d_visualization(self, current_state):
        """Render 3D motor visualization"""
        st.subheader("🎮 3D Motor Visualization")

        # Create 3D motor visualization using plotly
        fig = go.Figure()

        # Motor body (cylinder)
        theta = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(0, 2, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_cylinder = np.cos(theta_grid)
        y_cylinder = np.sin(theta_grid)

        fig.add_trace(
            go.Surface(
                x=x_cylinder,
                y=y_cylinder,
                z=z_grid,
                opacity=0.7,
                colorscale="Blues",
                name="Motor Body",
            )
        )

        # Rotor (rotating based on speed)
        speed = current_state.physical_state.get("speed_rpm", 0)
        rotation_angle = (speed / 60) * 2 * np.pi * current_state.timestamp

        x_rotor = [0, np.cos(rotation_angle)]
        y_rotor = [0, np.sin(rotation_angle)]
        z_rotor = [1, 1]

        fig.add_trace(
            go.Scatter3d(
                x=x_rotor,
                y=y_rotor,
                z=z_rotor,
                mode="lines+markers",
                line=dict(color="red", width=8),
                marker=dict(size=10),
                name="Rotor",
            )
        )

        # Bearing indicators (color based on wear)
        bearing_wear = current_state.physical_state.get("bearing_wear", 0)
        bearing_color = (
            "green" if bearing_wear < 0.3 else "yellow" if bearing_wear < 0.7 else "red"
        )

        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[0, 2],
                mode="markers",
                marker=dict(size=15, color=bearing_color),
                name="Bearings",
            )
        )

        # Vibration visualization
        vibration = current_state.physical_state.get("vibration_mm_s", 0)
        if vibration > 1:
            # Add vibration effect
            vibration_offset = vibration * 0.01
            fig.add_trace(
                go.Scatter3d(
                    x=[vibration_offset, -vibration_offset],
                    y=[vibration_offset, -vibration_offset],
                    z=[1, 1],
                    mode="markers",
                    marker=dict(size=8, color="orange", symbol="diamond"),
                    name="Vibration",
                )
            )

        fig.update_layout(
            title=f"3D Motor Model (Speed: {speed:.0f} RPM)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=800,
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Component status
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Motor Status**")
            st.success("✅ Running" if speed > 0 else "⏸️ Stopped")
            st.info(f"Speed: {speed:.0f} RPM")

        with col2:
            st.markdown("**Bearing Status**")
            if bearing_wear < 0.3:
                st.success("✅ Healthy")
            elif bearing_wear < 0.7:
                st.warning("⚠️ Moderate Wear")
            else:
                st.error("❌ Severe Wear")
            st.info(f"Wear Level: {bearing_wear:.1%}")

        with col3:
            st.markdown("**System Status**")
            health = current_state.health_metrics.get("overall_health", 0)
            if health > 0.8:
                st.success("✅ Optimal")
            elif health > 0.6:
                st.warning("⚠️ Degraded")
            else:
                st.error("❌ Critical")
            st.info(f"Health: {health:.1%}")

    def _render_anomaly_detection(self, current_state):
        """Render anomaly detection interface"""
        st.subheader("⚠️ Anomaly Detection")

        anomalies = current_state.anomalies

        if anomalies:
            st.error(f"🚨 {len(anomalies)} Active Anomalies Detected!")

            for i, anomaly in enumerate(anomalies):
                with st.expander(f"Anomaly {i + 1}: {anomaly.get('type', 'Unknown')}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Severity", f"{anomaly.get('severity', 0):.2f}")
                        st.metric("Confidence", f"{anomaly.get('confidence', 0):.1%}")
                        st.metric("Uncertainty", f"{anomaly.get('uncertainty', 0):.1%}")

                    with col2:
                        st.write("**Detection Methods:**")
                        for method in anomaly.get("detected_by", []):
                            st.write(f"• {method}")

                    st.write("**Details:**")
                    st.json(anomaly.get("details", {}))
        else:
            st.success("✅ No Anomalies Detected")

        # Anomaly history
        if len(st.session_state.anomaly_data) > 0:
            st.subheader("📈 Anomaly History")

            anomaly_df = pd.DataFrame(st.session_state.anomaly_data)

            fig = px.scatter(
                anomaly_df,
                x="timestamp",
                y="severity",
                color="type",
                size="confidence",
                hover_data=["uncertainty", "detected_by"],
                title="Anomaly Detection History",
            )

            st.plotly_chart(fig, use_container_width=True)

    def _render_causal_analysis(self, current_state):
        """Render causal analysis interface"""
        st.subheader("🔍 Causal Analysis")

        # Get current state vector for analysis
        state_vector = st.session_state.digital_twin.simulator.get_state_vector()

        # Create anomaly state for explanation
        anomaly_state = {
            "speed": state_vector[0],
            "torque": state_vector[1],
            "current": state_vector[2],
            "voltage": state_vector[3],
            "bearing_wear": state_vector[4],
            "temperature": state_vector[5],
            "vibration": state_vector[6],
            "lubrication": state_vector[7],
        }

        # Get causal explanation
        explanation = self.causal_inference.explain_anomaly(anomaly_state)

        if "error" not in explanation:
            # Root causes
            st.subheader("🎯 Root Causes")

            root_causes = explanation.get("root_causes", [])
            if root_causes:
                for i, cause in enumerate(root_causes):
                    with st.expander(
                        f"Root Cause {i + 1}: {cause.get('parameter', 'Unknown')}"
                    ):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Importance", f"{cause.get('importance', 0):.3f}")
                            st.metric("Confidence", f"{cause.get('confidence', 0):.1%}")

                        with col2:
                            st.metric(
                                "Effect Size", f"{cause.get('effect_size', 0):.3f}"
                            )
                            st.write("**Target:**", cause.get("target", "Unknown"))

                        st.write("**Mechanism:**")
                        st.info(cause.get("mechanism", "No mechanism available"))
            else:
                st.info("No significant root causes identified")

            # Recommendations
            st.subheader("💡 Recommendations")

            recommendations = explanation.get("recommendations", [])
            if recommendations:
                for i, rec in enumerate(recommendations):
                    st.write(f"{i + 1}. {rec}")
            else:
                st.info("No specific recommendations at this time")

            # Causal pathways
            st.subheader("🛤️ Causal Pathways")

            pathways = explanation.get("causal_pathways", [])
            if pathways:
                for pathway in pathways:
                    st.write(
                        f"**{pathway.get('root_cause', 'Unknown')}** → {' → '.join(pathway.get('pathway', []))}"
                    )
                    if pathway.get("final_effects"):
                        st.write(f"   Effects: {', '.join(pathway['final_effects'])}")
            else:
                st.info("No causal pathways identified")
        else:
            st.warning("Causal analysis not available - model not trained")

    def _render_performance_analytics(self):
        """Render performance analytics"""
        st.subheader("📈 Performance Analytics")

        if st.session_state.digital_twin:
            # Get performance summary
            summary = st.session_state.digital_twin.get_performance_summary()

            if summary:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Avg Efficiency", f"{summary.get('avg_efficiency', 0):.1%}"
                    )

                with col2:
                    st.metric(
                        "Min Efficiency", f"{summary.get('min_efficiency', 0):.1%}"
                    )

                with col3:
                    st.metric("Avg Health", f"{summary.get('avg_health', 0):.1%}")

                with col4:
                    st.metric("Total Anomalies", summary.get("total_anomalies", 0))

                # Performance over time
                if len(st.session_state.history_data) > 0:
                    st.subheader("📊 Performance Trends")

                    recent_data = st.session_state.history_data.tail(200)

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        subplot_titles=(
                            "Efficiency Over Time",
                            "Health Score Over Time",
                        ),
                        vertical_spacing=0.1,
                    )

                    # Efficiency trend
                    fig.add_trace(
                        go.Scatter(
                            x=recent_data.index,
                            y=recent_data["efficiency"],
                            name="Efficiency",
                            line=dict(color="blue"),
                        ),
                        row=1,
                        col=1,
                    )

                    # Health trend (if available)
                    if "health_score" in recent_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=recent_data.index,
                                y=recent_data["health_score"],
                                name="Health Score",
                                line=dict(color="green"),
                            ),
                            row=2,
                            col=1,
                        )

                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data available yet")
        else:
            st.info("Start the digital twin to see performance analytics")

    def _start_digital_twin(self):
        """Start the digital twin"""
        try:
            # Initialize digital twin
            st.session_state.digital_twin = DigitalTwin(self.config)

            # Start simulation in background
            asyncio.create_task(self._run_simulation())

            st.success("🚀 Digital Twin started successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to start digital twin: {e}")

    def _stop_digital_twin(self):
        """Stop the digital twin"""
        if st.session_state.digital_twin:
            st.session_state.digital_twin.stop()
            st.session_state.digital_twin = None
            st.success("⏹️ Digital Twin stopped")
            st.rerun()

    def _reset_system(self):
        """Reset the digital twin"""
        if st.session_state.digital_twin:
            st.session_state.digital_twin.reset()
            st.session_state.history_data = pd.DataFrame()
            st.session_state.anomaly_data = []
            st.success("🔄 System reset successfully")
            st.rerun()

    def _inject_fault(self, fault_type: str, severity: float):
        """Inject a fault into the system"""
        if st.session_state.digital_twin:
            try:
                st.session_state.digital_twin.simulator.inject_fault(
                    fault_type, severity
                )
                st.success(f"⚠️ Fault injected: {fault_type} (severity: {severity:.2f})")
            except Exception as e:
                st.error(f"Failed to inject fault: {e}")
        else:
            st.warning("Start the digital twin first")

    async def _run_simulation(self):
        """Run simulation in background"""
        if not st.session_state.digital_twin:
            return

        try:
            # Start simulation
            await st.session_state.digital_twin.start(duration=3600)  # Run for 1 hour

            # Update dashboard data periodically
            while (
                st.session_state.digital_twin
                and st.session_state.digital_twin.is_running
            ):
                current_state = st.session_state.digital_twin.get_current_state()

                if current_state:
                    # Update history data
                    new_row = {
                        "timestamp": current_state.timestamp,
                        "speed_rpm": current_state.physical_state.get("speed_rpm", 0),
                        "torque_nm": current_state.physical_state.get("torque_nm", 0),
                        "current_a": current_state.physical_state.get("current_a", 0),
                        "voltage_v": current_state.physical_state.get("voltage_v", 0),
                        "power_w": current_state.physical_state.get("power_w", 0),
                        "efficiency": current_state.physical_state.get("efficiency", 0),
                        "bearing_wear": current_state.physical_state.get(
                            "bearing_wear", 0
                        ),
                        "bearing_temp": current_state.physical_state.get(
                            "bearing_temp", 0
                        ),
                        "vibration_mm_s": current_state.physical_state.get(
                            "vibration_mm_s", 0
                        ),
                        "health_score": current_state.health_metrics.get(
                            "overall_health", 0
                        ),
                    }

                    st.session_state.history_data = pd.concat(
                        [st.session_state.history_data, pd.DataFrame([new_row])],
                        ignore_index=True,
                    )

                    # Update anomaly data
                    if current_state.anomalies:
                        for anomaly in current_state.anomalies:
                            anomaly_record = {
                                "timestamp": current_state.timestamp,
                                "type": anomaly.get("type", "Unknown"),
                                "severity": anomaly.get("severity", 0),
                                "confidence": anomaly.get("confidence", 0),
                                "uncertainty": anomaly.get("uncertainty", 0),
                                "detected_by": ", ".join(
                                    anomaly.get("detected_by", [])
                                ),
                            }
                            st.session_state.anomaly_data.append(anomaly_record)

                await asyncio.sleep(1)  # Update every second

        except Exception as e:
            st.error(f"Simulation error: {e}")


def main():
    """Main dashboard entry point"""
    dashboard = PhoenixDTDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
