"""
PhoenixDT API Server

Production-ready REST API with real-time WebSocket streaming.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import numpy as np
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel, Field

from ..core.config import PhoenixConfig
from ..core.digital_twin import DigitalTwin


# Pydantic Models
class SystemStatusResponse(BaseModel):
    """System status response"""

    status: str = Field(..., description="Current system status")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(default="2.0.0", description="System version")
    timestamp: datetime = Field(..., description="Response timestamp")


class DigitalTwinStateResponse(BaseModel):
    """Digital twin state response"""

    timestamp: float
    physical_state: dict[str, float]
    health_metrics: dict[str, float]
    anomalies: list[dict[str, Any]] = Field(default_factory=list)


class FaultInjectionRequest(BaseModel):
    """Fault injection request"""

    fault_type: str = Field(..., description="Type of fault to inject")
    severity: float = Field(..., ge=0, le=1, description="Fault severity (0-1)")


class ControlRequest(BaseModel):
    """Control request"""

    control_mode: str = Field(..., description="Control mode")
    manual_voltage: list[float] | None = Field(
        None, min_items=3, max_items=3, description="Manual 3-phase voltage"
    )


# Global variables
digital_twin: DigitalTwin | None = None


# Create FastAPI app
app = FastAPI(
    title="PhoenixDT API",
    description="Industrial Digital Twin with AI-powered Anomaly Detection and Self-Healing Control",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve simple dashboard"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PhoenixDT Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    </head>
    <body class="bg-gray-900 text-white">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold text-center mb-8">PhoenixDT Dashboard</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div class="bg-gray-800 rounded-lg p-6">
                    <h3 class="text-sm font-medium text-gray-400 mb-2">System Status</h3>
                    <p id="system-status" class="text-2xl font-bold text-green-400">Ready</p>
                </div>
                
                <div class="bg-gray-800 rounded-lg p-6">
                    <h3 class="text-sm font-medium text-gray-400 mb-2">Health Score</h3>
                    <p id="health-score" class="text-2xl font-bold text-blue-400">--%</p>
                </div>
                
                <div class="bg-gray-800 rounded-lg p-6">
                    <h3 class="text-sm font-medium text-gray-400 mb-2">Motor Speed</h3>
                    <p id="motor-speed" class="text-2xl font-bold text-yellow-400">-- RPM</p>
                </div>
                
                <div class="bg-gray-800 rounded-lg p-6">
                    <h3 class="text-sm font-medium text-gray-400 mb-2">Temperature</h3>
                    <p id="temperature" class="text-2xl font-bold text-red-400">--°C</p>
                </div>
            </div>
            
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <div class="bg-gray-800 rounded-lg p-6">
                    <h2 class="text-xl font-bold mb-4">System Control</h2>
                    <div class="space-y-4">
                        <button onclick="startSystem()" class="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg font-medium">Start System</button>
                        <button onclick="stopSystem()" class="w-full bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg font-medium">Stop System</button>
                        <button onclick="injectFault()" class="w-full bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded-lg font-medium">Inject Fault</button>
                    </div>
                </div>
                
                <div class="bg-gray-800 rounded-lg p-6">
                    <h2 class="text-xl font-bold mb-4">Real-time Data</h2>
                    <div id="data-chart"></div>
                </div>
            </div>
        </div>
        
        <script>
            let ws;
            
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = function() {
                    console.log('WebSocket connected');
                    document.getElementById('system-status').textContent = 'Connected';
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                ws.onclose = function() {
                    console.log('WebSocket disconnected');
                    document.getElementById('system-status').textContent = 'Disconnected';
                    setTimeout(connectWebSocket, 5000);
                };
            }
            
            function updateDashboard(data) {
                // Update metrics
                if (data.health_score !== undefined) {
                    document.getElementById('health-score').textContent = Math.round(data.health_score * 100) + '%';
                }
                
                if (data.physical_state !== undefined) {
                    document.getElementById('motor-speed').textContent = Math.round(data.physical_state.speed_rpm || 0) + ' RPM';
                    document.getElementById('temperature').textContent = Math.round(data.physical_state.temperature || 0) + '°C';
                }
                
                // Update chart
                updateChart(data);
            }
            
            function updateChart(data) {
                const trace = {
                    x: [1, 2, 3, 4, 5],
                    y: [
                        data.physical_state?.speed_rpm || 0,
                        data.physical_state?.torque_nm || 0,
                        data.physical_state?.current_a || 0,
                        data.physical_state?.power_w || 0,
                        data.health_score?.overall_health || 0
                    ],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'System Metrics'
                };
                
                const layout = {
                    title: 'Real-time System Metrics',
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Value' },
                    paper_bgcolor: '#1f2937',
                    plot_bgcolor: '#111827',
                    font: { color: 'white' }
                };
                
                Plotly.newPlot('data-chart', [trace], layout);
            }
            
            async function startSystem() {
                try {
                    const response = await axios.post('/api/start');
                    console.log('System started:', response.data);
                } catch (error) {
                    console.error('Failed to start system:', error);
                }
            }
            
            async function stopSystem() {
                try {
                    const response = await axios.post('/api/stop');
                    console.log('System stopped:', response.data);
                } catch (error) {
                    console.error('Failed to stop system:', error);
                }
            }
            
            async function injectFault() {
                try {
                    const response = await axios.post('/api/fault', {
                        fault_type: 'bearing_wear',
                        severity: 0.5
                    });
                    console.log('Fault injected:', response.data);
                } catch (error) {
                    console.error('Failed to inject fault:', error);
                }
            }
            
            // Initialize
            connectWebSocket();
        </script>
    </body>
    </html>
    """


@app.get("/api/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status"""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    return SystemStatusResponse(
        status="running" if digital_twin.is_running else "stopped",
        uptime=0.0,  # Would be calculated from actual uptime
        version="2.0.0",
        timestamp=datetime.now(),
    )


@app.get("/api/state", response_model=DigitalTwinStateResponse)
async def get_digital_twin_state():
    """Get current digital twin state"""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    current_state = digital_twin.get_current_state()
    if not current_state:
        raise HTTPException(status_code=504, detail="Digital twin state not available")

    return DigitalTwinStateResponse(
        timestamp=current_state.timestamp,
        physical_state=current_state.physical_state,
        health_metrics=current_state.health_metrics,
        anomalies=current_state.anomalies,
    )


@app.post("/api/start")
async def start_digital_twin(background_tasks: BackgroundTasks):
    """Start digital twin"""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    if digital_twin.is_running:
        return {"message": "Digital twin is already running"}

    try:
        # Start in background
        async def start_background():
            await digital_twin.start()
        
        background_tasks.add_task(start_background)  # Don't call the function
        
        logger.info("Digital twin start initiated")
        return {"message": "Digital twin start initiated"}
    except Exception as e:
        logger.error(f"Failed to start digital twin: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start digital twin: {e}"
        )


@app.post("/api/stop")
async def stop_digital_twin():
    """Stop digital twin"""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    try:
        await digital_twin.stop()
        logger.info("Digital twin stopped")
        return {"message": "Digital twin stopped successfully"}
    except Exception as e:
        logger.error(f"Failed to stop digital twin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop digital twin: {e}")


@app.post("/api/fault")
async def inject_fault(request: FaultInjectionRequest):
    """Inject fault into system"""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    try:
        # Inject fault using physics simulator
        await digital_twin.simulator.inject_fault(request.fault_type, request.severity)

        logger.info(
            f"Fault injected: {request.fault_type} (severity: {request.severity})"
        )
        return {"message": f"Fault {request.fault_type} injected successfully"}
    except Exception as e:
        logger.error(f"Failed to inject fault: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to inject fault: {e}")


@app.post("/api/control")
async def set_control(request: ControlRequest):
    """Set control parameters"""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    try:
        # Set control mode
        digital_twin.set_control_mode(request.control_mode)

        # Set manual control if provided
        if request.manual_voltage:
            digital_twin.set_manual_control(np.array(request.manual_voltage))

        logger.info(f"Control set to {request.control_mode} mode")
        return {"message": f"Control set to {request.control_mode} mode"}
    except Exception as e:
        logger.error(f"Failed to set control: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set control: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    await websocket.accept()

    try:
        while True:
            if digital_twin and digital_twin.is_running:
                current_state = digital_twin.get_current_state()
                if current_state:
                    # Send state data
                    await websocket.send_text(
                        json.dumps(
                            {
                                "timestamp": current_state.timestamp,
                                "physical_state": current_state.physical_state,
                                "health_metrics": current_state.health_metrics,
                                "anomalies": current_state.anomalies,
                            },
                            default=str,
                        )
                    )

            try:
                await asyncio.sleep(0.1)  # 10Hz update rate
            except asyncio.CancelledError:
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# ... (rest of the code remains the same)

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    global digital_twin

    # Initialize digital twin
    config = PhoenixConfig()
    digital_twin = DigitalTwin(config)

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server"""
    logger.info(f"Starting PhoenixDT API server on {host}:{port}")

    uvicorn.run(
        "phoenixdt.api.app:app", host=host, port=port, reload=reload, log_level="info"
    )


if __name__ == "__main__":
    run_server(reload=True)
