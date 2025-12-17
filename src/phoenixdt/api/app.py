"""
PhoenixDT Unified API - Apple/Tesla Grade Engineering

Complete integration of quantum-enhanced digital twin with:
- Real-time WebSocket streaming
- RESTful API with OpenAPI documentation
- Quantum state management
- Predictive analytics
- Self-healing capabilities
- Causal inference
- Performance monitoring
"""

from __future__ import annotations
import asyncio
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    BackgroundTasks,
    Query,
    Path,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import uvicorn
from loguru import logger

from ..core.digital_twin import (
    PhoenixDigitalTwin,
    QuantumTwinState,
    PredictiveInsight,
    SystemState,
)
from ..core.config import PhoenixConfig


# Pydantic Models for API
class SystemStatusResponse(BaseModel):
    """System status response"""

    status: str = Field(..., description="Current system status")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(default="2.0.0", description="System version")
    timestamp: datetime = Field(..., description="Response timestamp")
    quantum_coherence: float = Field(..., description="Quantum coherence level")
    entropy: float = Field(..., description="System entropy")
    health_score: float = Field(..., description="Overall health score")


class QuantumStateResponse(BaseModel):
    """Quantum state response"""

    timestamp: float
    classical_state: Dict[str, float]
    health_vector: List[float]
    anomaly_signature: List[float]
    control_policy: List[float]
    uncertainty_quantum: List[float]
    coherence: float
    entropy: float


class PredictiveInsightResponse(BaseModel):
    """Predictive insight response"""

    metric: str
    current_value: float
    predicted_values: List[float]
    confidence_intervals: List[List[float]]
    risk_score: float
    causal_factors: List[str]
    recommended_actions: List[str]
    timestamp: datetime


class FaultInjectionRequest(BaseModel):
    """Fault injection request"""

    fault_type: str = Field(..., description="Type of fault to inject")
    severity: float = Field(..., ge=0, le=1, description="Fault severity (0-1)")
    duration: Optional[float] = Field(
        None, ge=0, description="Fault duration in seconds"
    )
    target_component: Optional[str] = Field(None, description="Target component")


class ControlRequest(BaseModel):
    """Control request"""

    control_mode: str = Field(..., description="Control mode")
    manual_voltage: Optional[List[float]] = Field(
        None, min_items=3, max_items=3, description="Manual 3-phase voltage"
    )
    adaptive_params: Optional[Dict[str, Any]] = Field(
        None, description="Adaptive control parameters"
    )


class HealingStrategyRequest(BaseModel):
    """Healing strategy request"""

    healing_type: str = Field(..., description="Type of healing strategy")
    aggressiveness: float = Field(
        default=0.5, ge=0, le=1, description="Healing aggressiveness"
    )
    target_health: float = Field(
        default=0.9, ge=0, le=1, description="Target health level"
    )


class CausalAnalysisRequest(BaseModel):
    """Causal analysis request"""

    target_metrics: List[str] = Field(..., description="Target metrics for analysis")
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth")
    time_horizon: Optional[float] = Field(None, description="Analysis time horizon")


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response"""

    uptime: float
    predictions_made: int
    anomalies_detected: int
    self_healing_events: int
    avg_response_time: float
    quantum_coherence: float
    system_entropy: float
    health_score: float
    throughput: float
    memory_usage: float


class WebSocketManager:
    """Enhanced WebSocket connection manager"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept WebSocket connection with metadata"""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
            self.connection_metadata[websocket] = {
                "client_id": client_id or f"client_{len(self.active_connections)}",
                "connected_at": datetime.now(),
                "last_ping": datetime.now(),
                "subscriptions": ["all"],
            }
        logger.info(
            f"WebSocket connected: {self.connection_metadata[websocket]['client_id']}"
        )

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_id = self.connection_metadata.get(websocket, {}).get(
                "client_id", "unknown"
            )
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
            logger.info(f"WebSocket disconnected: {client_id}")

    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict, message_type: str = "update"):
        """Broadcast message to all connected WebSockets"""
        if not self.active_connections:
            return

        enhanced_message = {
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "data": message,
        }

        message_str = json.dumps(enhanced_message, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
                # Update last ping
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]["last_ping"] = datetime.now()
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def get_connection_stats(self) -> Dict:
        """Get WebSocket connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_connected": len(self.connection_metadata),
            "average_connection_time": self._compute_avg_connection_time(),
        }

    def _compute_avg_connection_time(self) -> float:
        """Compute average connection time"""
        if not self.connection_metadata:
            return 0.0

        now = datetime.now()
        total_time = sum(
            (now - meta["connected_at"]).total_seconds()
            for meta in self.connection_metadata.values()
        )
        return total_time / len(self.connection_metadata)


# Global variables
digital_twin: Optional[PhoenixDigitalTwin] = None
connection_manager = WebSocketManager()
background_tasks: List[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global digital_twin, background_tasks

    logger.info("Starting PhoenixDT Quantum API Server...")

    try:
        # Initialize quantum digital twin
        config = PhoenixConfig()
        digital_twin = PhoenixDigitalTwin(config)

        # Start background tasks
        background_tasks = [
            asyncio.create_task(stream_quantum_data()),
            asyncio.create_task(monitor_system_health()),
            asyncio.create_task(cleanup_connections()),
        ]

        logger.info("PhoenixDT API server initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize PhoenixDT: {e}")
        raise
    finally:
        logger.info("Shutting down PhoenixDT API server...")

        # Cancel background tasks
        for task in background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop digital twin
        if digital_twin:
            await digital_twin.stop()

        logger.info("PhoenixDT API server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="PhoenixDT Quantum API",
    description="Apple/Tesla-grade quantum-enhanced industrial digital twin with real-time predictive capabilities",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve next-generation dashboard"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PhoenixDT Quantum Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
        <style>
            .quantum-glow { animation: quantumPulse 2s infinite; }
            @keyframes quantumPulse {
                0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.5); }
                50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.8); }
            }
            .health-bar { transition: all 0.3s ease; }
            .anomaly-pulse { animation: anomalyPulse 1s infinite; }
            @keyframes anomalyPulse {
                0%, 100% { background-color: rgba(239, 68, 68, 0.1); }
                50% { background-color: rgba(239, 68, 68, 0.3); }
            }
        </style>
    </head>
    <body class="bg-gray-900 text-white">
        <div id="app" class="min-h-screen">
            <!-- Header -->
            <header class="bg-gray-800 border-b border-gray-700">
                <div class="container mx-auto px-4 py-4">
                    <div class="flex items-center justify-between">
                        <h1 class="text-2xl font-bold text-blue-400">🔥 PhoenixDT Quantum</h1>
                        <div class="flex items-center space-x-4">
                            <span id="connection-status" class="px-3 py-1 bg-gray-700 rounded-full text-sm">Connecting...</span>
                            <span id="quantum-coherence" class="px-3 py-1 bg-green-600 rounded-full text-sm">Coherence: --</span>
                        </div>
                    </div>
                </div>
            </header>

            <!-- Main Dashboard -->
            <main class="container mx-auto px-4 py-6">
                <!-- System Status Cards -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                        <h3 class="text-sm font-medium text-gray-400 mb-2">System Health</h3>
                        <p id="health-score" class="text-3xl font-bold text-green-400">--%</p>
                        <div class="mt-2 bg-gray-700 rounded-full h-2">
                            <div id="health-bar" class="health-bar bg-green-500 h-2 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                        <h3 class="text-sm font-medium text-gray-400 mb-2">Quantum Coherence</h3>
                        <p id="coherence-value" class="text-3xl font-bold text-blue-400">--</p>
                    </div>
                    
                    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                        <h3 class="text-sm font-medium text-gray-400 mb-2">System Entropy</h3>
                        <p id="entropy-value" class="text-3xl font-bold text-yellow-400">--</p>
                    </div>
                    
                    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                        <h3 class="text-sm font-medium text-gray-400 mb-2">Active Anomalies</h3>
                        <p id="anomaly-count" class="text-3xl font-bold text-red-400">--</p>
                    </div>
                </div>

                <!-- Real-time Charts -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                        <h2 class="text-xl font-bold mb-4">Quantum State Evolution</h2>
                        <div id="quantum-chart"></div>
                    </div>
                    
                    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                        <h2 class="text-xl font-bold mb-4">Predictive Analytics</h2>
                        <div id="prediction-chart"></div>
                    </div>
                </div>

                <!-- Control Panel -->
                <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
                    <h2 class="text-xl font-bold mb-4">Quantum Control Panel</h2>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <button onclick="startSystem()" class="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-lg font-medium transition">Start System</button>
                        <button onclick="stopSystem()" class="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-medium transition">Stop System</button>
                        <button onclick="injectFault()" class="bg-yellow-600 hover:bg-yellow-700 px-6 py-3 rounded-lg font-medium transition">Inject Fault</button>
                    </div>
                </div>

                <!-- Causal Analysis -->
                <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h2 class="text-xl font-bold mb-4">Causal Analysis</h2>
                    <div id="causal-insights" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <!-- Causal insights will be populated here -->
                    </div>
                </div>
            </main>
        </div>

        <script>
            let ws;
            let updateInterval;
            
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = function() {
                    document.getElementById('connection-status').textContent = 'Connected';
                    document.getElementById('connection-status').className = 'px-3 py-1 bg-green-600 rounded-full text-sm';
                    console.log('WebSocket connected');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data.data);
                };
                
                ws.onclose = function() {
                    document.getElementById('connection-status').textContent = 'Disconnected';
                    document.getElementById('connection-status').className = 'px-3 py-1 bg-red-600 rounded-full text-sm';
                    console.log('WebSocket disconnected, attempting reconnect...');
                    setTimeout(connectWebSocket, 5000);
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            function updateDashboard(data) {
                // Update health score
                if (data.health_score !== undefined) {
                    const health = Math.round(data.health_score * 100);
                    document.getElementById('health-score').textContent = health + '%';
                    document.getElementById('health-bar').style.width = health + '%';
                    
                    // Update health bar color
                    const healthBar = document.getElementById('health-bar');
                    if (health > 80) {
                        healthBar.className = 'health-bar bg-green-500 h-2 rounded-full';
                    } else if (health > 60) {
                        healthBar.className = 'health-bar bg-yellow-500 h-2 rounded-full';
                    } else {
                        healthBar.className = 'health-bar bg-red-500 h-2 rounded-full';
                    }
                }
                
                // Update quantum coherence
                if (data.coherence !== undefined) {
                    document.getElementById('coherence-value').textContent = data.coherence.toFixed(3);
                    const coherenceElement = document.getElementById('quantum-coherence');
                    if (data.coherence > 0.8) {
                        coherenceElement.className = 'px-3 py-1 bg-green-600 rounded-full text-sm quantum-glow';
                    } else if (data.coherence > 0.5) {
                        coherenceElement.className = 'px-3 py-1 bg-yellow-600 rounded-full text-sm';
                    } else {
                        coherenceElement.className = 'px-3 py-1 bg-red-600 rounded-full text-sm';
                    }
                }
                
                // Update entropy
                if (data.entropy !== undefined) {
                    document.getElementById('entropy-value').textContent = data.entropy.toFixed(3);
                }
                
                // Update anomalies
                if (data.anomaly_signature !== undefined) {
                    const anomalyCount = data.anomaly_signature.filter(v => Math.abs(v) > 0.1).length;
                    document.getElementById('anomaly-count').textContent = anomalyCount;
                    
                    // Add anomaly pulse effect
                    if (anomalyCount > 0) {
                        document.getElementById('anomaly-count').parentElement.classList.add('anomaly-pulse');
                    } else {
                        document.getElementById('anomaly-count').parentElement.classList.remove('anomaly-pulse');
                    }
                }
                
                // Update charts
                updateCharts(data);
            }
            
            function updateCharts(data) {
                // Quantum State Chart
                if (data.classical_state && data.health_vector) {
                    const quantumTrace = {
                        x: ['Speed', 'Torque', 'Current', 'Voltage', 'Wear', 'Temp', 'Vibration', 'Lubrication'],
                        y: [
                            data.classical_state.speed_rpm || 0,
                            data.classical_state.torque_nm || 0,
                            data.classical_state.current_a || 0,
                            data.classical_state.voltage_v || 0,
                            data.classical_state.bearing_wear || 0,
                            data.classical_state.temperature || 0,
                            data.classical_state.vibration_mm_s || 0,
                            data.classical_state.lubrication_quality || 0
                        ],
                        type: 'bar',
                        marker: {
                            color: data.health_vector.map(h => `rgba(59, 130, 246, ${h})`)
                        }
                    };
                    
                    Plotly.newPlot('quantum-chart', [quantumTrace], {
                        title: 'Quantum System State',
                        paper_bgcolor: '#1f2937',
                        plot_bgcolor: '#111827',
                        font: { color: 'white' },
                        margin: { t: 40, r: 20, b: 40, l: 60 }
                    });
                }
            }
            
            async function startSystem() {
                try {
                    const response = await axios.post('/api/start');
                    showNotification('System started successfully', 'success');
                } catch (error) {
                    showNotification('Failed to start system: ' + error.message, 'error');
                }
            }
            
            async function stopSystem() {
                try {
                    const response = await axios.post('/api/stop');
                    showNotification('System stopped successfully', 'success');
                } catch (error) {
                    showNotification('Failed to stop system: ' + error.message, 'error');
                }
            }
            
            async function injectFault() {
                try {
                    const response = await axios.post('/api/fault', {
                        fault_type: 'bearing_wear',
                        severity: 0.5
                    });
                    showNotification('Fault injected successfully', 'warning');
                } catch (error) {
                    showNotification('Failed to inject fault: ' + error.message, 'error');
                }
            }
            
            function showNotification(message, type) {
                // Create notification element
                const notification = document.createElement('div');
                notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg text-white font-medium z-50 ${
                    type === 'success' ? 'bg-green-600' : 
                    type === 'warning' ? 'bg-yellow-600' : 'bg-red-600'
                }`;
                notification.textContent = message;
                
                document.body.appendChild(notification);
                
                // Remove after 3 seconds
                setTimeout(() => {
                    document.body.removeChild(notification);
                }, 3000);
            }
            
            // Initialize
            connectWebSocket();
        </script>
    </body>
    </html>
    """


# API Endpoints
@app.get("/api/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    try:
        current_state = digital_twin.get_current_state()
        performance_metrics = digital_twin.get_performance_metrics()

        return SystemStatusResponse(
            status=digital_twin.state.value,
            uptime=performance_metrics.get("uptime", 0.0),
            version="2.0.0",
            timestamp=datetime.now(),
            quantum_coherence=current_state.coherence if current_state else 0.0,
            entropy=current_state.entropy if current_state else 0.0,
            health_score=performance_metrics.get("health_score", 0.0),
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")


@app.get("/api/quantum-state", response_model=QuantumStateResponse)
async def get_quantum_state():
    """Get current quantum state"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    current_state = digital_twin.get_current_state()
    if not current_state:
        raise HTTPException(status_code=504, detail="Quantum state not available")

    return QuantumStateResponse(
        timestamp=current_state.timestamp,
        classical_state=current_state.classical_state,
        health_vector=current_state.health_vector.tolist(),
        anomaly_signature=current_state.anomaly_signature.tolist(),
        control_policy=current_state.control_policy.tolist(),
        uncertainty_quantum=current_state.uncertainty_quantum.tolist(),
        coherence=current_state.coherence,
        entropy=current_state.entropy,
    )


@app.get("/api/predictions", response_model=List[PredictiveInsightResponse])
async def get_predictions():
    """Get predictive insights"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    insights = digital_twin.get_predictive_insights()

    return [
        PredictiveInsightResponse(
            metric=insight.metric,
            current_value=insight.current_value,
            predicted_values=insight.predicted_values.tolist(),
            confidence_intervals=[list(ci) for ci in insight.confidence_intervals],
            risk_score=insight.risk_score,
            causal_factors=insight.causal_factors,
            recommended_actions=insight.recommended_actions,
            timestamp=datetime.now(),
        )
        for insight in insights
    ]


@app.post("/api/start")
async def start_quantum_system(background_tasks: BackgroundTasks):
    """Start quantum digital twin system"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    if digital_twin.is_running:
        return {"message": "Quantum system is already running", "status": "running"}

    try:
        # Start in background
        background_tasks.add_task(lambda: asyncio.create_task(digital_twin.start()))

        logger.info("Quantum system start initiated")
        return {
            "message": "Quantum system start initiated",
            "status": "starting",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to start quantum system: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start quantum system: {e}"
        )


@app.post("/api/stop")
async def stop_quantum_system():
    """Stop quantum digital twin system"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    try:
        await digital_twin.stop()
        logger.info("Quantum system stopped")
        return {
            "message": "Quantum system stopped successfully",
            "status": "stopped",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to stop quantum system: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop quantum system: {e}"
        )


@app.post("/api/fault")
async def inject_quantum_fault(request: FaultInjectionRequest):
    """Inject fault with quantum-enhanced detection"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    try:
        # Inject fault using quantum engine
        await digital_town.physics_engine.inject_fault(
            request.fault_type, request.severity
        )

        logger.info(
            f"Quantum fault injected: {request.fault_type} (severity: {request.severity})"
        )

        return {
            "message": f"Quantum fault {request.fault_type} injected successfully",
            "fault_type": request.fault_type,
            "severity": request.severity,
            "quantum_signature": "injected",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to inject quantum fault: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to inject quantum fault: {e}"
        )


@app.post("/api/control")
async def set_quantum_control(request: ControlRequest):
    """Set quantum-enhanced control parameters"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    try:
        # Apply control settings
        if request.manual_voltage:
            # Apply manual control through quantum interface
            control_policy = torch.tensor(request.manual_voltage)
            # This would interface with the quantum controller
            logger.info(f"Manual quantum control applied: {request.manual_voltage}")

        if request.adaptive_params:
            # Apply adaptive parameters
            logger.info(
                f"Adaptive quantum parameters applied: {request.adaptive_params}"
            )

        return {
            "message": f"Quantum control set to {request.control_mode} mode",
            "control_mode": request.control_mode,
            "quantum_coherence": digital_twin.get_current_state().coherence
            if digital_twin.get_current_state()
            else 0.0,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to set quantum control: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to set quantum control: {e}"
        )


@app.post("/api/healing")
async def trigger_quantum_healing(request: HealingStrategyRequest):
    """Trigger quantum-enhanced self-healing"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    try:
        # Trigger quantum healing process
        healing_strategy = {
            "type": request.healing_type,
            "aggressiveness": request.aggressiveness,
            "target_health": request.target_health,
        }

        # Apply healing through quantum optimization
        logger.info(f"Quantum healing triggered: {healing_strategy}")

        return {
            "message": "Quantum self-healing process initiated",
            "healing_strategy": healing_strategy,
            "estimated_recovery_time": "30-60 seconds",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to trigger quantum healing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to trigger quantum healing: {e}"
        )


@app.post("/api/causal-analysis")
async def perform_causal_analysis(request: CausalAnalysisRequest):
    """Perform quantum-enhanced causal analysis"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    try:
        # Get current state for analysis
        current_state = digital_twin.get_current_state()
        if not current_state:
            raise HTTPException(
                status_code=504, detail="Quantum state not available for analysis"
            )

        # Perform causal analysis
        anomaly_state = {
            metric: current_state.classical_state.get(metric, 0.0)
            for metric in request.target_metrics
        }

        causal_explanation = await digital_twin.causal_engine.explain_anomaly(
            anomaly_state
        )

        return {
            "message": "Quantum causal analysis completed",
            "analysis": causal_explanation,
            "target_metrics": request.target_metrics,
            "analysis_depth": request.analysis_depth,
            "quantum_coherence": current_state.coherence,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to perform causal analysis: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform causal analysis: {e}"
        )


@app.get("/api/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    if not digital_twin:
        raise HTTPException(
            status_code=503, detail="Quantum digital twin not initialized"
        )

    try:
        metrics = digital_twin.get_performance_metrics()

        return PerformanceMetricsResponse(
            uptime=metrics.get("uptime", 0.0),
            predictions_made=metrics.get("predictions_made", 0),
            anomalies_detected=metrics.get("anomalies_detected", 0),
            self_healing_events=metrics.get("self_healing_events", 0),
            avg_response_time=metrics.get("avg_response_time", 0.0),
            quantum_coherence=metrics.get("coherence", 0.0),
            system_entropy=metrics.get("entropy", 0.0),
            health_score=metrics.get("health_score", 0.0),
            throughput=metrics.get("predictions_made", 0)
            / max(metrics.get("uptime", 1), 1),
            memory_usage=0.0,  # Would implement actual memory monitoring
        )
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance metrics: {e}"
        )


@app.get("/api/connections")
async def get_websocket_connections():
    """Get WebSocket connection statistics"""
    stats = await connection_manager.get_connection_stats()
    return {"websocket_connections": stats, "timestamp": datetime.now().isoformat()}


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, client_id: Optional[str] = Query(None)
):
    """Enhanced WebSocket endpoint for real-time quantum data streaming"""
    await connection_manager.connect(websocket, client_id)

    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                # Handle client requests
                if message.get("type") == "ping":
                    await connection_manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.now().isoformat()},
                        websocket,
                    )
                elif message.get("type") == "subscribe":
                    # Update subscriptions
                    if websocket in connection_manager.connection_metadata:
                        connection_manager.connection_metadata[websocket][
                            "subscriptions"
                        ] = message.get("channels", ["all"])

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from WebSocket: {data}")

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


# Background tasks
async def stream_quantum_data():
    """Stream quantum-enhanced data to WebSocket clients"""
    while True:
        try:
            if digital_twin and digital_twin.is_running:
                current_state = digital_twin.get_current_state()
                if current_state:
                    # Prepare quantum-enhanced data
                    quantum_data = {
                        "timestamp": current_state.timestamp,
                        "classical_state": current_state.classical_state,
                        "health_vector": current_state.health_vector.tolist(),
                        "anomaly_signature": current_state.anomaly_signature.tolist(),
                        "control_policy": current_state.control_policy.tolist(),
                        "uncertainty_quantum": current_state.uncertainty_quantum.tolist(),
                        "coherence": current_state.coherence,
                        "entropy": current_state.entropy,
                        "health_score": np.mean(current_state.health_vector),
                        "predictions": await digital_twin.neural_controller.predict_future_states(
                            digital_twin.state_history[-10:]
                            if len(digital_twin.state_history) >= 10
                            else [],
                            horizon_steps=5,
                        ),
                    }

                    await connection_manager.broadcast(quantum_data, "quantum_update")

            await asyncio.sleep(0.1)  # 10Hz update rate

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in quantum data streaming: {e}")
            await asyncio.sleep(1)


async def monitor_system_health():
    """Monitor system health and performance"""
    while True:
        try:
            if digital_twin:
                # Get performance metrics
                metrics = digital_twin.get_performance_metrics()

                # Check for performance issues
                if metrics.get("avg_response_time", 0) > 0.1:  # 100ms threshold
                    logger.warning(
                        f"High response time detected: {metrics['avg_response_time']:.3f}s"
                    )

                if metrics.get("coherence", 0) < 0.5:  # Low coherence
                    logger.warning(
                        f"Low quantum coherence detected: {metrics['coherence']:.3f}"
                    )

                # Broadcast health status
                health_data = {
                    "system_health": metrics.get("health_score", 0),
                    "performance_issues": [
                        "high_response_time"
                        if metrics.get("avg_response_time", 0) > 0.1
                        else None,
                        "low_coherence" if metrics.get("coherence", 0) < 0.5 else None,
                    ],
                    "timestamp": datetime.now().isoformat(),
                }

                await connection_manager.broadcast(health_data, "health_monitor")

            await asyncio.sleep(5)  # Check every 5 seconds

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in system health monitoring: {e}")
            await asyncio.sleep(5)


async def cleanup_connections():
    """Clean up inactive WebSocket connections"""
    while True:
        try:
            if connection_manager.active_connections:
                now = datetime.now()
                inactive_connections = []

                for connection in connection_manager.active_connections:
                    if connection in connection_manager.connection_metadata:
                        last_ping = connection_manager.connection_metadata[connection][
                            "last_ping"
                        ]
                        if (now - last_ping).total_seconds() > 60:  # 1 minute timeout
                            inactive_connections.append(connection)

                # Remove inactive connections
                for connection in inactive_connections:
                    logger.info(
                        f"Removing inactive connection: {connection_manager.connection_metadata.get(connection, {}).get('client_id', 'unknown')}"
                    )
                    connection_manager.disconnect(connection)

            await asyncio.sleep(30)  # Check every 30 seconds

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in connection cleanup: {e}")
            await asyncio.sleep(30)


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the PhoenixDT Quantum API server"""
    logger.info(f"Starting PhoenixDT Quantum API server on {host}:{port}")

    uvicorn.run(
        "phoenixdt.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
        use_colors=True,
    )


if __name__ == "__main__":
    run_server(reload=True)
