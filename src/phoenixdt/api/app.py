"""PhoenixDT API module."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from phoenixdt.core.config import PhoenixConfig
from phoenixdt.core.digital_twin import DigitalTwin

logger = logging.getLogger(__name__)


class MotorState(BaseModel):
    """Motor state model."""

    speed: float = Field(..., description="Motor speed in RPM")
    torque: float = Field(..., description="Motor torque in Nm")
    current: float = Field(..., description="Motor current in A")
    temperature: float = Field(..., description="Motor temperature in °C")
    vibration: float = Field(..., description="Motor vibration in mm/s")
    power: float = Field(..., description="Motor power in kW")
    efficiency: float = Field(..., description="Motor efficiency in %")


class ControlRequest(BaseModel):
    """Control request model."""

    target_speed: float | None = Field(None, description="Target speed in RPM")
    load_torque: float | None = Field(None, description="Load torque in Nm")


class SystemStatus(BaseModel):
    """System status model."""

    state: str = Field(..., description="System state")
    simulation_time: float = Field(..., description="Simulation time in seconds")
    motor: MotorState = Field(..., description="Motor state")
    control: Dict[str, float] = Field(..., description="Control signals")
    targets: Dict[str, float] = Field(..., description="Control targets")
    health: Dict[str, float] = Field(..., description="Health metrics")


class Anomaly(BaseModel):
    """Anomaly model."""

    type: str = Field(..., description="Anomaly type")
    severity: float = Field(..., description="Severity (0-1)")
    value: float = Field(..., description="Current value")
    threshold: float = Field(..., description="Threshold value")


# Global instances
digital_twin: DigitalTwin | None = None


class WebSocketManager:
    """WebSocket connection manager."""

    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connections."""
        disconnected = []
        for connection in self.connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)


ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    global digital_twin

    # Initialize digital twin
    config = PhoenixConfig()
    digital_twin = DigitalTwin(config)

    # Add callbacks
    digital_twin.add_state_callback(on_state_update)
    digital_twin.add_anomaly_callback(on_anomaly_detected)

    # Start simulation
    asyncio.create_task(digital_twin.start())

    yield

    # Cleanup
    if digital_twin:
        await digital_twin.stop()


def on_state_update(state):
    """State update callback."""
    if ws_manager.connections:
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(
                ws_manager.broadcast(
                    {
                        "type": "state_update",
                        "timestamp": time.time(),
                        "data": {
                            "speed": state.speed,
                            "torque": state.torque,
                            "current": state.current,
                            "temperature": state.temperature,
                            "vibration": state.vibration,
                            "power": state.power,
                            "efficiency": state.efficiency,
                            "health": state.health,
                        },
                    }
                )
            )
        except RuntimeError:
            pass


def on_anomaly_detected(anomalies):
    """Anomaly detection callback."""
    if ws_manager.connections and anomalies:
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(
                ws_manager.broadcast(
                    {
                        "type": "anomaly_detected",
                        "timestamp": time.time(),
                        "data": anomalies,
                    }
                )
            )
        except RuntimeError:
            pass


# Create FastAPI app
app = FastAPI(
    title="PhoenixDT API",
    description="Industrial Digital Twin API",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "PhoenixDT API", "version": "2.0.0"}


@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get system status."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    status = digital_twin.get_status()
    return SystemStatus(**status)


@app.post("/api/control")
async def set_control(control: ControlRequest):
    """Set control parameters."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    if control.target_speed is not None:
        digital_twin.set_target_speed(control.target_speed)

    if control.load_torque is not None:
        digital_twin.set_load_torque(control.load_torque)

    return {"message": "Control updated"}


@app.post("/api/start")
async def start_simulation():
    """Start simulation."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    if not digital_twin.is_running:
        asyncio.create_task(digital_twin.start())

    return {"message": "Simulation started"}


@app.post("/api/stop")
async def stop_simulation():
    """Stop simulation."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    await digital_twin.stop()
    return {"message": "Simulation stopped"}


@app.get("/api/anomalies", response_model=List[Anomaly])
async def get_anomalies():
    """Get current anomalies."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    status = digital_twin.get_status()

    # Extract anomalies from health check
    anomalies = []

    if status["motor"]["temperature"] > 100:
        anomalies.append(
            Anomaly(
                type="high_temperature",
                severity=min(1.0, status["motor"]["temperature"] / 120.0),
                value=status["motor"]["temperature"],
                threshold=100.0,
            )
        )

    if status["motor"]["vibration"] > 5.0:
        anomalies.append(
            Anomaly(
                type="high_vibration",
                severity=min(1.0, status["motor"]["vibration"] / 10.0),
                value=status["motor"]["vibration"],
                threshold=5.0,
            )
        )

    return anomalies


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    if not digital_twin:
        return {"status": "unhealthy", "reason": "Digital twin not initialized"}

    return {
        "status": "healthy" if digital_twin.is_running else "stopped",
        "uptime": digital_twin.simulation_time,
        "health": digital_twin.state.health,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or handle commands
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
