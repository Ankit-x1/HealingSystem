"""
FastAPI Application for PhoenixDT

Clean, production-ready REST API with WebSocket streaming.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from phoenixdt.core.config import PhoenixConfig
from phoenixdt.core.digital_twin import DigitalTwin


# Pydantic models for API
class MotorStateResponse(BaseModel):
    """Motor state response model."""

    speed: float = Field(..., description="Motor speed in RPM")
    torque: float = Field(..., description="Motor torque in Nm")
    current: float = Field(..., description="Motor current in A")
    temperature: float = Field(..., description="Motor temperature in °C")
    vibration: float = Field(..., description="Motor vibration in mm/s")
    power: float = Field(..., description="Motor power in kW")
    efficiency: float = Field(..., description="Motor efficiency in %")


class ControlRequest(BaseModel):
    """Control request model."""

    target_speed: Optional[float] = Field(None, description="Target speed in RPM")
    load_torque: Optional[float] = Field(None, description="Load torque in Nm")


class SystemStatusResponse(BaseModel):
    """System status response model."""

    state: str = Field(..., description="System state")
    simulation_time: float = Field(..., description="Simulation time in seconds")
    motor: MotorStateResponse = Field(..., description="Motor state")
    targets: Dict[str, float] = Field(..., description="Control targets")


class AnomalyResponse(BaseModel):
    """Anomaly response model."""

    type: str = Field(..., description="Anomaly type")
    severity: str = Field(..., description="Anomaly severity")
    value: float = Field(..., description="Anomaly value")
    possible_causes: List[str] = Field(..., description="Possible causes")


# Global digital twin instance
digital_twin: Optional[DigitalTwin] = None
websocket_connections: List[WebSocket] = []


class ConnectionManager:
    """WebSocket connection manager."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception:
            # Connection might be closed
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected WebSockets."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global digital_twin

    # Initialize digital twin
    config = PhoenixConfig()
    digital_twin = DigitalTwin(config)

    # Add callbacks for WebSocket updates
    digital_twin.add_state_callback(on_state_update)
    digital_twin.add_anomaly_callback(on_anomaly_detected)

    # Start digital twin
    await digital_twin.start()

    yield

    # Cleanup
    if digital_twin:
        await digital_twin.stop()


# Create FastAPI app
app = FastAPI(
    title="PhoenixDT API",
    description="Industrial Digital Twin API for Motor Systems",
    version="1.0.0",
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


# Callback functions
def on_state_update(state):
    """Callback for state updates."""
    if manager.active_connections:
        asyncio.create_task(
            manager.broadcast(
                {
                    "type": "state_update",
                    "data": {
                        "speed": state.speed,
                        "torque": state.torque,
                        "current": state.current,
                        "temperature": state.temperature,
                        "vibration": state.vibration,
                        "power": state.power,
                        "efficiency": state.efficiency,
                    },
                }
            )
        )


def on_anomaly_detected(anomalies):
    """Callback for anomaly detection."""
    if manager.active_connections and anomalies:
        asyncio.create_task(
            manager.broadcast({"type": "anomaly_detected", "data": anomalies})
        )


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "PhoenixDT API", "version": "1.0.0"}


@app.get("/api/status", response_model=SystemStatusResponse)
async def get_status():
    """Get current system status."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    status = digital_twin.get_status()

    return SystemStatusResponse(
        state=status["state"],
        simulation_time=status["simulation_time"],
        motor=MotorStateResponse(**status["motor"]),
        targets=status["targets"],
    )


@app.post("/api/control")
async def set_control(control: ControlRequest):
    """Set control parameters."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    if control.target_speed is not None:
        digital_twin.set_target_speed(control.target_speed)

    if control.load_torque is not None:
        digital_twin.set_load_torque(control.load_torque)

    return {"message": "Control parameters updated successfully"}


@app.get("/api/anomalies", response_model=List[AnomalyResponse])
async def get_anomalies():
    """Get current anomalies."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    # Get recent anomalies from analyzer
    anomalies = digital_twin.analyzer.analyze_anomalies()

    return [
        AnomalyResponse(
            type=anomaly["type"],
            severity=anomaly["severity"],
            value=anomaly["value"],
            possible_causes=anomaly["possible_causes"],
        )
        for anomaly in anomalies
    ]


@app.post("/api/start")
async def start_simulation():
    """Start the simulation."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    await digital_twin.start()
    return {"message": "Simulation started"}


@app.post("/api/stop")
async def stop_simulation():
    """Stop the simulation."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    await digital_twin.stop()
    return {"message": "Simulation stopped"}


@app.post("/api/pause")
async def pause_simulation():
    """Pause the simulation."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    await digital_twin.pause()
    return {"message": "Simulation paused"}


@app.post("/api/resume")
async def resume_simulation():
    """Resume the simulation."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    await digital_twin.resume()
    return {"message": "Simulation resumed"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    if not digital_twin:
        raise HTTPException(status_code=503, detail="Digital twin not initialized")

    return {
        "status": "healthy",
        "digital_twin_state": digital_twin.system_state.value,
        "uptime": digital_twin.simulation_time,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back or handle specific commands
            await manager.send_personal_message(
                {"type": "echo", "data": data}, websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logging.error(f"Global exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    config = PhoenixConfig()
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=config.interface.api_port,
        reload=config.interface.api_port == 8000,  # Only reload in development
    )
