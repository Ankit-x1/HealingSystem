"""
Digital Twin Core Engine

Orchestrates simulation, ML models, and control systems
to create a comprehensive digital twin platform.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import time
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from ..core.config import Config
from ..simulation.motor_simulator import MotorSimulator, MotorParameters
from ..ml.failure_synthesizer import FailureSynthesizer
from ..ml.anomaly_detector import AnomalyDetector
from ..control.rl_controller import RLController
from ..control.pid_controller import PIDController


@dataclass
class TwinState:
    """Digital twin state snapshot"""

    timestamp: float
    physical_state: Dict[str, float]
    predicted_state: Dict[str, float]
    control_actions: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    health_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]


class DigitalTwin:
    """Core digital twin engine integrating simulation, ML, and control"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        # Initialize components
        self.simulator = MotorSimulator()
        self.failure_synthesizer = FailureSynthesizer(self.config.ml)
        self.anomaly_detector = AnomalyDetector(self.config.ml)
        self.rl_controller = RLController(self.config.control)
        self.pid_controller = PIDController(self.config.control.pid_gains)

        # State management
        self.current_state: Optional[TwinState] = None
        self.state_history: List[TwinState] = []
        self.is_running = False
        self.simulation_time = 0.0

        # Control mode
        self.control_mode = "rl"  # "rl", "pid", "manual"
        self.manual_control = np.zeros(3)

        # Callbacks for external interfaces
        self.state_callbacks: List[Callable] = []
        self.anomaly_callbacks: List[Callable] = []

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info("Digital twin initialized")

    async def start(self, duration: Optional[float] = None):
        """Start digital twin simulation"""
        self.is_running = True
        start_time = time.time()

        logger.info(f"Starting digital twin simulation")

        try:
            while self.is_running:
                if duration and (time.time() - start_time) > duration:
                    break

                # Execute simulation step
                await self._step()

                # Control loop frequency
                await asyncio.sleep(1.0 / self.config.control.control_frequency)

        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self.is_running = False
            logger.info("Digital twin simulation stopped")

    async def _step(self):
        """Execute one simulation step"""
        # Get current load (could be from external source)
        load_torque = self._get_load_torque()

        # Get control action based on mode
        if self.control_mode == "rl":
            voltage_command = await self.rl_controller.get_action(
                self.simulator.get_state_vector()
            )
        elif self.control_mode == "pid":
            voltage_command = self.pid_controller.compute(
                setpoint=self.config.simulation.motor_speed * 2 * np.pi / 60,
                current_value=self.simulator.speed,
            )
            voltage_command = np.full(3, voltage_command)
        else:  # manual
            voltage_command = self.manual_control

        # Simulate physical system
        physical_state = self.simulator.step(
            dt=self.config.simulation.dt,
            load_torque=load_torque,
            voltage_command=voltage_command,
        )

        # Generate failure scenarios for training
        if np.random.random() < 0.01:  # 1% chance
            synthetic_failure = self.failure_synthesizer.generate_failure(
                self.simulator.get_state_vector()
            )
            self.simulator.inject_fault(
                synthetic_failure["type"], synthetic_failure["severity"]
            )

        # Detect anomalies
        state_vector = self.simulator.get_state_vector()
        anomalies = await self.anomaly_detector.detect(state_vector)

        # Predict future state
        predicted_state = await self._predict_future_state(
            state_vector, voltage_command
        )

        # Calculate health and performance metrics
        health_metrics = self._calculate_health_metrics()
        performance_metrics = self._calculate_performance_metrics(physical_state)

        # Create twin state
        self.current_state = TwinState(
            timestamp=self.simulator.time,
            physical_state=physical_state,
            predicted_state=predicted_state,
            control_actions={"voltage": voltage_command.tolist()},
            anomalies=anomalies,
            health_metrics=health_metrics,
            performance_metrics=performance_metrics,
        )

        # Store in history
        self.state_history.append(self.current_state)

        # Update simulation time
        self.simulation_time += self.config.simulation.dt

        # Trigger callbacks
        await self._trigger_callbacks()

        # Update RL controller with new state
        if self.control_mode == "rl":
            reward = self._calculate_reward(physical_state, health_metrics)
            await self.rl_controller.update(
                state_vector, voltage_command, reward, state_vector
            )

    def _get_load_torque(self) -> float:
        """Get current load torque (could be from external source)"""
        # Simulate varying load
        base_load = self.config.simulation.load_torque
        variation = np.sin(self.simulation_time * 0.1) * 2.0
        noise = np.random.normal(0, 0.5)
        return base_load + variation + noise

    async def _predict_future_state(
        self, current_state: np.ndarray, control_action: np.ndarray
    ) -> Dict[str, float]:
        """Predict future system state using ML models"""
        # Simple physics-based prediction for now
        # In production, this would use trained neural networks

        future_speed = current_state[0] * 1.01  # Slight speed increase
        future_torque = current_state[1] * 0.98  # Slight torque decrease
        future_wear = current_state[4] * 1.001  # Gradual wear increase

        return {
            "speed_rpm": future_speed * 60 / (2 * np.pi),
            "torque_nm": future_torque,
            "bearing_wear": future_wear,
            "predicted_horizon": 1.0,  # seconds
        }

    def _calculate_health_metrics(self) -> Dict[str, float]:
        """Calculate system health metrics"""
        bearing = self.simulator.bearing_state

        # Overall health score (0-1, 1=healthy)
        bearing_health = 1.0 - bearing.wear_level
        thermal_health = max(0, 1.0 - (bearing.temperature - 25) / 100)
        vibration_health = max(0, 1.0 - bearing.vibration_rms / 10)

        overall_health = np.mean([bearing_health, thermal_health, vibration_health])

        # Remaining useful life (hours)
        wear_rate = bearing.wear_level / max(1, self.simulation_time / 3600)
        rul = max(0, (1.0 - bearing.wear_level) / max(wear_rate, 1e-6))

        return {
            "overall_health": overall_health,
            "bearing_health": bearing_health,
            "thermal_health": thermal_health,
            "vibration_health": vibration_health,
            "remaining_useful_life_hours": rul,
            "wear_rate_per_hour": wear_rate,
        }

    def _calculate_performance_metrics(
        self, physical_state: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate system performance metrics"""
        efficiency = physical_state.get("efficiency", 0.9)
        power = physical_state.get("power_w", 0)
        speed_error = abs(
            physical_state.get("speed_rpm", 0) - self.config.simulation.motor_speed
        )

        # Energy efficiency score
        efficiency_score = efficiency / self.config.simulation.motor_power

        # Speed regulation score
        speed_score = max(0, 1.0 - speed_error / self.config.simulation.motor_speed)

        # Overall performance score
        performance_score = (efficiency_score + speed_score) / 2

        return {
            "efficiency": efficiency,
            "power_consumption": power,
            "speed_regulation": speed_score,
            "performance_score": performance_score,
            "energy_cost_per_hour": power * 0.1 / 1000,  # Assuming $0.1/kWh
        }

    def _calculate_reward(
        self, physical_state: Dict[str, float], health_metrics: Dict[str, float]
    ) -> float:
        """Calculate reward for RL controller"""
        # Multi-objective reward function
        efficiency_reward = physical_state.get("efficiency", 0.9)
        health_reward = health_metrics.get("overall_health", 1.0)
        speed_reward = (
            1.0
            - abs(
                physical_state.get("speed_rpm", 0) - self.config.simulation.motor_speed
            )
            / self.config.simulation.motor_speed
        )

        # Weighted combination
        reward = 0.4 * efficiency_reward + 0.4 * health_reward + 0.2 * speed_reward

        return reward

    async def _trigger_callbacks(self):
        """Trigger registered callbacks"""
        if not self.current_state:
            return

        # State callbacks
        for callback in self.state_callbacks:
            try:
                await callback(self.current_state)
            except Exception as e:
                logger.warning(f"State callback error: {e}")

        # Anomaly callbacks
        if self.current_state.anomalies:
            for callback in self.anomaly_callbacks:
                try:
                    await callback(self.current_state.anomalies)
                except Exception as e:
                    logger.warning(f"Anomaly callback error: {e}")

    def set_control_mode(self, mode: str):
        """Set control mode ('rl', 'pid', 'manual')"""
        if mode not in ["rl", "pid", "manual"]:
            raise ValueError("Control mode must be 'rl', 'pid', or 'manual'")

        self.control_mode = mode
        logger.info(f"Control mode changed to: {mode}")

    def set_manual_control(self, voltage: np.ndarray):
        """Set manual control voltage"""
        self.manual_control = voltage
        if self.control_mode != "manual":
            logger.warning("Setting manual control but mode is not 'manual'")

    def add_state_callback(self, callback: Callable):
        """Add callback for state updates"""
        self.state_callbacks.append(callback)

    def add_anomaly_callback(self, callback: Callable):
        """Add callback for anomaly detection"""
        self.anomaly_callbacks.append(callback)

    def stop(self):
        """Stop digital twin simulation"""
        self.is_running = False
        logger.info("Digital twin stop requested")

    def reset(self):
        """Reset digital twin to initial state"""
        self.simulator.reset()
        self.current_state = None
        self.state_history = []
        self.simulation_time = 0.0
        logger.info("Digital twin reset")

    def get_current_state(self) -> Optional[TwinState]:
        """Get current twin state"""
        return self.current_state

    def get_state_history(self, limit: Optional[int] = None) -> List[TwinState]:
        """Get state history"""
        if limit:
            return self.state_history[-limit:]
        return self.state_history

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.state_history:
            return {}

        # Extract metrics
        efficiencies = [
            s.performance_metrics.get("efficiency", 0) for s in self.state_history
        ]
        health_scores = [
            s.health_metrics.get("overall_health", 0) for s in self.state_history
        ]
        anomaly_count = sum(len(s.anomalies) for s in self.state_history)

        return {
            "avg_efficiency": np.mean(efficiencies),
            "min_efficiency": np.min(efficiencies),
            "avg_health": np.mean(health_scores),
            "min_health": np.min(health_scores),
            "total_anomalies": anomaly_count,
            "uptime": self.simulation_time,
            "total_steps": len(self.state_history),
        }

    def export_data(self, filepath: Path):
        """Export twin data to file"""
        if not self.state_history:
            logger.warning("No data to export")
            return

        # Convert to DataFrame
        data = []
        for state in self.state_history:
            row = {
                "timestamp": state.timestamp,
                **state.physical_state,
                **state.predicted_state,
                **state.health_metrics,
                **state.performance_metrics,
                "anomaly_count": len(state.anomalies),
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Data exported to {filepath}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.stop()
        self.executor.shutdown(wait=True)
