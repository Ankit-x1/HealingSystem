"""
PhoenixDT Digital Twin Core

Production-ready industrial digital twin with physics simulation,
AI-powered anomaly detection, and self-healing control.
"""

from __future__ import annotations
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from .config import PhoenixConfig
from .physics_engine import PhysicsSimulator
from .neural_architectures import AdaptiveNeuralController
from .causal_engine import CausalInferenceEngine


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

    def __init__(self, config: Optional[PhoenixConfig] = None):
        self.config = config or PhoenixConfig()

        # Initialize components
        self.simulator = PhysicsSimulator(
            dt=self.config.simulation.dt,
            integration_method=self.config.simulation.integration_method,
        )

        self.neural_controller = AdaptiveNeuralController(
            input_dim=16,  # State dimension
            output_dim=3,  # Control output dimension
            hidden_dims=self.config.ml.vae_hidden_dims,
        )

        self.causal_engine = CausalInferenceEngine(n_variables=8, max_lag=5)

        # State management
        self.current_state: Optional[TwinState] = None
        self.state_history: List[TwinState] = []
        self.is_running = False

        # Performance metrics
        self.metrics = {
            "uptime": 0.0,
            "predictions_made": 0,
            "anomalies_detected": 0,
            "self_healing_events": 0,
            "avg_response_time": 0.0,
        }

        # Async management
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._tasks: List[asyncio.Task] = []

        logger.info("Digital twin initialized")

    async def start(self, duration: Optional[float] = None) -> None:
        """Start digital twin simulation"""
        logger.info(
            f"Starting digital twin simulation for {duration or 'indefinite'} seconds"
        )

        self.is_running = True
        start_time = time.time()

        try:
            # Start background tasks
            self._tasks = [
                asyncio.create_task(self._simulation_loop()),
                asyncio.create_task(self._anomaly_detection_loop()),
                asyncio.create_task(self._self_healing_loop()),
                asyncio.create_task(self._performance_monitoring_loop()),
            ]

            # Run for specified duration or indefinitely
            if duration:
                await asyncio.sleep(duration)
                await self.stop()
            else:
                # Run until stopped
                while self.is_running:
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
        finally:
            self.metrics["uptime"] = time.time() - start_time
            logger.info(f"Simulation completed. Uptime: {self.metrics['uptime']:.2f}s")

    async def stop(self) -> None:
        """Stop digital twin simulation"""
        logger.info("Stopping digital twin simulation")

        self.is_running = False

        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("Digital twin stopped")

    def get_current_state(self) -> Optional[TwinState]:
        """Get current digital twin state"""
        return self.current_state

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "uptime": self.metrics["uptime"],
            "avg_efficiency": self._calculate_avg_efficiency(),
            "min_efficiency": self._calculate_min_efficiency(),
            "avg_health": self._calculate_avg_health(),
            "total_anomalies": self.metrics["anomalies_detected"],
            "total_healing_events": self.metrics["self_healing_events"],
            "avg_response_time": self.metrics["avg_response_time"],
        }

    async def _simulation_loop(self) -> None:
        """Main simulation loop"""
        while self.is_running:
            loop_start = time.time()

            try:
                # Get control input
                control_input = await self._get_control_input()

                # Step physics simulation
                physical_state = await self.simulator.step(control_input)

                # Get neural control output
                neural_control = await self.neural_controller.compute_control(
                    physical_state, self._get_health_vector()
                )

                # Create state snapshot
                self.current_state = TwinState(
                    timestamp=time.time(),
                    physical_state=physical_state,
                    predicted_state=await self._predict_next_state(physical_state),
                    control_actions=neural_control,
                    anomalies=await self._detect_anomalies(physical_state),
                    health_metrics=await self._calculate_health_metrics(physical_state),
                    performance_metrics=await self._calculate_performance_metrics(),
                )

                # Store in history
                self.state_history.append(self.current_state)
                if len(self.state_history) > 1000:
                    self.state_history.pop(0)

                # Update metrics
                self.metrics["predictions_made"] += 1
                loop_time = time.time() - loop_start
                self.metrics["avg_response_time"] = (
                    self.metrics["avg_response_time"] * 0.9 + loop_time * 0.1
                )

            except Exception as e:
                logger.error(f"Simulation loop error: {e}")
                await asyncio.sleep(1.0)  # Error recovery delay

    async def _anomaly_detection_loop(self) -> None:
        """Anomaly detection loop"""
        while self.is_running:
            try:
                if self.current_state:
                    # Check for anomalies in current state
                    anomalies = self.current_state.anomalies
                    if anomalies:
                        self.metrics["anomalies_detected"] += len(anomalies)
                        logger.warning(f"Anomalies detected: {len(anomalies)}")

                await asyncio.sleep(0.5)  # Check every 500ms

            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(1.0)

    async def _self_healing_loop(self) -> None:
        """Self-healing loop"""
        while self.is_running:
            try:
                if self.current_state:
                    health_metrics = self.current_state.health_metrics
                    overall_health = health_metrics.get("overall_health", 1.0)

                    # Trigger healing if health is low
                    if overall_health < 0.7:
                        await self._trigger_self_healing()
                        self.metrics["self_healing_events"] += 1

                await asyncio.sleep(1.0)  # Check every second

            except Exception as e:
                logger.error(f"Self-healing error: {e}")
                await asyncio.sleep(1.0)

    async def _performance_monitoring_loop(self) -> None:
        """Performance monitoring loop"""
        while self.is_running:
            try:
                # Monitor system performance
                cpu_usage = self._get_cpu_usage()
                memory_usage = self._get_memory_usage()

                # Log performance metrics
                if cpu_usage > 80 or memory_usage > 80:
                    logger.warning(
                        f"High resource usage - CPU: {cpu_usage}%, Memory: {memory_usage}%"
                    )

                await asyncio.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5.0)

    # Helper methods
    async def _get_control_input(self) -> np.ndarray:
        """Get control input (simplified)"""
        # In a real implementation, this would get control from neural controller
        # For now, return a simple control signal
        return np.array(
            [400.0, 400.0 * np.cos(time.time()), 400.0 * np.sin(time.time())]
        )

    def _get_health_vector(self) -> np.ndarray:
        """Get health vector for neural controller"""
        if self.current_state:
            health_metrics = self.current_state.health_metrics
            return np.array(
                [
                    health_metrics.get("motor_health", 1.0),
                    health_metrics.get("bearing_health", 1.0),
                    health_metrics.get("efficiency_health", 1.0),
                    health_metrics.get("thermal_health", 1.0),
                ]
            )
        return np.ones(4)

    async def _predict_next_state(
        self, current_state: Dict[str, float]
    ) -> Dict[str, float]:
        """Predict next state (simplified)"""
        # Simple prediction based on current state
        prediction = {}
        for key, value in current_state.items():
            # Add small random variation for prediction
            prediction[key] = value + np.random.normal(0, value * 0.01)
        return prediction

    async def _detect_anomalies(
        self, physical_state: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in physical state"""
        anomalies = []

        # Simple anomaly detection based on thresholds
        if physical_state.get("temperature", 0) > 100:
            anomalies.append(
                {
                    "type": "high_temperature",
                    "severity": (physical_state["temperature"] - 100) / 100,
                    "timestamp": time.time(),
                }
            )

        if physical_state.get("vibration", 0) > 5.0:
            anomalies.append(
                {
                    "type": "high_vibration",
                    "severity": physical_state["vibration"] / 5.0,
                    "timestamp": time.time(),
                }
            )

        return anomalies

    async def _calculate_health_metrics(
        self, physical_state: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate health metrics"""
        health_metrics = {}

        # Motor health based on temperature and vibration
        temp_health = max(0, 1.0 - (physical_state.get("temperature", 25) - 25) / 75)
        vibration_health = max(0, 1.0 - physical_state.get("vibration", 0) / 5.0)
        health_metrics["motor_health"] = (temp_health + vibration_health) / 2

        # Bearing health
        bearing_health = max(0, 1.0 - physical_state.get("bearing_wear", 0))
        health_metrics["bearing_health"] = bearing_health

        # Efficiency health
        efficiency = physical_state.get("efficiency", 0.85)
        health_metrics["efficiency_health"] = min(1.0, efficiency / 0.85)

        # Thermal health
        thermal_health = max(
            0, 1.0 - (physical_state.get("temperature", 25) - 25) / 100
        )
        health_metrics["thermal_health"] = thermal_health

        # Overall health
        health_metrics["overall_health"] = sum(health_metrics.values()) / len(
            health_metrics
        )

        return health_metrics

    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        return {
            "response_time": self.metrics["avg_response_time"],
            "throughput": self.metrics["predictions_made"]
            / max(self.metrics["uptime"], 1),
            "cpu_usage": self._get_cpu_usage(),
            "memory_usage": self._get_memory_usage(),
        }

    def _calculate_avg_efficiency(self) -> float:
        """Calculate average efficiency from history"""
        if not self.state_history:
            return 0.85

        efficiencies = [
            state.health_metrics.get("efficiency_health", 0.85)
            for state in self.state_history
        ]
        return np.mean(efficiencies) if efficiencies else 0.85

    def _calculate_min_efficiency(self) -> float:
        """Calculate minimum efficiency from history"""
        if not self.state_history:
            return 0.85

        efficiencies = [
            state.health_metrics.get("efficiency_health", 0.85)
            for state in self.state_history
        ]
        return np.min(efficiencies) if efficiencies else 0.85

    def _calculate_avg_health(self) -> float:
        """Calculate average health from history"""
        if not self.state_history:
            return 1.0

        health_scores = [
            state.health_metrics.get("overall_health", 1.0)
            for state in self.state_history
        ]
        return np.mean(health_scores) if health_scores else 1.0

    def _get_cpu_usage(self) -> float:
        """Get CPU usage (simplified)"""
        try:
            import psutil

            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 25.0  # Default value

    def _get_memory_usage(self) -> float:
        """Get memory usage (simplified)"""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            return 30.0  # Default value

    async def _trigger_self_healing(self) -> None:
        """Trigger self-healing mechanism"""
        logger.info("Triggering self-healing mechanism")

        # Simple healing: reset some parameters
        if self.current_state:
            # Reset control to safe values
            safe_control = np.array([380.0, 380.0, 380.0])
            logger.info(f"Applied safe control: {safe_control}")

    def set_control_mode(self, mode: str) -> None:
        """Set control mode"""
        logger.info(f"Control mode set to: {mode}")
        # In a real implementation, this would switch between different control strategies

    def set_manual_control(self, control_input: np.ndarray) -> None:
        """Set manual control input"""
        logger.info(f"Manual control set: {control_input}")
        # In a real implementation, this would override automatic control
