"""PhoenixDT digital twin core module."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from .config import PhoenixConfig

logger = logging.getLogger(__name__)


@dataclass
class MotorState:
    """Motor state representation."""

    speed: float = 0.0  # RPM
    torque: float = 0.0  # Nm
    current: float = 0.0  # A
    temperature: float = 25.0  # °C
    vibration: float = 0.0  # mm/s
    power: float = 0.0  # kW
    efficiency: float = 0.0  # %
    health: float = 1.0  # 0-1 scale


@dataclass
class ControlSignal:
    """Control signal output."""

    voltage: float = 0.0  # V
    frequency: float = 0.0  # Hz
    duty_cycle: float = 0.0  # %


class DigitalTwin:
    """Digital twin engine for motor simulation."""

    def __init__(self, config: PhoenixConfig | None = None):
        self.config = config or PhoenixConfig()

        # Core state
        self.state = MotorState()
        self.control = ControlSignal()
        self.target_speed = self.config.simulation.motor_speed
        self.load_torque = self.config.simulation.load_torque

        # Runtime
        self.is_running = False
        self.simulation_time = 0.0
        self.start_time = 0.0

        # Control state
        self._integral_error = 0.0

        # Callbacks
        self._state_callbacks = []
        self._anomaly_callbacks = []

        logger.info("Digital twin initialized")

    async def start(self, duration: float | None = None) -> None:
        """Start simulation."""
        logger.info(
            f"Starting simulation{' for ' + str(duration) + 's' if duration else ''}"
        )

        self.is_running = True
        self.start_time = time.time()

        try:
            await self._simulation_loop(duration)
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
        finally:
            self.is_running = False
            logger.info("Simulation stopped")

    async def stop(self) -> None:
        """Stop simulation."""
        self.is_running = False

    def add_state_callback(self, callback) -> None:
        """Add state update callback."""
        self._state_callbacks.append(callback)

    def add_anomaly_callback(self, callback) -> None:
        """Add anomaly detection callback."""
        self._anomaly_callbacks.append(callback)

    def set_target_speed(self, speed: float) -> None:
        """Set target motor speed."""
        self.target_speed = max(0, min(speed, self.config.simulation.motor_speed * 1.5))
        logger.info(f"Target speed: {self.target_speed} RPM")

    def set_load_torque(self, torque: float) -> None:
        """Set load torque."""
        self.load_torque = max(0, torque)
        logger.info(f"Load torque: {self.load_torque} Nm")

    def get_status(self) -> dict[str, Any]:
        """Get current system status."""
        return {
            "state": "running" if self.is_running else "stopped",
            "simulation_time": self.simulation_time,
            "motor": {
                "speed": self.state.speed,
                "torque": self.state.torque,
                "current": self.state.current,
                "temperature": self.state.temperature,
                "vibration": self.state.vibration,
                "power": self.state.power,
                "efficiency": self.state.efficiency,
            },
            "control": {
                "voltage": self.control.voltage,
                "frequency": self.control.frequency,
                "duty_cycle": self.control.duty_cycle,
            },
            "targets": {
                "speed": self.target_speed,
                "load_torque": self.load_torque,
            },
            "health": {
                "overall": self.state.health,
                "thermal": self._calculate_thermal_health(),
                "mechanical": self._calculate_mechanical_health(),
            },
        }

    async def _simulation_loop(self, duration: float | None) -> None:
        """Main simulation loop."""
        dt = self.config.simulation.dt

        # Motor parameters
        motor_inertia = 0.1
        motor_friction = 0.01
        motor_resistance = 10.0
        torque_constant = 0.5
        thermal_capacity = 500.0
        thermal_resistance = 0.1
        rated_power = self.config.simulation.motor_power * 1000
        rated_speed = self.config.simulation.motor_speed

        while self.is_running:
            loop_start = time.time()

            # PI Control
            speed_error = self.target_speed - self.state.speed
            self._integral_error += speed_error * dt
            self._integral_error = max(-100, min(100, self._integral_error))

            kp, ki = 0.1, 0.01
            control_output = kp * speed_error + ki * self._integral_error

            self.control.voltage = max(0, min(480, control_output))
            self.control.frequency = max(0, min(60, self.state.speed / 60))
            self.control.duty_cycle = (self.control.voltage / 480) * 100

            # Motor physics
            self.state.current = (
                self.control.voltage / motor_resistance
                if self.control.voltage > 0
                else 0
            )
            self.state.torque = self.state.current * torque_constant

            # Mechanical dynamics
            friction_torque = motor_friction * self.state.speed * 2 * 3.14159 / 60
            net_torque = self.state.torque - self.load_torque - friction_torque
            angular_acceleration = net_torque / motor_inertia

            self.state.speed += angular_acceleration * dt * 60 / (2 * 3.14159)
            self.state.speed = max(0, self.state.speed)

            # Power and efficiency
            mechanical_power = self.state.torque * self.state.speed * 2 * 3.14159 / 60
            electrical_power = self.control.voltage * self.state.current
            self.state.power = mechanical_power / 1000
            self.state.efficiency = (
                (mechanical_power / electrical_power * 100)
                if electrical_power > 0
                else 0
            )
            self.state.efficiency = min(100, max(0, self.state.efficiency))

            # Thermal dynamics
            power_losses = electrical_power - mechanical_power
            temp_rise = power_losses / thermal_capacity * dt
            ambient_temp = 25.0
            cooling = (self.state.temperature - ambient_temp) / thermal_resistance * dt
            self.state.temperature += temp_rise - cooling

            # Vibration
            base_vibration = 0.05 * (self.state.speed / rated_speed)
            random_vibration = (hash(str(time.time())) % 100) / 10000.0
            self.state.vibration = base_vibration + random_vibration

            # Health calculation
            self.state.health = self._calculate_overall_health()

            # Update simulation time
            self.simulation_time = time.time() - self.start_time

            # Check duration
            if duration and self.simulation_time >= duration:
                break

            # Notify callbacks
            await self._notify_state_update()
            await self._check_anomalies()

            # Loop timing
            elapsed = time.time() - loop_start
            if elapsed < dt:
                await asyncio.sleep(dt - elapsed)

    def _calculate_thermal_health(self) -> float:
        """Calculate thermal health score."""
        max_temp = self.config.control.safety_limits.get("max_temperature", 120.0)
        optimal_temp = 60.0
        deviation = abs(self.state.temperature - optimal_temp)
        return max(0, 1.0 - deviation / (max_temp - optimal_temp))

    def _calculate_mechanical_health(self) -> float:
        """Calculate mechanical health score."""
        max_vibration = self.config.control.safety_limits.get("max_vibration", 10.0)
        good_vibration = 2.0

        if self.state.vibration <= good_vibration:
            return 1.0

        return max(
            0,
            1.0
            - (self.state.vibration - good_vibration)
            / (max_vibration - good_vibration),
        )

    def _calculate_overall_health(self) -> float:
        """Calculate overall health score."""
        thermal = self._calculate_thermal_health()
        mechanical = self._calculate_mechanical_health()
        efficiency_health = min(1.0, self.state.efficiency / 85.0)

        return thermal * 0.4 + mechanical * 0.4 + efficiency_health * 0.2

    async def _notify_state_update(self) -> None:
        """Notify state callbacks."""
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.state)
                else:
                    callback(self.state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    async def _check_anomalies(self) -> None:
        """Check for anomalies and notify."""
        anomalies = []

        # Temperature anomaly
        max_temp = self.config.control.safety_limits.get("max_temperature", 120.0)
        if self.state.temperature > max_temp:
            anomalies.append(
                {
                    "type": "high_temperature",
                    "severity": min(1.0, (self.state.temperature - max_temp) / 50.0),
                    "value": self.state.temperature,
                    "threshold": max_temp,
                }
            )

        # Vibration anomaly
        max_vibration = self.config.control.safety_limits.get("max_vibration", 10.0)
        if self.state.vibration > max_vibration:
            anomalies.append(
                {
                    "type": "high_vibration",
                    "severity": min(1.0, self.state.vibration / max_vibration),
                    "value": self.state.vibration,
                    "threshold": max_vibration,
                }
            )

        # Current anomaly
        max_current = self.config.control.safety_limits.get("max_current", 50.0)
        if self.state.current > max_current:
            anomalies.append(
                {
                    "type": "overcurrent",
                    "severity": min(1.0, self.state.current / max_current),
                    "value": self.state.current,
                    "threshold": max_current,
                }
            )

        # Notify if anomalies found
        if anomalies and self._anomaly_callbacks:
            for callback in self._anomaly_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(anomalies)
                    else:
                        callback(anomalies)
                except Exception as e:
                    logger.error(f"Anomaly callback error: {e}")
