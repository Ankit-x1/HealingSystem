"""
3-Phase Motor Simulator with Bearing Degradation Physics - Python 3.12.2 compatible

Implements realistic physics-based simulation of industrial 3-phase induction motors
with progressive bearing degradation, thermal effects, and vibration dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.integrate import odeint
import pandas as pd
from loguru import logger


@dataclass
class MotorParameters:
    """Physical parameters of 3-phase induction motor"""

    # Electrical parameters
    rated_power: float = 5000.0  # W
    rated_voltage: float = 400.0  # V (line-to-line)
    rated_frequency: float = 50.0  # Hz
    poles: int = 4
    efficiency: float = 0.92

    # Motor parameters
    stator_resistance: float = 0.5  # Ohms
    rotor_resistance: float = 0.4  # Ohms
    stator_inductance: float = 0.004  # H
    rotor_inductance: float = 0.004  # H
    mutual_inductance: float = 0.1  # H

    # Mechanical parameters
    inertia: float = 0.05  # kg.m^2
    friction_coefficient: float = 0.001  # Nm.s/rad
    rated_speed: float = 1800.0  # RPM

    # Bearing parameters
    bearing_stiffness: float = 1e6  # N/m
    bearing_damping: float = 100.0  # N.s/m
    bearing_mass: float = 2.0  # kg


@dataclass
class BearingState:
    """Bearing degradation state"""

    wear_level: float = 0.0  # 0-1 (0=healthy, 1=failure)
    temperature: float = 25.0  # °C
    vibration_rms: float = 0.0  # mm/s
    lubrication_quality: float = 1.0  # 0-1

    # Defect parameters
    inner_race_defect: float = 0.0  # 0-1
    outer_race_defect: float = 0.0  # 0-1
    ball_defect: float = 0.0  # 0-1
    cage_defect: float = 0.0  # 0-1


class MotorSimulator:
    """High-fidelity 3-phase motor simulator with bearing degradation"""

    def __init__(self, params: Optional[MotorParameters] = None):
        self.params = params or MotorParameters()
        self.bearing_state = BearingState()

        # State variables
        self.time = 0.0
        self.speed = 0.0  # rad/s
        self.torque = 0.0  # Nm
        self.current = np.zeros(3)  # 3-phase currents
        self.voltage = np.zeros(3)  # 3-phase voltages

        # History for analysis
        self.history: List[Dict] = []

        logger.info("Motor simulator initialized")

    def _calculate_bearing_forces(
        self, speed: float, wear: float
    ) -> Tuple[float, float]:
        """Calculate bearing reaction forces based on wear and speed"""
        # Base bearing forces increase with wear
        force_multiplier = 1.0 + 5.0 * wear

        # Vibration frequency components
        bpfi = self.params.poles * speed / (2 * np.pi)  # Ball pass frequency inner
        bpfo = speed / (2 * np.pi)  # Ball pass frequency outer
        bsf = speed / (2 * np.pi) * 0.5  # Ball spin frequency

        # Generate vibration forces
        t = self.time
        fx = force_multiplier * (
            0.1 * np.sin(2 * np.pi * bpfi * t)
            + 0.05 * np.sin(2 * np.pi * bpfo * t)
            + 0.02 * np.sin(2 * np.pi * bsf * t)
        )

        fy = force_multiplier * (
            0.1 * np.cos(2 * np.pi * bpfi * t)
            + 0.05 * np.cos(2 * np.pi * bpfo * t)
            + 0.02 * np.cos(2 * np.pi * bsf * t)
        )

        return fx, fy

    def _calculate_motor_dynamics(
        self, state: np.ndarray, t: float, load_torque: float
    ) -> np.ndarray:
        """Calculate motor electrical and mechanical dynamics"""
        speed, current_d, current_q = state

        # Voltage equations (d-q reference frame)
        voltage_d = self.params.rated_voltage / np.sqrt(3)
        voltage_q = self.params.rated_voltage / np.sqrt(3)

        # Back-EMF
        back_emf_d = -speed * self.params.mutual_inductance * current_q
        back_emf_q = speed * self.params.mutual_inductance * current_d

        # Current derivatives
        didt = (
            voltage_d - self.params.stator_resistance * current_d + back_emf_d
        ) / self.params.stator_inductance
        dqdt = (
            voltage_q - self.params.stator_resistance * current_q + back_emf_q
        ) / self.params.stator_inductance

        # Electromagnetic torque
        em_torque = (
            1.5
            * self.params.poles
            * self.params.mutual_inductance
            * (current_d * current_q)
        )

        # Mechanical dynamics with bearing effects
        bearing_torque = (
            self.bearing_state.wear_level * 0.1 * speed
        )  # Friction increase due to wear
        friction_torque = self.params.friction_coefficient * speed

        # Speed derivative
        dsdt = (
            em_torque - load_torque - friction_torque - bearing_torque
        ) / self.params.inertia

        return np.array([dsdt, didt, dqdt])

    def _update_bearing_degradation(self, dt: float, load_factor: float):
        """Update bearing degradation based on operating conditions"""
        # Temperature rise due to friction and wear
        friction_heat = self.bearing_state.wear_level * load_factor * 50
        ambient_temp = 25.0

        # Thermal dynamics
        temp_rise = friction_heat * dt / (self.bearing_state.lubrication_quality * 100)
        self.bearing_state.temperature = ambient_temp + temp_rise

        # Wear progression (Arrhenius model)
        wear_rate = 1e-6 * np.exp(0.05 * (self.bearing_state.temperature - 25))
        wear_rate *= load_factor * (2.0 - self.bearing_state.lubrication_quality)

        self.bearing_state.wear_level = min(
            1.0, self.bearing_state.wear_level + wear_rate * dt
        )

        # Update defect probabilities
        if self.bearing_state.wear_level > 0.3:
            self.bearing_state.inner_race_defect = min(
                1.0, self.bearing_state.wear_level - 0.3
            )
        if self.bearing_state.wear_level > 0.5:
            self.bearing_state.outer_race_defect = min(
                1.0, self.bearing_state.wear_level - 0.5
            )
        if self.bearing_state.wear_level > 0.7:
            self.bearing_state.ball_defect = min(
                1.0, self.bearing_state.wear_level - 0.7
            )

        # Calculate vibration RMS
        fx, fy = self._calculate_bearing_forces(
            self.speed, self.bearing_state.wear_level
        )
        self.bearing_state.vibration_rms = np.sqrt(fx**2 + fy**2)

    def step(
        self,
        dt: float,
        load_torque: float,
        voltage_command: Optional[np.ndarray] = None,
    ) -> Dict:
        """Simulate one time step"""
        # Update voltage command
        if voltage_command is not None:
            self.voltage = voltage_command
        else:
            # Default 3-phase balanced voltage
            t = self.time
            self.voltage = (
                self.params.rated_voltage
                / np.sqrt(3)
                * np.array(
                    [
                        np.sin(2 * np.pi * self.params.rated_frequency * t),
                        np.sin(
                            2 * np.pi * self.params.rated_frequency * t - 2 * np.pi / 3
                        ),
                        np.sin(
                            2 * np.pi * self.params.rated_frequency * t + 2 * np.pi / 3
                        ),
                    ]
                )
            )

        # Solve motor dynamics
        initial_state = np.array([self.speed, self.current[0], self.current[1]])
        t_span = [0, dt]

        try:
            solution = odeint(
                self._calculate_motor_dynamics,
                initial_state,
                t_span,
                args=(load_torque,),
            )
            self.speed = solution[-1, 0]
            self.current[0] = solution[-1, 1]
            self.current[1] = solution[-1, 2]
            self.current[2] = -(self.current[0] + self.current[1])  # Kirchhoff's law
        except Exception as e:
            logger.warning(f"Integration failed: {e}")

        # Calculate torque
        self.torque = (
            1.5
            * self.params.poles
            * self.params.mutual_inductance
            * (self.current[0] * self.current[1])
        )

        # Update bearing degradation
        load_factor = abs(load_torque) / (
            self.params.rated_power / (self.params.rated_speed * 2 * np.pi / 60)
        )
        self._update_bearing_degradation(dt, load_factor)

        # Update time
        self.time += dt

        # Create state snapshot
        state = {
            "time": self.time,
            "speed_rpm": self.speed * 60 / (2 * np.pi),
            "torque_nm": self.torque,
            "current_a": np.linalg.norm(self.current),
            "voltage_v": np.linalg.norm(self.voltage),
            "power_w": np.dot(self.current, self.voltage),
            "bearing_wear": self.bearing_state.wear_level,
            "bearing_temp": self.bearing_state.temperature,
            "vibration_mm_s": self.bearing_state.vibration_rms,
            "efficiency": self._calculate_efficiency(load_torque),
        }

        self.history.append(state)
        return state

    def _calculate_efficiency(self, load_torque: float) -> float:
        """Calculate motor efficiency considering bearing wear"""
        # Base efficiency
        base_efficiency = self.params.efficiency

        # Efficiency degradation due to bearing wear
        wear_loss = self.bearing_state.wear_level * 0.15  # Up to 15% loss

        # Temperature effects
        temp_loss = max(0, (self.bearing_state.temperature - 60) * 0.001)

        efficiency = base_efficiency - wear_loss - temp_loss
        return max(0.1, min(1.0, efficiency))

    def inject_fault(self, fault_type: str, severity: float):
        """Inject specific fault into the system"""
        if fault_type == "bearing_wear":
            self.bearing_state.wear_level = min(1.0, severity)
        elif fault_type == "lubrication_loss":
            self.bearing_state.lubrication_quality = max(0.1, 1.0 - severity)
        elif fault_type == "overload":
            # Simulate temporary overload
            self.torque *= 1.0 + severity
        elif fault_type == "voltage_unbalance":
            # Create voltage unbalance
            self.voltage[0] *= 1.0 + severity * 0.1
            self.voltage[1] *= 1.0 - severity * 0.05
            self.voltage[2] *= 1.0 - severity * 0.05

        logger.info(f"Injected fault: {fault_type} with severity {severity}")

    def get_state_vector(self) -> np.ndarray:
        """Get current state as vector for ML algorithms"""
        return np.array(
            [
                self.speed,
                self.torque,
                np.linalg.norm(self.current),
                np.linalg.norm(self.voltage),
                self.bearing_state.wear_level,
                self.bearing_state.temperature,
                self.bearing_state.vibration_rms,
                self.bearing_state.lubrication_quality,
            ]
        )

    def reset(self):
        """Reset simulator to initial conditions"""
        self.time = 0.0
        self.speed = 0.0
        self.torque = 0.0
        self.current = np.zeros(3)
        self.voltage = np.zeros(3)
        self.bearing_state = BearingState()
        self.history = []
        logger.info("Motor simulator reset")

    def get_history_dataframe(self) -> pd.DataFrame:
        """Get simulation history as pandas DataFrame"""
        return pd.DataFrame(self.history)
