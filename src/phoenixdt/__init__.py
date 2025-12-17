"""
PhoenixDT: Failure-Aware Digital Twin with Anomaly Synthesis & Self-Healing Control

A cutting-edge industrial digital twin platform that combines:
- Physics-based simulation of industrial systems
- Generative AI for failure synthesis when real data is scarce
- Reinforcement learning for adaptive control
- Causal inference for explainable decisions
- Real-time anomaly detection with uncertainty quantification
"""

__version__ = "0.1.0"
__author__ = "Ankit Karki"
__email__ = "karkiankit101@gmail.com"

from .core.digital_twin import DigitalTwin
from .core.config import Config
from .simulation.motor_simulator import MotorSimulator
from .ml.failure_synthesizer import FailureSynthesizer
from .control.rl_controller import RLController
from .interfaces.opcua_server import OpcUaServer

__all__ = [
    "DigitalTwin",
    "Config",
    "MotorSimulator",
    "FailureSynthesizer",
    "RLController",
    "OpcUaServer",
]
