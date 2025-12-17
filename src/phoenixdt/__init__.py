"""
PhoenixDT - Industrial Digital Twin

Production-ready industrial digital twin with physics simulation,
AI-powered anomaly detection, and self-healing control.
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Ankit Karki"
__email__ = "karkiankit101@gmail.com"
__phone__ = "+1 6282389233"
__license__ = "MIT"
__description__ = (
    "Industrial digital twin with AI-powered anomaly detection and self-healing control"
)
__url__ = "https://github.com/Ankit-x1/HealingSystem"

# Core Components
from .core.digital_twin import DigitalTwin
from .core.config import PhoenixConfig
from .physics_engine import PhysicsSimulator
from .neural_architectures import AdaptiveNeuralController
from .causal_engine import CausalInferenceEngine

# API and Interfaces
from .api.app import create_app, run_server
from .cli import main as cli_main

# Version and Metadata
__all__ = [
    "DigitalTwin",
    "PhoenixConfig",
    "PhysicsSimulator",
    "AdaptiveNeuralController",
    "CausalInferenceEngine",
    "create_app",
    "run_server",
    "cli_main",
    "__version__",
    "__author__",
    "__email__",
    "__phone__",
    "__license__",
    "__description__",
    "__url__",
]
