"""PhoenixDT - Industrial Digital Twin."""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Ankit Karki"
__email__ = "karkiankit101@gmail.com"
__license__ = "MIT"
__description__ = "Industrial digital twin with motor simulation"
__url__ = "https://github.com/Ankit-x1/HealingSystem"

# Core components
from .core.config import PhoenixConfig
from .core.digital_twin import DigitalTwin, MotorState, ControlSignal
from .api.app import app
from .cli import cli

__all__ = [
    "DigitalTwin",
    "MotorState",
    "ControlSignal",
    "PhoenixConfig",
    "app",
    "cli",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "__url__",
]
