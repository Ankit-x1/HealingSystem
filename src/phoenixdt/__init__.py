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

# API and Interfaces
from .api.app import app
from .cli import cli
from .core.config import PhoenixConfig

# Core Components
from .core.digital_twin import DigitalTwin


# Version and Metadata
__all__ = [
    "DigitalTwin",
    "PhoenixConfig",
    "app",
    "cli",
    "__version__",
    "__author__",
    "__email__",
    "__phone__",
    "__license__",
    "__description__",
    "__url__",
]
