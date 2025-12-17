"""
Shared utilities for PhoenixDT system
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class PhoenixDTError(Exception):
    """Base exception for PhoenixDT system"""
    pass


class QuantumError(PhoenixDTError):
    """Quantum computation related errors"""
    pass


class NeuralError(PhoenixDTError):
    """Neural network related errors"""
    pass


class PhysicsError(PhoenixDTError):
    """Physics simulation related errors"""
    pass


async def timeout_wrapper(coro, timeout_seconds: float, operation_name: str):
    """Wrap coroutine with timeout and error handling"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"{operation_name} timed out after {timeout_seconds} seconds")
        raise PhoenixDTError(f"{operation_name} timed out")
    except Exception as e:
        logger.error(f"{operation_name} failed: {e}")
        raise PhoenixDTError(f"{operation_name} failed: {e}") from e


def validate_tensor(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    """Validate tensor dimensions and properties"""
    if not isinstance(tensor, torch.Tensor):
        raise NeuralError(f"{name} must be a torch.Tensor")
    
    if tensor.dim() != expected_dim:
        raise NeuralError(f"{name} must have {expected_dim} dimensions, got {tensor.dim()}")
    
    if torch.isnan(tensor).any():
        raise NeuralError(f"{name} contains NaN values")


def create_default_control_vector(size: int = 16) -> torch.Tensor:
    """Create default control vector with proper dimensions"""
    return torch.zeros(size, dtype=torch.float32)


def get_system_metrics() -> Dict[str, Any]:
    """Get current system performance metrics"""
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "timestamp": time.time()
        }
    except ImportError:
        return {"timestamp": time.time(), "cpu_percent": 0, "memory_percent": 0}


class ComponentRegistry:
    """Registry for system components"""
    
    def __init__(self):
        self._components: Dict[str, Any] = {}
    
    def register(self, name: str, component: Any) -> None:
        """Register a component"""
        self._components[name] = component
        logger.info(f"Registered component: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """Get a registered component"""
        return self._components.get(name)
    
    def list_components(self) -> list[str]:
        """List all registered components"""
        return list(self._components.keys())


# Global component registry
registry = ComponentRegistry()
