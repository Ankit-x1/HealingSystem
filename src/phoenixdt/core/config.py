"""PhoenixDT configuration module."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class IntegrationMethod(str, Enum):
    """Numerical integration methods."""

    EULER = "euler"
    RK45 = "rk45"


class SimulationConfig(BaseModel):
    """Simulation configuration parameters."""

    dt: float = Field(default=0.001, description="Time step in seconds")
    duration: float = Field(default=60.0, description="Simulation duration in seconds")
    motor_power: float = Field(default=5.0, description="Motor power in kW")
    motor_speed: float = Field(default=1800.0, description="Motor speed in RPM")
    load_torque: float = Field(default=10.0, description="Load torque in Nm")
    integration_method: IntegrationMethod = Field(
        default=IntegrationMethod.RK45, description="Numerical integration method"
    )


class ControlConfig(BaseModel):
    """Control system configuration parameters."""

    control_frequency: float = Field(
        default=100.0, description="Control frequency in Hz"
    )
    safety_limits: dict[str, float] = Field(
        default={"max_current": 50.0, "max_temperature": 120.0, "max_vibration": 10.0},
        description="Safety limits for control variables",
    )
    pid_gains: dict[str, float] = Field(
        default={"kp": 1.0, "ki": 0.1, "kd": 0.01}, description="PID controller gains"
    )


class InterfaceConfig(BaseModel):
    """Interface configuration parameters."""

    api_port: int = Field(default=8000, description="REST API port")


class PhoenixConfig(BaseModel):
    """Main configuration class for PhoenixDT."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> PhoenixConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def to_yaml(self, config_path: str | Path | None = None) -> str:
        """Export configuration to YAML string."""
        config_dict = self.model_dump()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        if config_path:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(yaml_str)

        return yaml_str
