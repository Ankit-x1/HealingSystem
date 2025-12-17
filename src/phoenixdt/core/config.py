"""
PhoenixDT Core Configuration

Production-ready configuration for industrial digital twin.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from enum import Enum


class IntegrationMethod(str, Enum):
    """Numerical integration methods"""

    EULER = "euler"
    RK45 = "rk45"
    ADAPTIVE = "adaptive"


class SimulationConfig(BaseModel):
    """Simulation configuration parameters"""

    dt: float = Field(default=0.001, description="Time step in seconds")
    duration: float = Field(default=60.0, description="Simulation duration in seconds")
    motor_power: float = Field(default=5.0, description="Motor power in kW")
    motor_speed: float = Field(default=1800.0, description="Motor speed in RPM")
    load_torque: float = Field(default=10.0, description="Load torque in Nm")
    integration_method: IntegrationMethod = Field(
        default=IntegrationMethod.RK45, description="Numerical integration method"
    )


class MLConfig(BaseModel):
    """Machine learning configuration parameters"""

    vae_latent_dim: int = Field(default=32, description="VAE latent dimension")
    vae_hidden_dims: list = Field(
        default=[256, 128, 64], description="VAE hidden layers"
    )
    rl_algorithm: str = Field(default="SAC", description="RL algorithm to use")
    rl_learning_rate: float = Field(default=3e-4, description="RL learning rate")
    anomaly_threshold: float = Field(
        default=0.95, description="Anomaly detection threshold"
    )


class ControlConfig(BaseModel):
    """Control system configuration parameters"""

    control_frequency: float = Field(
        default=100.0, description="Control frequency in Hz"
    )
    safety_limits: Dict[str, float] = Field(
        default={"max_current": 50.0, "max_temperature": 120.0, "max_vibration": 10.0},
        description="Safety limits for control variables",
    )
    pid_gains: Dict[str, float] = Field(
        default={"kp": 1.0, "ki": 0.1, "kd": 0.01}, description="PID controller gains"
    )


class InterfaceConfig(BaseModel):
    """Interface configuration parameters"""

    api_port: int = Field(default=8000, description="REST API port")
    opcua_port: int = Field(default=4840, description="OPC-UA server port")
    dashboard_port: int = Field(default=8501, description="Dashboard port")
    prometheus_port: int = Field(default=9090, description="Prometheus port")


class PhoenixConfig(BaseModel):
    """Main configuration class for PhoenixDT"""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)

    class Config:
        env_prefix = "PHOENIXDT"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "PhoenixConfig":
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def to_yaml(self, config_path: Optional[Union[str, Path]] = None) -> str:
        """Export configuration to YAML string"""
        config_dict = self.dict()

        # Remove sensitive information
        if "password" in config_dict:
            del config_dict["password"]

        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        if config_path:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(yaml_str)

        return yaml_str
