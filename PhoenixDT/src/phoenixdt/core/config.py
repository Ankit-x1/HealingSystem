"""
Configuration management for PhoenixDT
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class SimulationConfig(BaseModel):
    """Simulation configuration parameters"""

    dt: float = Field(default=0.001, description="Time step in seconds")
    duration: float = Field(default=60.0, description="Simulation duration in seconds")
    motor_power: float = Field(default=5.0, description="Motor power in kW")
    motor_speed: float = Field(default=1800.0, description="Motor speed in RPM")
    load_torque: float = Field(default=10.0, description="Load torque in Nm")


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

    opcua_port: int = Field(default=4840, description="OPC-UA server port")
    dashboard_port: int = Field(default=8501, description="Dashboard port")
    api_port: int = Field(default=8000, description="REST API port")
    prometheus_port: int = Field(default=9090, description="Prometheus port")


class Config(BaseSettings):
    """Main configuration class for PhoenixDT"""

    simulation: SimulationConfig = SimulationConfig()
    ml: MLConfig = MLConfig()
    control: ControlConfig = ControlConfig()
    interface: InterfaceConfig = InterfaceConfig()

    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    # Data paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    model_dir: Path = Field(default=Path("data/models"), description="Model directory")

    class Config:
        env_prefix = "PHOENIXDT_"
        env_file = ".env"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file"""
        config_dict = self.model_dump()
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Global configuration instance
config = Config()
