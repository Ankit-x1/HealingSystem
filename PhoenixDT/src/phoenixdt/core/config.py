"""
Configuration management for PhoenixDT - Python 3.12.2 compatible
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class SimulationConfig(BaseModel):
    """Simulation configuration parameters"""

    dt: float = Field(default=0.001, description="Time step in seconds")
    duration: float = Field(default=60.0, description="Simulation duration in seconds")
    motor_power: float = Field(default=5.0, description="Motor power in kW")
    motor_speed: float = Field(default=1800.0, description="Motor speed in RPM")
    load_torque: float = Field(default=10.0, description="Load torque in Nm")

    @validator("dt")
    def validate_dt(cls, v):
        if v <= 0:
            raise ValueError("Time step must be positive")
        return v


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

    @validator("vae_latent_dim")
    def validate_latent_dim(cls, v):
        if v <= 0:
            raise ValueError("Latent dimension must be positive")
        return v

    @validator("anomaly_threshold")
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Anomaly threshold must be between 0 and 1")
        return v


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

    @validator("control_frequency")
    def validate_frequency(cls, v):
        if v <= 0:
            raise ValueError("Control frequency must be positive")
        return v


class InterfaceConfig(BaseModel):
    """Interface configuration parameters"""

    opcua_port: int = Field(default=4840, description="OPC-UA server port")
    dashboard_port: int = Field(default=8501, description="Dashboard port")
    api_port: int = Field(default=8000, description="REST API port")
    prometheus_port: int = Field(default=9090, description="Prometheus port")

    @validator("opcua_port", "dashboard_port", "api_port", "prometheus_port")
    def validate_ports(cls, v):
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v


class Config(BaseSettings):
    """Main configuration class for PhoenixDT - Python 3.12.2 compatible"""

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
        case_sensitive = False

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        except FileNotFoundError:
            # Return default config if file doesn't exist
            return cls()
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")

    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file"""
        config_dict = self.model_dump()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []

        # Check port conflicts
        ports = [
            self.interface.opcua_port,
            self.interface.dashboard_port,
            self.interface.api_port,
            self.interface.prometheus_port,
        ]
        if len(set(ports)) != len(ports):
            issues.append(
                "Port conflicts detected - all interface ports must be unique"
            )

        # Check data directory
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create data directories: {e}")

        return {"valid": len(issues) == 0, "issues": issues}


# Global configuration instance
config = Config()
