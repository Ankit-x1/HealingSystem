"""
PhoenixDT Configuration - Apple/Tesla Grade

Next-generation configuration with quantum parameters,
neural architecture settings, and production optimizations.
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
    QUANTUM = "quantum"
    ADAPTIVE = "adaptive"


class QuantumConfig(BaseModel):
    """Quantum system configuration"""

    state_dim: int = Field(default=8, description="Quantum state dimension")
    coherence_time: float = Field(
        default=1.0, description="Quantum coherence time (seconds)"
    )
    decoherence_rate: float = Field(
        default=0.01, description="Quantum decoherence rate"
    )
    entanglement_strength: float = Field(
        default=0.5, description="Quantum entanglement strength"
    )
    superposition_capacity: int = Field(
        default=8, description="Quantum superposition capacity"
    )
    measurement_precision: float = Field(
        default=0.95, description="Quantum measurement precision"
    )
    uncertainty_quantization: float = Field(
        default=0.01, description="Uncertainty quantization level"
    )


class NeuralConfig(BaseModel):
    """Neural architecture configuration"""

    input_dim: int = Field(default=16, description="Neural network input dimension")
    output_dim: int = Field(default=3, description="Neural network output dimension")
    hidden_dims: List[int] = Field(
        default=[256, 128, 64], description="Hidden layer dimensions"
    )
    activation: str = Field(default="swish", description="Activation function")
    dropout_rate: float = Field(default=0.1, description="Dropout rate")
    batch_norm: bool = Field(default=True, description="Use batch normalization")
    residual_connections: bool = Field(
        default=True, description="Use residual connections"
    )
    attention_heads: int = Field(default=8, description="Multi-head attention heads")
    quantum_layers: bool = Field(
        default=True, description="Use quantum-inspired layers"
    )
    nas_enabled: bool = Field(
        default=True, description="Enable neural architecture search"
    )
    meta_learning: bool = Field(default=True, description="Enable meta-learning")


class PhysicsConfig(BaseModel):
    """Physics simulation configuration"""

    dt: float = Field(default=0.001, description="Base time step (seconds)")
    integration_method: IntegrationMethod = Field(
        default=IntegrationMethod.RK45, description="Integration method"
    )
    adaptive_timestep: bool = Field(
        default=True, description="Enable adaptive timestep"
    )
    max_timestep: float = Field(default=0.01, description="Maximum timestep (seconds)")
    min_timestep: float = Field(default=1e-6, description="Minimum timestep (seconds)")
    error_tolerance: float = Field(
        default=1e-6, description="Integration error tolerance"
    )
    material_degradation: bool = Field(
        default=True, description="Enable material degradation"
    )
    thermal_coupling: bool = Field(default=True, description="Enable thermal coupling")
    sensor_noise: bool = Field(default=True, description="Enable sensor noise modeling")


class CausalConfig(BaseModel):
    """Causal inference configuration"""

    n_variables: int = Field(default=8, description="Number of causal variables")
    max_lag: int = Field(default=5, description="Maximum causal lag")
    discovery_algorithm: str = Field(
        default="quantum_enhanced", description="Causal discovery algorithm"
    )
    real_time_learning: bool = Field(
        default=True, description="Enable real-time causal learning"
    )
    intervention_modeling: bool = Field(
        default=True, description="Enable intervention modeling"
    )
    counterfactual_reasoning: bool = Field(
        default=True, description="Enable counterfactual reasoning"
    )
    uncertainty_quantification: bool = Field(
        default=True, description="Enable causal uncertainty"
    )


class PredictionConfig(BaseModel):
    """Predictive analytics configuration"""

    horizon_steps: int = Field(default=10, description="Prediction horizon steps")
    update_interval: float = Field(
        default=1.0, description="Prediction update interval (seconds)"
    )
    confidence_threshold: float = Field(
        default=0.95, description="Prediction confidence threshold"
    )
    uncertainty_propagation: bool = Field(
        default=True, description="Enable uncertainty propagation"
    )
    ensemble_prediction: bool = Field(
        default=True, description="Use ensemble prediction"
    )
    temporal_fusion: bool = Field(default=True, description="Enable temporal fusion")


class HealingConfig(BaseModel):
    """Self-healing configuration"""

    threshold: float = Field(default=0.3, description="Healing activation threshold")
    check_interval: float = Field(
        default=0.5, description="Healing check interval (seconds)"
    )
    aggressiveness: float = Field(
        default=0.5, description="Healing aggressiveness (0-1)"
    )
    quantum_optimization: bool = Field(
        default=True, description="Use quantum optimization"
    )
    adaptive_strategies: bool = Field(
        default=True, description="Enable adaptive healing strategies"
    )
    recovery_time_target: float = Field(
        default=30.0, description="Target recovery time (seconds)"
    )


class AnomalyConfig(BaseModel):
    """Anomaly detection configuration"""

    threshold: float = Field(default=0.1, description="Anomaly detection threshold")
    quantum_enhanced: bool = Field(
        default=True, description="Use quantum-enhanced detection"
    )
    uncertainty_aware: bool = Field(
        default=True, description="Uncertainty-aware detection"
    )
    multivariate: bool = Field(
        default=True, description="Multivariate anomaly detection"
    )
    temporal_consistency: bool = Field(
        default=True, description="Temporal consistency checking"
    )


class SystemConfig(BaseModel):
    """System configuration"""

    max_workers: int = Field(default=4, description="Maximum worker threads")
    history_size: int = Field(default=1000, description="State history size")
    log_level: str = Field(default="INFO", description="Logging level")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    profiling_enabled: bool = Field(
        default=False, description="Enable performance profiling"
    )
    debug_mode: bool = Field(default=False, description="Enable debug mode")


class InterfaceConfig(BaseModel):
    """Interface configuration"""

    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    websocket_port: int = Field(default=8000, description="WebSocket port")
    opcua_enabled: bool = Field(default=True, description="Enable OPC-UA interface")
    opcua_port: int = Field(default=4840, description="OPC-UA server port")
    prometheus_enabled: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )
    prometheus_port: int = Field(default=9090, description="Prometheus port")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")


class PhoenixConfig(BaseSettings):
    """
    Apple/Tesla-grade PhoenixDT configuration

    Features:
    - Quantum-enhanced parameters
    - Neural architecture search
    - Real-time causal inference
    - Self-healing capabilities
    - Production optimizations
    """

    # Core configurations
    quantum: QuantumConfig = Field(default_factory=QuantumConfig)
    neural: NeuralConfig = Field(default_factory=NeuralConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    causal: CausalConfig = Field(default_factory=CausalConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    healing: HealingConfig = Field(default_factory=HealingConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)

    class Config:
        env_prefix = "PHOENIXDT"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("quantum", pre=True)
    def validate_quantum(cls, v):
        """Validate quantum configuration"""
        if v.state_dim < 2 or v.state_dim > 16:
            raise ValueError("Quantum state dimension must be between 2 and 16")
        if v.coherence_time <= 0:
            raise ValueError("Quantum coherence time must be positive")
        return v

    @validator("neural", pre=True)
    def validate_neural(cls, v):
        """Validate neural configuration"""
        if v.input_dim <= 0 or v.output_dim <= 0:
            raise ValueError("Neural dimensions must be positive")
        if not v.hidden_dims:
            raise ValueError("Hidden dimensions cannot be empty")
        return v

    @validator("physics", pre=True)
    def validate_physics(cls, v):
        """Validate physics configuration"""
        if v.dt <= 0 or v.dt > 1.0:
            raise ValueError("Physics timestep must be between 0 and 1 second")
        if v.min_timestep >= v.max_timestep:
            raise ValueError("Min timestep must be less than max timestep")
        return v

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

    def get_effective_config(self) -> Dict[str, Any]:
        """Get effective configuration with all defaults applied"""
        return {
            "quantum": self.quantum.dict(),
            "neural": self.neural.dict(),
            "physics": self.physics.dict(),
            "causal": self.causal.dict(),
            "prediction": self.prediction.dict(),
            "healing": self.healing.dict(),
            "anomaly": self.anomaly.dict(),
            "system": self.system.dict(),
            "interface": self.interface.dict(),
        }

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check quantum configuration
        if self.quantum.state_dim > 12:
            issues.append("High quantum state dimension may impact performance")

        # Check neural configuration
        total_params = self.neural.input_dim * self.neural.hidden_dims[0]
        if total_params > 100000:
            issues.append("Large neural network may impact performance")

        # Check physics configuration
        if self.physics.dt < 1e-6:
            issues.append("Very small timestep may cause numerical instability")

        # Check system configuration
        if self.system.max_workers > 16:
            issues.append("High worker count may impact system resources")

        return issues

    def get_performance_profile(self) -> str:
        """Get performance profile based on configuration"""
        complexity_score = 0

        # Quantum complexity
        complexity_score += self.quantum.state_dim * 2
        if self.quantum.entanglement_strength > 0.7:
            complexity_score += 10

        # Neural complexity
        complexity_score += sum(self.neural.hidden_dims) / 100
        if self.neural.attention_heads > 8:
            complexity_score += 5

        # Physics complexity
        if self.physics.adaptive_timestep:
            complexity_score += 3
        if self.physics.integration_method == IntegrationMethod.QUANTUM:
            complexity_score += 10

        # Determine profile
        if complexity_score < 20:
            return "lightweight"
        elif complexity_score < 50:
            return "standard"
        elif complexity_score < 100:
            return "high_performance"
        else:
            return "enterprise"


# Default configuration for different use cases
def get_default_config(profile: str = "standard") -> PhoenixConfig:
    """Get default configuration for different profiles"""

    if profile == "lightweight":
        return PhoenixConfig(
            quantum=QuantumConfig(state_dim=4, coherence_time=0.5),
            neural=NeuralConfig(hidden_dims=[64, 32], attention_heads=4),
            physics=PhysicsConfig(dt=0.01, adaptive_timestep=False),
            system=SystemConfig(max_workers=2, history_size=100),
        )

    elif profile == "high_performance":
        return PhoenixConfig(
            quantum=QuantumConfig(
                state_dim=12, coherence_time=2.0, entanglement_strength=0.8
            ),
            neural=NeuralConfig(hidden_dims=[512, 256, 128], attention_heads=16),
            physics=PhysicsConfig(
                dt=0.0001, integration_method=IntegrationMethod.QUANTUM
            ),
            system=SystemConfig(
                max_workers=8, history_size=5000, profiling_enabled=True
            ),
        )

    elif profile == "enterprise":
        return PhoenixConfig(
            quantum=QuantumConfig(
                state_dim=16, coherence_time=5.0, entanglement_strength=0.9
            ),
            neural=NeuralConfig(hidden_dims=[1024, 512, 256, 128], attention_heads=32),
            physics=PhysicsConfig(
                dt=0.00001, integration_method=IntegrationMethod.QUANTUM
            ),
            system=SystemConfig(
                max_workers=16, history_size=10000, profiling_enabled=True
            ),
            interface=InterfaceConfig(cors_origins=["https://enterprise.com"]),
        )

    else:  # standard
        return PhoenixConfig()


# Configuration validation and utilities
def validate_config_file(config_path: Union[str, Path]) -> bool:
    """Validate configuration file"""
    try:
        config = PhoenixConfig.from_yaml(config_path)
        issues = config.validate_configuration()

        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("Configuration is valid")
        return True

    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def create_sample_config(output_path: Optional[Union[str, Path]] = None) -> None:
    """Create sample configuration file"""
    config = get_default_config("standard")

    if output_path:
        config.to_yaml(output_path)
        print(f"Sample configuration created at: {output_path}")
    else:
        print(config.to_yaml())
