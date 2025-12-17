# PhoenixDT: Failure-Aware Digital Twin 🔥

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](deployment/docker/)

> **PhoenixDT** is a cutting-edge industrial digital twin platform that combines physics-based simulation with AI-powered anomaly detection, self-healing control, and explainable causal inference.

##  Key Features

### AI-Powered Intelligence
- **Generative Failure Synthesis**: VAE-based synthetic failure generation when real data is scarce
- **Multi-Algorithm Anomaly Detection**: Ensemble approach with uncertainty quantification
- **Reinforcement Learning Control**: SAC-based adaptive control for optimal performance
- **Causal Inference**: Explainable AI for root cause analysis

###  High-Fidelity Simulation
- **Physics-Based Motor Model**: Realistic 3-phase induction motor simulation
- **Bearing Degradation Physics**: Progressive wear modeling with thermal effects
- **Real-Time Dynamics**: Sub-millisecond timestep accuracy
- **Multi-Domain Integration**: Electrical, mechanical, and thermal domains

###  Industrial Integration
- **OPC-UA Server**: Industrial protocol support for real-world deployment
- **REST API**: Modern web service interface
- **Prometheus Metrics**: Cloud-native monitoring
- **Docker/Kubernetes**: Production-ready deployment

###  Interactive Visualization
- **3D Motor Visualization**: Real-time animated 3D models
- **Real-Time Dashboard**: Streamlit-based monitoring interface
- **Causal Graphs**: Interactive root cause analysis
- **Performance Analytics**: Historical trend analysis

##  Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ankit-x1/HealingSystem.git
cd HealingSystem/PhoenixDT

# Install dependencies
pip install -r requirements.txt

# Install PhoenixDT
pip install -e .
```

### Basic Usage

```python
from phoenixdt import DigitalTwin, Config

# Initialize digital twin
config = Config()
digital_twin = DigitalTwin(config)

# Start simulation
import asyncio
asyncio.run(digital_twin.start(duration=60))  # Run for 60 seconds

# Get current state
state = digital_twin.get_current_state()
print(f"System health: {state.health_metrics['overall_health']:.2%}")
```

### Docker Deployment

```bash
# Build and run with Docker Compose
cd deployment/docker
docker-compose up -d

# Access dashboard at http://localhost:8501
# Access API at http://localhost:8000
# Access OPC-UA at opc.tcp://localhost:4840
```

##  Documentation

### Architecture Overview

PhoenixDT consists of several integrated components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI/ML Layer   │    │  Simulation     │    │  Interface      │
│                 │    │  Layer          │    │  Layer          │
│ • Failure Synth │◄──►│ • Motor Model   │◄──►│ • OPC-UA Server │
│ • Anomaly Det   │    │ • Bearing Phys  │    │ • REST API      │
│ • Causal Inf    │    │ • Thermal Dyn   │    │ • Dashboard     │
│ • RL Control    │    │ • Vibration     │    │ • Prometheus    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Digital Twin Engine (`core/digital_twin.py`)
- Orchestrates all system components
- Manages simulation state and history
- Provides unified interface for control and monitoring

#### 2. Motor Simulator (`simulation/motor_simulator.py`)
- Physics-based 3-phase induction motor model
- Bearing degradation with thermal effects
- Real-time dynamics with configurable timestep

#### 3. Failure Synthesizer (`ml/failure_synthesizer.py`)
- VAE-based synthetic failure generation
- Multiple failure modes (bearing wear, lubrication loss, etc.)
- Realistic anomaly patterns for training

#### 4. Anomaly Detector (`ml/anomaly_detector.py`)
- Multi-algorithm ensemble (Isolation Forest, Autoencoder, Bayesian)
- Uncertainty quantification for reliable detection
- Real-time processing with configurable thresholds

#### 5. RL Controller (`control/rl_controller.py`)
- Soft Actor-Critic (SAC) implementation
- Continuous control with safety constraints
- Adaptive learning from system feedback

#### 6. Causal Inference (`ml/causal_inference.py`)
- Root cause analysis using DoWhy
- Explainable AI for industrial applications
- Causal graph visualization and analysis

##  Interactive Dashboard

Launch the interactive dashboard:

```bash
streamlit run src/phoenixdt/dashboard/app.py
```

The dashboard provides:

- **Real-time Monitoring**: Live motor parameters and health metrics
- **3D Visualization**: Interactive motor model with animated components
- **Anomaly Detection**: Real-time alerts with detailed explanations
- **Causal Analysis**: Root cause analysis with recommendations
- **Control Interface**: Switch between RL, PID, and manual control
- **Fault Injection**: Test system resilience with synthetic failures

## 🔧 Configuration

PhoenixDT uses a hierarchical configuration system:

```yaml
# config.yaml
simulation:
  dt: 0.001              # Time step (seconds)
  duration: 60.0          # Simulation duration (seconds)
  motor_power: 5.0        # Motor power (kW)
  motor_speed: 1800.0     # Motor speed (RPM)

ml:
  vae_latent_dim: 32      # VAE latent dimension
  rl_algorithm: "SAC"      # RL algorithm
  anomaly_threshold: 0.95  # Anomaly detection threshold

control:
  control_frequency: 100.0  # Control frequency (Hz)
  safety_limits:
    max_current: 50.0      # Maximum current (A)
    max_temperature: 120.0  # Maximum temperature (°C)

interface:
  opcua_port: 4840         # OPC-UA server port
  dashboard_port: 8501      # Dashboard port
  api_port: 8000           # REST API port
```

##  Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=phoenixdt --cov-report=html

# Run specific test categories
pytest tests/test_simulation.py
pytest tests/test_ml.py
pytest tests/test_control.py
```

##  Deployment

### Docker

```bash
# Build image
docker build -f deployment/docker/Dockerfile -t phoenixdt .

# Run container
docker run -p 8000:8000 -p 8501:8501 -p 4840:4840 phoenixdt
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/k8s/phoenixdt-deployment.yaml

# Check deployment
kubectl get pods -n phoenixdt
kubectl get services -n phoenixdt
```

### Cloud Deployment

PhoenixDT is cloud-native and can be deployed on:

- **AWS**: EKS, ECS, or EC2 with Docker
- **Google Cloud**: GKE or Cloud Run
- **Azure**: AKS or Container Instances
- **On-Premise**: Kubernetes cluster or Docker Swarm

##  API Reference

### REST API

```python
import requests

# Get system status
response = requests.get("http://localhost:8000/status")
print(response.json())

# Get current state
response = requests.get("http://localhost:8000/state")
state = response.json()

# Inject fault
response = requests.post(
    "http://localhost:8000/fault",
    json={"type": "bearing_wear", "severity": 0.5}
)
```

### OPC-UA

```python
from asyncua import Client

async def connect_opcua():
    client = Client("opc.tcp://localhost:4840/phoenixdt/server/")
    await client.connect()
    
    # Read motor speed
    speed = await client.get_node("ns=2;s=Motor_Speed").read_value()
    print(f"Motor speed: {speed} RPM")
    
    await client.disconnect()
```

##  Use Cases

### Manufacturing
- **Predictive Maintenance**: Detect bearing failures before they occur
- **Quality Control**: Optimize motor control for consistent production
- **Energy Optimization**: Reduce power consumption while maintaining performance

### Energy Sector
- **Power Generation**: Monitor and control turbine motors
- **Grid Stability**: Adaptive control for frequency regulation
- **Asset Management**: Extend equipment life through optimal operation

### Transportation
- **Electric Vehicles**: Motor health monitoring and optimization
- **Railway Systems**: Predictive maintenance for traction motors
- **Aerospace**: Redundant control systems for critical applications

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/Ankit-x1/HealingSystem.git
cd HealingSystem/PhoenixDT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **PyTorch**: For deep learning frameworks
- **Stable-Baselines3**: For reinforcement learning algorithms  
- **DoWhy**: For causal inference capabilities
- **Streamlit**: For the interactive dashboard
- **asyncua**: For OPC-UA server implementation

##  Contact

- **Author**: Ankit Karki
- **Email**: karkiankit101@gmail.com
- **GitHub**: [@Ankit-x1](https://github.com/Ankit-x1)
- **Project**: https://github.com/Ankit-x1/HealingSystem

##  Roadmap

### Version 0.2 (Q1 2024)
- [ ] Multi-motor digital twin support
- [ ] Advanced failure prediction models
- [ ] Mobile dashboard application
- [ ] Cloud deployment templates

### Version 0.3 (Q2 2024)
- [ ] Integration with industrial IoT platforms
- [ ] Advanced causal discovery algorithms
- [ ] Digital twin export/import functionality
- [ ] Enterprise security features

### Version 1.0 (Q3 2024)
- [ ] Production-ready hardening
- [ ] Comprehensive documentation
- [ ] Performance benchmarking suite
- [ ] Commercial licensing options

---

*** If this project interests you, please give it a star on GitHub!**

Built with  by [Ankit Karki](https://github.com/Ankit-x1)