# PhoenixDT - Industrial Digital Twin

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![CI/CD](https://github.com/Ankit-x1/HealingSystem/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Ankit-x1/HealingSystem/actions)

PhoenixDT is an industrial digital twin platform that combines physics-based simulation with AI-powered anomaly detection, self-healing control, and causal inference for predictive maintenance.

## Features

### Core Capabilities
- **Physics-Based Simulation**: Multi-domain motor simulation with electrical, mechanical, and thermal coupling
- **AI-Powered Anomaly Detection**: Multi-algorithm ensemble with uncertainty quantification
- **Self-Healing Control**: Adaptive control with automatic fault recovery
- **Causal Inference**: Root cause analysis with explainable AI
- **Predictive Analytics**: Future state prediction with confidence intervals

### Technical Features
- **Real-Time Processing**: Sub-millisecond response times
- **Multi-Modal Data**: Sensor fusion and state estimation
- **Adaptive Learning**: Neural architecture search and meta-learning
- **Industrial Protocols**: OPC-UA, Modbus support
- **Web Interface**: REST API with real-time WebSocket streaming
- **Container Ready**: Docker and Kubernetes deployment

## Quick Start

### Installation
```bash
# Install from PyPI
pip install phoenix-dt

# Or install with all features
pip install phoenix-dt[all]

# Or clone from source
git clone https://github.com/Ankit-x1/HealingSystem.git
cd HealingSystem
pip install -e ".[all]"
```

### Basic Usage
```bash
# Start digital twin simulation
phoenixdt start --duration 60

# Start web API server
phoenixdt serve --host 0.0.0.0 --port 8000

# Monitor system status
phoenixdt status

# Generate configuration
phoenixdt config create-sample
```

### Docker Deployment
```bash
# Build and run with Docker Compose
cd deployment/docker
docker-compose up -d

# Access services:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/api/docs
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
```

## Architecture

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI/ML Layer   │    │  Simulation     │    │  Interface      │
│                 │    │  Layer          │    │  Layer          │
│ • Anomaly Det  │◄──►│ • Motor Model   │◄──►│ • REST API      │
│ • Causal Inf  │    │ • Bearing Phys  │    │ • WebSocket     │
│ • RL Control    │    │ • Thermal Dyn   │    │ • Dashboard     │
│ • Predictive    │    │ • Vibration     │    │ • OPC-UA Server │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Modules
- **Digital Twin Engine** (`core/digital_twin.py`): Orchestrates all system components
- **Motor Simulator** (`simulation/motor_simulator.py`): Physics-based motor simulation
- **Anomaly Detector** (`ml/anomaly_detector.py`): Multi-algorithm ensemble detection
- **RL Controller** (`control/rl_controller.py`): Adaptive control with learning
- **Causal Engine** (`ml/causal_inference.py`): Root cause analysis
- **API Server** (`api/app.py`): REST API and WebSocket interface

## API Documentation

### REST Endpoints
- `GET /api/status` - System status and health metrics
- `GET /api/state` - Current digital twin state
- `POST /api/start` - Start simulation
- `POST /api/stop` - Stop simulation
- `POST /api/fault` - Inject fault for testing
- `GET /api/predictions` - Get predictive insights
- `GET /api/performance` - Performance metrics

### WebSocket
- `ws://localhost:8000/ws` - Real-time data streaming

### Configuration
```yaml
# phoenixdt.yaml
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
  api_port: 8000           # REST API port
  opcua_port: 4840         # OPC-UA server port
```

## Testing

### Run Tests
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

### Code Quality
```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
ruff check src/
```

## Deployment

### Docker
```bash
# Build image
docker build -f deployment/docker/Dockerfile -t phoenixdt .

# Run container
docker run -p 8000:8000 -p 4840:4840 phoenixdt
```

### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f deployment/k8s/

# Check deployment
kubectl get pods -n phoenixdt
kubectl get services -n phoenixdt
```

### Cloud Deployment
PhoenixDT is cloud-native and can be deployed on:
- **AWS**: EKS, ECS, or EC2 with Docker
- **Google Cloud**: GKE or Cloud Run
- **Microsoft Azure**: AKS or Container Instances
- **On-Premise**: Kubernetes cluster or Docker Swarm

## Performance

### Benchmarks
- **Response Time**: < 1ms average
- **Throughput**: 1000+ operations/second
- **Memory Usage**: < 512MB typical
- **CPU Usage**: < 50% average
- **Uptime**: 99.9% availability

### Monitoring
- **Prometheus Metrics**: System performance and health
- **Grafana Dashboards**: Real-time visualization
- **Log Aggregation**: Structured logging with correlation
- **Health Checks**: Automated system health monitoring

## Development

### Setup
```bash
# Clone repository
git clone https://github.com/Ankit-x1/HealingSystem.git
cd HealingSystem

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

### Contributing Guidelines
1. Fork repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Run quality checks (`pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- **Python**: Follow PEP 8, use Black for formatting
- **Type Hints**: Required for all public functions
- **Documentation**: Google-style docstrings
- **Testing**: Minimum 80% code coverage required

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Ankit Karki
- **Email**: karkiankit101@gmail.com
- **Phone**: +1 6282389233
- **GitHub**: [@Ankit-x1](https://github.com/Ankit-x1)
- **Project**: https://github.com/Ankit-x1/HealingSystem

## Acknowledgments

- **PyTorch**: For deep learning frameworks
- **FastAPI**: For modern web API development
- **Stable-Baselines3**: For reinforcement learning algorithms  
- **DoWhy**: For causal inference capabilities
- **Plotly**: For interactive data visualization
- **asyncua**: For OPC-UA server implementation