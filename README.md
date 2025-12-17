# PhoenixDT - Industrial Digital Twin

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

PhoenixDT is a simplified industrial digital twin focused on essential motor simulation with real-time monitoring and anomaly detection.

## Features

### Core Capabilities
- Real-time motor simulation with physics-based modeling
- Intelligent anomaly detection with configurable thresholds
- Health monitoring with thermal and mechanical metrics
- Simple PI control with target speed and load torque adjustment
- REST API for integration and monitoring
- WebSocket streaming for real-time data updates
- Command-line interface for system control

## Quick Start

### Installation
```bash
pip install phoenix-dt
```

### Basic Usage

#### CLI Commands
```bash
# Run simulation for 60 seconds
phoenixdt run --duration 60 --speed 1800 --load 10

# Start API server
phoenixdt serve --port 8000

# Check system status
phoenixdt status

# Show configuration
phoenixdt config

# Run quick test
phoenixdt test
```

#### Python API
```python
from phoenixdt import DigitalTwin, PhoenixConfig

# Create digital twin
config = PhoenixConfig()
twin = DigitalTwin(config)

# Set parameters
twin.set_target_speed(1800)  # RPM
twin.set_load_torque(10)     # Nm

# Run simulation
await twin.start(duration=60)

# Get status
status = twin.get_status()
print(f"Motor speed: {status['motor']['speed']} RPM")
print(f"Health: {status['health']['overall']}")
```

## API Endpoints

### Core Endpoints
- `GET /api/status` - System status and motor state
- `POST /api/control` - Set target speed and load torque
- `POST /api/start` - Start simulation
- `POST /api/stop` - Stop simulation
- `GET /api/anomalies` - Get current anomalies
- `GET /api/health` - Health check

### WebSocket
- `WS /ws` - Real-time state updates and anomaly notifications

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   CLI/API       │    │  Digital Twin   │
│                 │◄──►│                 │
│ • REST API      │    │ • Motor Physics │
│ • WebSocket     │    │ • PI Control    │
│ • CLI Commands  │    │ • Health Monitor│
└─────────────────┘    └─────────────────┘
```

### Core Components
- **DigitalTwin**: Main simulation engine
- **MotorState**: Clean state representation
- **ControlSignal**: Control output management
- **Health Monitor**: Real-time health assessment
- **Anomaly Detector**: Threshold-based anomaly detection

## Configuration

```yaml
# phoenixdt.yaml
simulation:
  dt: 0.001              # Time step (seconds)
  motor_power: 5.0        # Motor power (kW)
  motor_speed: 1800.0     # Motor speed (RPM)
  load_torque: 10.0       # Load torque (Nm)

control:
  safety_limits:
    max_current: 50.0      # Maximum current (A)
    max_temperature: 120.0  # Maximum temperature (°C)
    max_vibration: 10.0    # Maximum vibration (mm/s)

interface:
  api_port: 8000          # REST API port
```

## Motor Model

### Physics Simulation
- **Electrical**: Voltage, current, resistance
- **Mechanical**: Torque, speed, inertia, friction
- **Thermal**: Temperature dynamics with cooling
- **Power**: Mechanical and electrical power calculation

### Health Metrics
- **Thermal Health**: Temperature deviation from optimal
- **Mechanical Health**: Vibration level assessment
- **Overall Health**: Weighted combination of all metrics

## Docker Deployment

```bash
# Build image
docker build -t phoenixdt .

# Run container
docker run -p 8000:8000 phoenixdt
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=phoenixdt --cov-report=html

# Run quick system test
phoenixdt test
```

## Performance

- Response Time: < 1ms average
- Memory Usage: < 100MB typical
- CPU Usage: < 10% average
- Simulation Accuracy: Real-time physics modeling

## Development

### Setup
```bash
# Clone repository
git clone https://github.com/Ankit-x1/HealingSystem.git
cd HealingSystem

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
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

## API Examples

### Control Motor
```bash
curl -X POST "http://localhost:8000/api/control" \
  -H "Content-Type: application/json" \
  -d '{"target_speed": 1500, "load_torque": 15}'
```

### Get Status
```bash
curl "http://localhost:8000/api/status"
```

### WebSocket Client
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
};
```

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run quality checks (`pre-commit run --all-files`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

## License

This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Ankit Karki
- **Email**: karkiankit101@gmail.com
- **GitHub**: [@Ankit-x1](https://github.com/Ankit-x1)
- **Project**: https://github.com/Ankit-x1/HealingSystem