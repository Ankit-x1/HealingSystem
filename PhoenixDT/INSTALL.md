# PhoenixDT - Installation and Testing Guide

## Quick Setup

```bash
# Navigate to PhoenixDT directory
cd PhoenixDT

# Install dependencies
pip install -r requirements.txt

# Install PhoenixDT package
pip install -e .

# Run basic test
python -c "from phoenixdt import DigitalTwin; print(' Installation successful!')"
```

## Running the System

### Option 1: Interactive Dashboard
```bash
streamlit run src/phoenixdt/dashboard/app.py
```
Access at http://localhost:8501

### Option 2: Command Line Digital Twin
```bash
python -m phoenixdt.main --mode twin --duration 60
```

### Option 3: Docker Deployment
```bash
cd deployment/docker
docker-compose up -d
```

## Testing Components

### Test Motor Simulator
```python
from phoenixdt.simulation.motor_simulator import MotorSimulator

sim = MotorSimulator()
state = sim.step(dt=0.01, load_torque=10.0)
print(f"Motor speed: {state['speed_rpm']:.1f} RPM")
print(f"Bearing wear: {state['bearing_wear']:.3f}")
```

### Test Failure Synthesis
```python
from phoenixdt.ml.failure_synthesizer import FailureSynthesizer
from phoenixdt.core.config import MLConfig
import numpy as np

synthesizer = FailureSynthesizer(MLConfig())
state = np.random.randn(8)
failure = synthesizer.generate_failure(state, "bearing_wear", 0.5)
print(f"Generated failure: {failure['type']}")
```

### Test Anomaly Detection
```python
from phoenixdt.ml.anomaly_detector import AnomalyDetector
from phoenixdt.core.config import MLConfig
import numpy as np

detector = AnomalyDetector(MLConfig())
normal_data = np.random.randn(1000, 8)
detector.fit(normal_data)

anomalies = await detector.detect(np.random.randn(8) * 5)  # Anomalous
print(f"Detected {len(anomalies)} anomalies")
```

## Key Features Demonstrated

 **Physics-Based Simulation**: Realistic motor dynamics with bearing degradation  
 **AI-Powered Anomaly Detection**: Multi-algorithm ensemble with uncertainty  
 **Generative Failure Synthesis**: VAE-based synthetic failure generation  
 **Reinforcement Learning Control**: SAC algorithm for adaptive control  
 **Causal Inference**: Explainable AI for root cause analysis  
 **Industrial Integration**: OPC-UA server and REST API  
 **Interactive Dashboard**: Real-time 3D visualization  
 **Production Deployment**: Docker/Kubernetes ready  

## Project Structure

```
PhoenixDT/
├── src/phoenixdt/           # Main source code
│   ├── core/               # Digital twin engine
│   ├── simulation/          # Physics simulator
│   ├── ml/                 # ML components
│   ├── control/             # Control systems
│   ├── interfaces/          # Industrial protocols
│   └── dashboard/          # Web interface
├── tests/                  # Test suite
├── deployment/             # Docker/K8s configs
├── configs/                # Configuration files
└── scripts/               # Setup scripts
```

This is a production-ready industrial digital twin system that combines cutting-edge AI with practical engineering! 🚀