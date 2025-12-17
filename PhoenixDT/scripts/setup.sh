#!/bin/bash

# PhoenixDT Development Setup Script - Python 3.12.2 compatible

set -e

echo "🔥 Setting up PhoenixDT Development Environment for Python 3.12.2..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and setuptools
echo "⬆️ Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install dependencies with Python 3.12.2 compatibility
echo "📚 Installing Python 3.12.2 compatible dependencies..."

# Install core packages first
echo "Installing core scientific packages..."
pip install numpy>=1.24.0 scipy>=1.10.0 pandas>=2.0.0 matplotlib>=3.7.0

echo "Installing machine learning packages..."
pip install torch>=2.0.0 torchvision>=0.15.0 scikit-learn>=1.3.0
pip install xgboost>=1.7.0 lightgbm>=3.3.0

echo "Installing reinforcement learning packages..."
pip install stable-baselines3>=2.0.0 gymnasium>=0.28.0 ray>=2.5.0

echo "Installing causal inference packages..."
pip install dowhy>=0.11.0 econml>=0.13.0 causalml>=0.12.0

echo "Installing industrial communication packages..."
pip install asyncua>=1.0.0 pymodbus>=3.4.0

echo "Installing dashboard and visualization packages..."
pip install streamlit>=1.25.0 plotly>=5.15.0 dash>=2.11.0 bokeh>=3.2.0

echo "Installing deployment and monitoring packages..."
pip install fastapi>=0.100.0 uvicorn>=0.23.0 pydantic>=2.0.0
pip install pydantic-settings>=2.0.0 prometheus-client>=0.17.0 psutil>=5.9.0

echo "Installing utility packages..."
pip install python-dotenv>=1.0.0 loguru>=0.7.0 typer>=0.9.0 rich>=13.4.0

echo "Installing testing and development packages..."
pip install pytest>=7.4.0 pytest-asyncio>=0.21.0 pytest-cov>=4.1.0
pip install black>=23.7.0 isort>=5.12.0 flake8>=6.0.0 mypy>=1.5.0

# Install PhoenixDT in development mode
echo "🔧 Installing PhoenixDT in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/models data/samples logs

# Run basic tests to verify installation
echo "🧪 Running basic verification tests..."

# Test core imports
python3 -c "
import sys
print('Python version:', sys.version)

try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except ImportError as e:
    print('❌ NumPy import failed:', e)

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
except ImportError as e:
    print('❌ PyTorch import failed:', e)

try:
    import streamlit
    print('✅ Streamlit:', streamlit.__version__)
except ImportError as e:
    print('❌ Streamlit import failed:', e)

try:
    from phoenixdt import DigitalTwin
    print('✅ PhoenixDT imported successfully!')
except ImportError as e:
    print('❌ PhoenixDT import failed:', e)
"

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Docker detected - you can use Docker deployment"
else
    echo "⚠️ Docker not found - Docker deployment unavailable"
fi

# Check if CUDA is available
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "🚀 CUDA detected - GPU acceleration available"
else
    echo "⚠️ CUDA not detected - using CPU only"
fi

echo ""
echo "🎉 PhoenixDT development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Ensure virtual environment is active: source venv/bin/activate"
echo "2. Run dashboard: streamlit run src/phoenixdt/dashboard/app.py"
echo "3. Or run digital twin: python -m phoenixdt.main --mode twin"
echo "4. Or use Docker: cd deployment/docker && docker-compose up"
echo ""
echo "📖 For more information, see README.md"
echo ""
echo "🔥 PhoenixDT is ready for Python 3.12.2!"