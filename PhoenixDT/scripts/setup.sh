#!/bin/bash

# PhoenixDT Development Setup Script

set -e

echo "🔥 Setting up PhoenixDT Development Environment..."

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

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install PhoenixDT in development mode
echo "🔧 Installing PhoenixDT in development mode..."
pip install -e .

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/models data/samples logs

# Run tests to verify installation
echo "🧪 Running tests to verify installation..."
pytest tests/ -v

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
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the dashboard: streamlit run src/phoenixdt/dashboard/app.py"
echo "3. Or run the digital twin: python -m phoenixdt.main --mode twin"
echo "4. Or use Docker: cd deployment/docker && docker-compose up"
echo ""
echo "📖 For more information, see README.md"