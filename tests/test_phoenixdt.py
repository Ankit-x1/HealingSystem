"""
Test suite for PhoenixDT Industrial Digital Twin
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from phoenixdt.core.config import PhoenixConfig
from phoenixdt.core.digital_twin import DigitalTwin


class TestPhoenixConfig:
    """Test PhoenixDT configuration"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = PhoenixConfig()
        assert config.simulation.dt == 0.001
        assert config.ml.vae_latent_dim == 32
        assert config.control.control_frequency == 100.0


class TestDigitalTwin:
    """Test digital twin core functionality"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PhoenixConfig()

    @pytest.fixture
    def digital_twin(self, config):
        """Create digital twin instance"""
        return DigitalTwin(config)

    def test_initialization(self, digital_twin):
        """Test digital twin initialization"""
        assert digital_twin is not None
        assert digital_twin.config is not None
        assert digital_twin.simulator is not None
        assert digital_twin.neural_controller is not None
        assert digital_twin.causal_engine is not None

    @pytest.mark.asyncio
    async def test_start_stop(self, digital_twin):
        """Test start and stop functionality"""
        # Test start
        assert not digital_twin.is_running

        # Start for a short duration
        start_task = asyncio.create_task(digital_twin.start(duration=0.1))

        # Wait a bit for startup
        await asyncio.sleep(0.05)
        assert digital_twin.is_running

        # Stop
        await digital_twin.stop()
        assert not digital_twin.is_running

        # Cancel start task if still running
        start_task.cancel()

    def test_get_current_state(self, digital_twin):
        """Test state retrieval"""
        # Initially should be None
        assert digital_twin.get_current_state() is None

        # Mock a running state
        digital_twin.is_running = True
        digital_twin.current_state = Mock()
        digital_twin.current_state.physical_state = {"speed": 1800.0}

        state = digital_twin.get_current_state()
        assert state is not None
        assert state.physical_state["speed"] == 1800.0

    def test_performance_summary(self, digital_twin):
        """Test performance summary"""
        summary = digital_twin.get_performance_summary()
        assert "uptime" in summary
        assert "avg_efficiency" in summary
        assert "total_anomalies" in summary


class TestAPI:
    """Test API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from phoenixdt.api.app import create_app

        return TestClient(create_app())

    def test_get_status(self, client):
        """Test status endpoint"""
        response = client.get("/api/status")
        assert response.status_code == 503  # Service not initialized

    def test_start_endpoint(self, client):
        """Test start endpoint"""
        response = client.post("/api/start")
        assert response.status_code == 503  # Service not initialized


class TestPhysicsEngine:
    """Test physics simulation engine"""

    def test_physics_initialization(self):
        """Test physics engine initialization"""
        from phoenixdt.physics_engine import PhysicsSimulator

        simulator = PhysicsSimulator()
        assert simulator is not None
        assert simulator.dt == 0.001

    def test_step_simulation(self):
        """Test simulation step"""
        from phoenixdt.physics_engine import PhysicsSimulator

        simulator = PhysicsSimulator()

        # Test step with control input
        import numpy as np

        control_input = np.array([400.0, 400.0, 400.0])

        # This should not raise an exception
        state = asyncio.run(simulator.step(control_input))
        assert state is not None
        assert "speed_rpm" in state or "omega" in state


class TestNeuralController:
    """Test neural controller"""

    def test_neural_initialization(self):
        """Test neural controller initialization"""
        from phoenixdt.neural_architectures import AdaptiveNeuralController

        controller = AdaptiveNeuralController(16, 3, [64, 32])
        assert controller is not None
        assert controller.input_dim == 16
        assert controller.output_dim == 3


class TestCausalEngine:
    """Test causal inference engine"""

    def test_causal_initialization(self):
        """Test causal engine initialization"""
        from phoenixdt.causal_engine import CausalInferenceEngine

        engine = CausalInferenceEngine(8, 5)
        assert engine is not None
        assert engine.n_variables == 8
        assert engine.max_lag == 5


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_simulation_cycle(self):
        """Test complete simulation cycle"""
        config = PhoenixConfig()
        digital_twin = DigitalTwin(config)

        # Test full cycle
        await digital_twin.start(duration=0.1)
        await digital_twin.stop()

        # Should complete without errors
        assert True
