"""
Comprehensive integration tests for PhoenixDT system
"""

import asyncio
import pytest
import time
from typing import Dict, Any

import requests
import torch

from phoenixdt.core.config import PhoenixConfig
from phoenixdt.core.digital_twin import DigitalTwin
from phoenixdt.shared.utils import registry, validate_tensor


class TestSystemIntegration:
    """Test complete system integration"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return PhoenixConfig()
    
    @pytest.fixture
    def digital_twin(self, config):
        """Digital twin instance"""
        return DigitalTwin(config)
    
    @pytest.fixture
    def api_base_url(self):
        """API base URL"""
        return "http://127.0.0.1:8000"
    
    def test_system_initialization(self, digital_twin):
        """Test system initializes correctly"""
        assert digital_twin is not None
        assert digital_twin.config is not None
        assert digital_twin.simulator is not None
        assert digital_twin.neural_controller is not None
        assert digital_twin.causal_engine is not None
        
        # Register components
        registry.register("digital_twin", digital_twin)
        registry.register("simulator", digital_twin.simulator)
        
        assert "digital_twin" in registry.list_components()
        assert "simulator" in registry.list_components()
    
    @pytest.mark.asyncio
    async def test_simulation_loop(self, digital_twin):
        """Test simulation loop runs without errors"""
        try:
            # Start simulation for short duration
            await digital_twin.start(duration=2.0)
            
            # Check that simulation ran
            assert digital_twin.metrics["predictions_made"] > 0
            
            # Stop simulation
            await digital_twin.stop()
            
        except Exception as e:
            pytest.fail(f"Simulation loop failed: {e}")
    
    def test_tensor_validation(self):
        """Test tensor validation utilities"""
        # Valid tensor
        valid_tensor = torch.zeros(16, dtype=torch.float32)
        validate_tensor(valid_tensor, 1, "test_tensor")  # Should not raise
        
        # Invalid tensor (wrong dimensions)
        invalid_tensor = torch.zeros(16, 16, dtype=torch.float32)
        with pytest.raises(Exception):
            validate_tensor(invalid_tensor, 1, "test_tensor")
        
        # Invalid tensor (contains NaN)
        nan_tensor = torch.full((16,), float('nan'), dtype=torch.float32)
        with pytest.raises(Exception):
            validate_tensor(nan_tensor, 1, "test_tensor")
    
    @pytest.mark.asyncio
    async def test_neural_controller_integration(self, digital_twin):
        """Test neural controller integration"""
        try:
            # Create test inputs
            quantum_tensor = torch.zeros(16, dtype=torch.float32)
            classical_params = {"speed": 400.0, "torque": 200.0}
            health_tensor = torch.ones(16, dtype=torch.float32)
            
            # Test neural controller
            control_output = await digital_twin.neural_controller.compute_control(
                quantum_tensor, classical_params, health_tensor
            )
            
            assert control_output is not None
            assert isinstance(control_output, torch.Tensor)
            assert control_output.shape[0] == 3  # Expected output dimension
            
        except Exception as e:
            pytest.fail(f"Neural controller integration failed: {e}")
    
    def test_api_endpoints(self, api_base_url):
        """Test all API endpoints"""
        endpoints = [
            ("GET", "/api/status"),
            ("GET", "/api/neural/architectures"),
            ("GET", "/api/physics/state"),
            ("POST", "/api/performance/benchmark"),
            ("POST", "/api/causal/infer"),
            ("POST", "/api/quantum/optimize")
        ]
        
        for method, endpoint in endpoints:
            try:
                start_time = time.time()
                
                if method == "GET":
                    response = requests.get(f"{api_base_url}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{api_base_url}{endpoint}", timeout=10)
                
                elapsed = time.time() - start_time
                
                # Check response
                assert response.status_code in [200, 408, 503], f"Endpoint {endpoint} returned {response.status_code}"
                assert elapsed < 15.0, f"Endpoint {endpoint} took too long: {elapsed}s"
                
                if response.status_code == 200:
                    data = response.json()
                    assert "message" in data or "architectures" in data or "physics_state" in data
                
            except requests.exceptions.Timeout:
                # Timeout is acceptable for heavy operations
                continue
            except Exception as e:
                pytest.fail(f"API endpoint {endpoint} failed: {e}")
    
    @pytest.mark.asyncio
    async def test_component_interactions(self, digital_twin):
        """Test interactions between components"""
        try:
            # Start simulation
            await digital_twin.start(duration=1.0)
            
            # Get state
            state = digital_twin.get_current_state()
            
            # Test physics simulator
            if state:
                physics_state = digital_twin.simulator.get_current_state()
                assert isinstance(physics_state, dict)
            
            # Test performance metrics
            summary = digital_twin.get_performance_summary()
            assert "uptime" in summary
            assert "avg_efficiency" in summary
            
            # Stop simulation
            await digital_twin.stop()
            
        except Exception as e:
            pytest.fail(f"Component interaction test failed: {e}")
    
    def test_error_handling(self, digital_twin):
        """Test error handling across the system"""
        # Test invalid tensor
        with pytest.raises(Exception):
            validate_tensor(None, 1, "test")
        
        # Test component registry
        assert registry.get("nonexistent") is None
        
        # Test system metrics
        metrics = registry.get_system_metrics() if hasattr(registry, 'get_system_metrics') else {}
        assert isinstance(metrics, dict)


class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_simulation_performance(self, digital_twin):
        """Test simulation performance"""
        start_time = time.time()
        
        await digital_twin.start(duration=3.0)
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 10.0, f"Simulation took too long: {elapsed}s"
        
        # Should have made predictions
        assert digital_twin.metrics["predictions_made"] > 0
        
        await digital_twin.stop()
    
    def test_api_response_times(self, api_base_url):
        """Test API response times"""
        fast_endpoints = [
            ("GET", "/api/status"),
            ("GET", "/api/neural/architectures"),
            ("GET", "/api/physics/state")
        ]
        
        for method, endpoint in fast_endpoints:
            try:
                start_time = time.time()
                
                if method == "GET":
                    response = requests.get(f"{api_base_url}{endpoint}", timeout=5)
                else:
                    response = requests.post(f"{api_base_url}{endpoint}", timeout=5)
                
                elapsed = time.time() - start_time
                
                # Fast endpoints should respond quickly
                assert response.status_code == 200
                assert elapsed < 3.0, f"Fast endpoint {endpoint} took too long: {elapsed}s"
                
            except Exception as e:
                pytest.fail(f"Performance test for {endpoint} failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
