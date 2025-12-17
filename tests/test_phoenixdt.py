"""Tests for PhoenixDT."""

import pytest
import asyncio
from phoenixdt import DigitalTwin, PhoenixConfig


class TestDigitalTwin:
    """Test digital twin functionality."""

    def test_initialization(self):
        """Test digital twin initialization."""
        config = PhoenixConfig()
        twin = DigitalTwin(config)

        assert twin.state.speed == 0.0
        assert twin.state.temperature == 25.0
        assert twin.state.health == 1.0
        assert not twin.is_running

    def test_set_target_speed(self):
        """Test setting target speed."""
        config = PhoenixConfig()
        twin = DigitalTwin(config)

        twin.set_target_speed(1500)
        assert twin.target_speed == 1500

        # Test speed limit
        twin.set_target_speed(10000)
        assert twin.target_speed <= config.simulation.motor_speed * 1.5

    def test_set_load_torque(self):
        """Test setting load torque."""
        config = PhoenixConfig()
        twin = DigitalTwin(config)

        twin.set_load_torque(15)
        assert twin.load_torque == 15

        # Test negative torque
        twin.set_load_torque(-5)
        assert twin.load_torque == 0.0

    @pytest.mark.asyncio
    async def test_simulation_start_stop(self):
        """Test simulation start and stop."""
        config = PhoenixConfig()
        twin = DigitalTwin(config)

        # Start simulation briefly
        task = asyncio.create_task(twin.start(duration=0.1))
        await asyncio.sleep(0.05)

        assert twin.is_running
        assert twin.simulation_time > 0

        # Wait for completion
        await task

        assert not twin.is_running
        assert twin.simulation_time >= 0.1

    def test_get_status(self):
        """Test status retrieval."""
        config = PhoenixConfig()
        twin = DigitalTwin(config)

        status = twin.get_status()

        assert "state" in status
        assert "motor" in status
        assert "control" in status
        assert "targets" in status
        assert "health" in status

        assert status["state"] == "stopped"
        assert status["motor"]["speed"] == 0.0
        assert status["targets"]["speed"] == twin.target_speed

    def test_health_calculation(self):
        """Test health calculation."""
        config = PhoenixConfig()
        twin = DigitalTwin(config)

        # Normal conditions
        thermal_health = twin._calculate_thermal_health()
        mechanical_health = twin._calculate_mechanical_health()
        overall_health = twin._calculate_overall_health()

        assert 0 <= thermal_health <= 1
        assert 0 <= mechanical_health <= 1
        assert 0 <= overall_health <= 1

    @pytest.mark.asyncio
    async def test_callbacks(self):
        """Test callback functionality."""
        config = PhoenixConfig()
        twin = DigitalTwin(config)

        state_updates = []
        anomaly_updates = []

        def state_callback(state):
            state_updates.append(state)

        def anomaly_callback(anomalies):
            anomaly_updates.append(anomalies)

        twin.add_state_callback(state_callback)
        twin.add_anomaly_callback(anomaly_callback)

        # Run briefly
        await twin.start(duration=0.1)

        # Should have received state updates
        assert len(state_updates) > 0


class TestPhoenixConfig:
    """Test configuration functionality."""

    def test_default_config(self):
        """Test default configuration."""
        config = PhoenixConfig()

        assert config.simulation.dt == 0.001
        assert config.simulation.motor_power == 5.0
        assert config.simulation.motor_speed == 1800.0
        assert config.control.safety_limits["max_current"] == 50.0
        assert config.interface.api_port == 8000

    def test_config_validation(self):
        """Test configuration validation."""
        config = PhoenixConfig()

        # Test valid values
        config.simulation.dt = 0.01
        config.simulation.motor_power = 10.0

        assert config.simulation.dt == 0.01
        assert config.simulation.motor_power == 10.0
