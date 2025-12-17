"""
Test suite for PhoenixDT motor simulator
"""

import pytest
import numpy as np
from phoenixdt.simulation.motor_simulator import (
    MotorSimulator,
    MotorParameters,
    BearingState,
)


class TestMotorSimulator:
    """Test motor simulator functionality"""

    def test_initialization(self):
        """Test simulator initialization"""
        sim = MotorSimulator()
        assert sim.time == 0.0
        assert sim.speed == 0.0
        assert sim.torque == 0.0
        assert len(sim.history) == 0

    def test_step_simulation(self):
        """Test single simulation step"""
        sim = MotorSimulator()
        state = sim.step(dt=0.01, load_torque=10.0)

        assert "time" in state
        assert "speed_rpm" in state
        assert "torque_nm" in state
        assert "current_a" in state
        assert "efficiency" in state
        assert state["time"] == 0.01

    def test_bearing_degradation(self):
        """Test bearing degradation over time"""
        sim = MotorSimulator()
        initial_wear = sim.bearing_state.wear_level

        # Run simulation for some time
        for _ in range(100):
            sim.step(dt=0.1, load_torque=20.0)

        # Wear should increase
        assert sim.bearing_state.wear_level > initial_wear
        assert sim.bearing_state.wear_level <= 1.0

    def test_fault_injection(self):
        """Test fault injection functionality"""
        sim = MotorSimulator()

        # Inject bearing wear fault
        sim.inject_fault("bearing_wear", 0.5)
        assert sim.bearing_state.wear_level == 0.5

        # Inject lubrication loss
        sim.inject_fault("lubrication_loss", 0.8)
        assert sim.bearing_state.lubrication_quality == 0.2

    def test_state_vector(self):
        """Test state vector extraction"""
        sim = MotorSimulator()
        sim.step(dt=0.01, load_torque=10.0)

        state_vector = sim.get_state_vector()
        assert len(state_vector) == 8  # 8 parameters
        assert all(isinstance(x, (int, float)) for x in state_vector)

    def test_reset_functionality(self):
        """Test simulator reset"""
        sim = MotorSimulator()

        # Run simulation
        for _ in range(10):
            sim.step(dt=0.01, load_torque=10.0)

        # Reset
        sim.reset()

        assert sim.time == 0.0
        assert sim.speed == 0.0
        assert sim.torque == 0.0
        assert len(sim.history) == 0
        assert sim.bearing_state.wear_level == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
