"""
Test suite for PhoenixDT control systems
"""

import pytest
import numpy as np
import torch
from phoenixdt.control.rl_controller import RLController, PIDController
from phoenixdt.core.config import ControlConfig


class TestPIDController:
    """Test PID controller functionality"""

    def test_initialization(self):
        """Test PID controller initialization"""
        gains = {"kp": 1.0, "ki": 0.1, "kd": 0.01}
        controller = PIDController(gains)

        assert controller.kp == 1.0
        assert controller.ki == 0.1
        assert controller.kd == 0.01
        assert controller.integral == 0.0
        assert controller.prev_error == 0.0

    def test_control_output(self):
        """Test PID control output"""
        gains = {"kp": 1.0, "ki": 0.1, "kd": 0.01}
        controller = PIDController(gains)

        # Test control output
        output = controller.compute(setpoint=100.0, current_value=90.0)

        # Should be positive (need to increase value)
        assert output > 0

        # Test with higher current value
        output2 = controller.compute(setpoint=100.0, current_value=110.0)

        # Should be negative (need to decrease value)
        assert output2 < 0

    def test_reset(self):
        """Test controller reset"""
        gains = {"kp": 1.0, "ki": 0.1, "kd": 0.01}
        controller = PIDController(gains)

        # Generate some control action
        controller.compute(setpoint=100.0, current_value=90.0)

        # Reset
        controller.reset()

        assert controller.integral == 0.0
        assert controller.prev_error == 0.0
        assert controller.prev_time is None


class TestRLController:
    """Test RL controller functionality"""

    def test_initialization(self):
        """Test RL controller initialization"""
        config = ControlConfig()
        controller = RLController(config)

        assert controller.state_dim == 8
        assert controller.action_dim == 3
        assert not controller.is_training
        assert len(controller.experience_buffer) == 0

    def test_action_generation(self):
        """Test action generation"""
        config = ControlConfig()
        controller = RLController(config)

        # Generate action from state
        state = np.random.randn(8)
        action = controller.get_action(state)

        assert len(action) == 3
        assert all(isinstance(a, (int, float)) for a in action)
        assert all(-500 <= a <= 500 for a in action)  # Action bounds

    def test_deterministic_action(self):
        """Test deterministic action generation"""
        config = ControlConfig()
        controller = RLController(config)

        # Generate deterministic action
        state = np.random.randn(8)
        action1 = controller.get_action(state, deterministic=True)
        action2 = controller.get_action(state, deterministic=True)

        # Should be the same for deterministic mode
        np.testing.assert_array_almost_equal(action1, action2)

    @pytest.mark.asyncio
    async def test_experience_update(self):
        """Test experience buffer update"""
        config = ControlConfig()
        controller = RLController(config)

        # Add experience
        state = np.random.randn(8)
        action = np.random.randn(3)
        reward = 1.0
        next_state = np.random.randn(8)

        await controller.update(state, action, reward, next_state)

        assert len(controller.experience_buffer) == 1

    def test_training_control(self):
        """Test training mode control"""
        config = ControlConfig()
        controller = RLController(config)

        # Start training
        controller.start_training()
        assert controller.is_training

        # Stop training
        controller.stop_training()
        assert not controller.is_training

    def test_model_save_load(self):
        """Test model saving and loading"""
        config = ControlConfig()
        controller = RLController(config)

        # Save model
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            controller.save_model(f.name)

            # Load model
            new_controller = RLController(config)
            new_controller.load_model(f.name)

            # Should have same training step
            assert new_controller.training_step == controller.training_step


if __name__ == "__main__":
    pytest.main([__file__])
