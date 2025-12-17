"""
Reinforcement Learning Controller using SAC Algorithm

Implements Soft Actor-Critic for continuous control of industrial systems
with safety constraints and adaptive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
from collections import deque, namedtuple
import random
from loguru import logger

from ..core.config import ControlConfig


@dataclass
class RLExperience:
    """Experience tuple for RL training"""

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ActorNetwork(nn.Module):
    """Actor network for SAC policy"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            )
            prev_dim = hidden_dim

        # Mean and log std layers
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        layers.append(nn.ReLU())

        self.feature_extractor = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        # Action scaling
        self.action_scale = torch.tensor(1.0)
        self.action_bias = torch.tensor(0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log std"""
        features = self.feature_extractor(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)

        # Clamp log std for stability
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action using reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforce action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action


class CriticNetwork(nn.Module):
    """Critic network for SAC Q-function"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        # Q1 network
        q1_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            q1_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            )
            prev_dim = hidden_dim

        q1_layers.append(nn.Linear(prev_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2 network
        q2_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            q2_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            )
            prev_dim = hidden_dim

        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning Q1 and Q2 values"""
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


class MotorControlEnv(gym.Env):
    """Custom environment for motor control RL training"""

    def __init__(self, motor_simulator):
        super().__init__()

        self.motor_sim = motor_simulator
        self.max_voltage = 500.0  # V

        # State space: [speed, torque, current, voltage, wear, temp, vibration, lubrication]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 20, 0, 0]),
            high=np.array([3000, 100, 100, 500, 1, 150, 20, 1]),
            dtype=np.float32,
        )

        # Action space: 3-phase voltages
        self.action_space = spaces.Box(
            low=-self.max_voltage, high=self.max_voltage, shape=(3,), dtype=np.float32
        )

        # Target values
        self.target_speed = 1800.0  # RPM
        self.max_load = 50.0  # Nm

        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 1000

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        self.motor_sim.reset()
        self.episode_step = 0

        state = self.motor_sim.get_state_vector()
        return state.astype(np.float32), {}

    def step(self, action):
        """Execute one step"""
        # Apply action
        voltage_command = np.clip(action, -self.max_voltage, self.max_voltage)

        # Random load disturbance
        load_torque = np.random.uniform(5, self.max_load)

        # Simulate
        state_dict = self.motor_sim.step(
            dt=0.01,  # 10ms timestep
            load_torque=load_torque,
            voltage_command=voltage_command,
        )

        # Get new state
        state = self.motor_sim.get_state_vector()

        # Calculate reward
        reward = self._calculate_reward(state_dict, voltage_command)

        # Check termination
        done = self._check_termination(state_dict)
        truncated = self.episode_step >= self.max_episode_steps

        self.episode_step += 1

        return (
            state.astype(np.float32),
            reward,
            done,
            truncated,
            {"state_dict": state_dict, "load_torque": load_torque},
        )

    def _calculate_reward(self, state_dict: Dict, action: np.ndarray) -> float:
        """Multi-objective reward function"""
        # Speed tracking reward
        speed_error = abs(state_dict["speed_rpm"] - self.target_speed)
        speed_reward = max(0, 1.0 - speed_error / self.target_speed)

        # Efficiency reward
        efficiency_reward = state_dict.get("efficiency", 0.9)

        # Health reward (penalize wear)
        health_reward = 1.0 - state_dict.get("bearing_wear", 0.0)

        # Power consumption penalty
        power_penalty = -state_dict.get("power_w", 0) / 10000.0

        # Action smoothness penalty
        action_penalty = -np.linalg.norm(action) / self.max_voltage

        # Combined reward
        reward = (
            0.3 * speed_reward
            + 0.2 * efficiency_reward
            + 0.2 * health_reward
            + 0.1 * power_penalty
            + 0.1 * action_penalty
        )

        return reward

    def _check_termination(self, state_dict: Dict) -> bool:
        """Check if episode should terminate"""
        # Safety constraints
        if state_dict.get("bearing_temp", 25) > 120:
            return True

        if state_dict.get("vibration_mm_s", 0) > 15:
            return True

        if state_dict.get("bearing_wear", 0) > 0.9:
            return True

        return False


class RLController:
    """SAC-based RL controller for motor control"""

    def __init__(self, config: ControlConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network dimensions
        self.state_dim = 8
        self.action_dim = 3
        self.hidden_dims = [256, 256]

        # Initialize networks
        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(
            self.device
        )
        self.critic = CriticNetwork(
            self.state_dim, self.action_dim, self.hidden_dims
        ).to(self.device)
        self.critic_target = CriticNetwork(
            self.state_dim, self.action_dim, self.hidden_dims
        ).to(self.device)

        # Copy weights to target network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.rl_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.rl_learning_rate
        )

        # SAC hyperparameters
        self.alpha = 0.2  # Entropy coefficient
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Soft update parameter

        # Experience buffer
        self.buffer_size = 100000
        self.batch_size = 256
        self.experience_buffer = deque(maxlen=self.buffer_size)

        # Training state
        self.is_training = False
        self.training_step = 0

        # Action bounds
        self.action_low = -500.0
        self.action_high = 500.0

        logger.info("RL controller initialized")

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                _, _, mean_action = self.actor.sample(state_tensor)
                action = mean_action.cpu().numpy()[0]
            else:
                action, _, _ = self.actor.sample(state_tensor)
                action = action.cpu().numpy()[0]

        # Scale to action bounds
        action = np.clip(action, self.action_low, self.action_high)

        return action

    async def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        """Update policy with new experience"""
        # Store experience
        experience = RLExperience(state, action, reward, next_state, done)
        self.experience_buffer.append(experience)

        # Train if enough samples
        if len(self.experience_buffer) >= self.batch_size and self.is_training:
            await self._train_step()

    async def _train_step(self):
        """Perform one training step"""
        # Sample batch
        batch = random.sample(self.experience_buffer, self.batch_size)

        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.FloatTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)

        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)

        # Update actor
        actor_loss = self._update_actor(states)

        # Update alpha (entropy coefficient)
        alpha_loss = self._update_alpha(states)

        # Soft update target network
        self._soft_update_target()

        self.training_step += 1

        if self.training_step % 100 == 0:
            logger.info(
                f"Training step {self.training_step}: "
                f"Critic={critic_loss:.4f}, Actor={actor_loss:.4f}"
            )

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Update critic networks"""
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            # Compute target Q-values
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs

            # TD target
            target_q = rewards + self.gamma * (1 - dones) * min_q_target

        # Current Q-values
        q1, q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, states: torch.Tensor) -> float:
        """Update actor network"""
        # Sample actions
        actions, log_probs, _ = self.actor.sample(states)

        # Compute Q-values
        q1, q2 = self.critic(states, actions)
        min_q = torch.min(q1, q2)

        # Actor loss (maximize Q + entropy)
        actor_loss = (self.alpha * log_probs - min_q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, states: torch.Tensor) -> float:
        """Update entropy coefficient"""
        with torch.no_grad():
            _, log_probs, _ = self.actor.sample(states)

        # Alpha loss (target entropy)
        target_entropy = -self.action_dim  # Heuristic
        alpha_loss = -(self.alpha * (log_probs + target_entropy).detach()).mean()

        # Update alpha (if learnable)
        # For simplicity, keeping alpha fixed

        return alpha_loss.item()

    def _soft_update_target(self):
        """Soft update of target network"""
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def start_training(self):
        """Start training mode"""
        self.is_training = True
        logger.info("RL controller training started")

    def stop_training(self):
        """Stop training mode"""
        self.is_training = False
        logger.info("RL controller training stopped")

    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "training_step": self.training_step,
                "config": self.config,
            },
            filepath,
        )
        logger.info(f"RL model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]

        # Update target network
        self.critic_target.load_state_dict(self.critic.state_dict())

        logger.info(f"RL model loaded from {filepath}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "training_step": self.training_step,
            "buffer_size": len(self.experience_buffer),
            "is_training": self.is_training,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "tau": self.tau,
        }


class PIDController:
    """Traditional PID controller for comparison"""

    def __init__(self, gains: Dict[str, float]):
        self.kp = gains.get("kp", 1.0)
        self.ki = gains.get("ki", 0.1)
        self.kd = gains.get("kd", 0.01)

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, setpoint: float, current_value: float) -> float:
        """Compute PID control output"""
        import time

        current_time = time.time()
        if self.prev_time is None:
            self.prev_time = current_time

        dt = current_time - self.prev_time

        # Error terms
        error = setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Update state
        self.prev_error = error
        self.prev_time = current_time

        return output

    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
