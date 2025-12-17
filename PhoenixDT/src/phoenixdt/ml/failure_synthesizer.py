"""
Generative Failure Synthesis using Variational Autoencoders - Python 3.12.2 compatible

Creates realistic failure scenarios when real failure data is scarce.
Uses VAEs to learn the distribution of normal operations and generate
anomalous patterns that represent different failure modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from loguru import logger

from ..core.config import MLConfig


@dataclass
class FailureMode:
    """Definition of a failure mode"""

    name: str
    description: str
    severity_range: Tuple[float, float]  # (min, max) severity
    affected_parameters: List[str]
    symptom_patterns: Dict[str, float]  # parameter -> expected change


class FailureVAE(nn.Module):
    """Variational Autoencoder for failure synthesis - Python 3.12.2 compatible"""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction"""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class FailureSynthesizer:
    """Generates synthetic failure scenarios using trained VAE"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize VAE
        self.input_dim = 8  # State vector dimension
        self.vae = FailureVAE(
            input_dim=self.input_dim,
            hidden_dims=config.vae_hidden_dims,
            latent_dim=config.vae_latent_dim,
        ).to(self.device)

        # Define failure modes
        self.failure_modes = self._define_failure_modes()

        # Training data storage
        self.normal_data: Optional[np.ndarray] = None
        self.is_trained = False

        logger.info("Failure synthesizer initialized")

    def _define_failure_modes(self) -> Dict[str, FailureMode]:
        """Define known failure modes"""
        return {
            "bearing_wear": FailureMode(
                name="bearing_wear",
                description="Progressive bearing wear due to fatigue",
                severity_range=(0.1, 0.9),
                affected_parameters=["bearing_wear", "vibration", "temperature"],
                symptom_patterns={
                    "bearing_wear": 1.0,
                    "vibration_mm_s": 2.0,
                    "bearing_temp": 1.5,
                },
            ),
            "lubrication_loss": FailureMode(
                name="lubrication_loss",
                description="Loss of lubrication in bearings",
                severity_range=(0.2, 0.8),
                affected_parameters=["temperature", "vibration", "efficiency"],
                symptom_patterns={
                    "bearing_temp": 3.0,
                    "vibration_mm_s": 1.5,
                    "efficiency": -0.2,
                },
            ),
            "overload": FailureMode(
                name="overload",
                description="Excessive mechanical load",
                severity_range=(0.1, 1.0),
                affected_parameters=["current", "temperature", "speed"],
                symptom_patterns={
                    "current_a": 2.0,
                    "bearing_temp": 1.2,
                    "speed_rpm": -0.3,
                },
            ),
            "voltage_unbalance": FailureMode(
                name="voltage_unbalance",
                description="Unbalanced 3-phase voltage supply",
                severity_range=(0.05, 0.3),
                affected_parameters=["current", "vibration", "torque"],
                symptom_patterns={
                    "current_a": 1.5,
                    "vibration_mm_s": 1.0,
                    "torque_nm": -0.2,
                },
            ),
            "cooling_failure": FailureMode(
                name="cooling_failure",
                description="Failure of cooling system",
                severity_range=(0.3, 0.9),
                affected_parameters=["temperature", "efficiency"],
                symptom_patterns={"bearing_temp": 4.0, "efficiency": -0.3},
            ),
        }

    def train(self, normal_data: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train VAE on normal operating data"""
        logger.info(f"Training failure VAE on {len(normal_data)} samples")

        self.normal_data = normal_data

        # Prepare data
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(normal_data).to(self.device)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Optimizer
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        # Training loop
        self.vae.train()
        for epoch in range(epochs):
            total_loss = 0
            recon_loss = 0
            kl_loss = 0

            for batch in dataloader:
                x = batch[0]

                # Forward pass
                recon, mu, logvar = self.vae(x)

                # Compute losses
                recon_loss_batch = F.mse_loss(recon, x, reduction="sum")
                kl_loss_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                total_loss_batch = recon_loss_batch + 0.1 * kl_loss_batch

                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()

                total_loss += total_loss_batch.item()
                recon_loss += recon_loss_batch.item()
                kl_loss += kl_loss_batch.item()

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Total={total_loss:.2f}, "
                    f"Recon={recon_loss:.2f}, KL={kl_loss:.2f}"
                )

        self.is_trained = True
        logger.info("Failure VAE training completed")

    def generate_failure(
        self,
        current_state: np.ndarray,
        failure_type: Optional[str] = None,
        severity: Optional[float] = None,
    ) -> Dict:
        """Generate a synthetic failure scenario"""
        if not self.is_trained:
            logger.warning("VAE not trained, generating random failure")
            return self._generate_random_failure(current_state)

        # Select failure mode
        if failure_type is None:
            failure_type = np.random.choice(list(self.failure_modes.keys()))

        if failure_type not in self.failure_modes:
            raise ValueError(f"Unknown failure type: {failure_type}")

        mode = self.failure_modes[failure_type]

        # Select severity
        if severity is None:
            severity = np.random.uniform(*mode.severity_range)

        # Generate anomalous state
        with torch.no_grad():
            # Encode current state
            x = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
            mu, logvar = self.vae.encode(x)

            # Add controlled noise to create anomaly
            z = mu + severity * torch.randn_like(mu)

            # Decode to get anomalous state
            anomalous_state = self.vae.decode(z).cpu().numpy()[0]

        # Apply failure-specific modifications
        modified_state = self._apply_failure_symptoms(
            current_state, anomalous_state, mode, severity
        )

        return {
            "type": failure_type,
            "severity": severity,
            "description": mode.description,
            "original_state": current_state.tolist(),
            "anomalous_state": modified_state.tolist(),
            "symptoms": mode.symptom_patterns,
            "timestamp": np.random.randint(0, 1000000),
        }

    def _apply_failure_symptoms(
        self,
        original: np.ndarray,
        anomalous: np.ndarray,
        mode: FailureMode,
        severity: float,
    ) -> np.ndarray:
        """Apply failure-specific symptom patterns to generated anomaly"""
        modified = anomalous.copy()

        # Parameter mapping (index -> parameter name)
        param_names = [
            "speed",
            "torque",
            "current",
            "voltage",
            "bearing_wear",
            "temperature",
            "vibration",
            "lubrication",
        ]

        # Apply symptom patterns
        for param, change_factor in mode.symptom_patterns.items():
            if param in param_names:
                idx = param_names.index(param)
                if change_factor > 0:
                    # Increase parameter
                    modified[idx] = original[idx] * (1 + severity * change_factor)
                else:
                    # Decrease parameter
                    modified[idx] = original[idx] * (1 + severity * change_factor)

        # Ensure physical constraints
        modified[0] = max(0, modified[0])  # Speed >= 0
        modified[1] = max(0, modified[1])  # Torque >= 0
        modified[2] = max(0, modified[2])  # Current >= 0
        modified[4] = np.clip(modified[4], 0, 1)  # Wear level [0,1]
        modified[5] = max(20, modified[5])  # Temperature >= 20°C
        modified[6] = max(0, modified[6])  # Vibration >= 0
        modified[7] = np.clip(modified[7], 0, 1)  # Lubrication [0,1]

        return modified

    def _generate_random_failure(self, current_state: np.ndarray) -> Dict:
        """Generate random failure when VAE is not trained"""
        failure_type = np.random.choice(list(self.failure_modes.keys()))
        mode = self.failure_modes[failure_type]
        severity = np.random.uniform(*mode.severity_range)

        # Simple random perturbation
        noise = np.random.normal(0, severity * 0.1, len(current_state))
        anomalous_state = current_state + noise

        return {
            "type": failure_type,
            "severity": severity,
            "description": mode.description,
            "original_state": current_state.tolist(),
            "anomalous_state": anomalous_state.tolist(),
            "symptoms": mode.symptom_patterns,
            "timestamp": np.random.randint(0, 1000000),
        }

    def generate_failure_dataset(
        self, normal_data: np.ndarray, samples_per_mode: int = 100
    ) -> pd.DataFrame:
        """Generate comprehensive failure dataset"""
        if not self.is_trained:
            logger.warning("Training VAE on provided data first")
            self.train(normal_data, epochs=50)

        failure_data = []

        for failure_type, mode in self.failure_modes.items():
            for i in range(samples_per_mode):
                # Select random normal sample
                idx = np.random.randint(0, len(normal_data))
                current_state = normal_data[idx]

                # Generate failure
                failure = self.generate_failure(current_state, failure_type)

                # Create data row
                row = {
                    "failure_type": failure_type,
                    "severity": failure["severity"],
                    "timestamp": failure["timestamp"],
                    **{
                        f"state_{j}": val
                        for j, val in enumerate(failure["anomalous_state"])
                    },
                }
                failure_data.append(row)

        return pd.DataFrame(failure_data)

    def evaluate_failure_realism(self, failure_data: np.ndarray) -> Dict[str, float]:
        """Evaluate how realistic generated failures are"""
        if not self.is_trained:
            return {"realism_score": 0.0}

        with torch.no_grad():
            x = torch.FloatTensor(failure_data).to(self.device)
            recon, mu, logvar = self.vae.encode(x)

            # Calculate reconstruction error
            recon_data = self.vae.decode(mu)
            recon_error = F.mse_loss(recon_data, x, reduction="none")
            avg_recon_error = recon_error.mean().item()

            # Calculate KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            avg_kl_div = kl_div.mean().item()

        # Realism score based on reconstruction error and KL divergence
        realism_score = 1.0 / (1.0 + avg_recon_error + 0.1 * avg_kl_div)

        return {
            "realism_score": realism_score,
            "reconstruction_error": avg_recon_error,
            "kl_divergence": avg_kl_div,
        }

    def save_model(self, filepath: Path):
        """Save trained VAE model"""
        if not self.is_trained:
            logger.warning("Model not trained, nothing to save")
            return

        torch.save(
            {
                "vae_state_dict": self.vae.state_dict(),
                "config": self.config,
                "input_dim": self.input_dim,
                "failure_modes": self.failure_modes,
            },
            filepath,
        )

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path):
        """Load trained VAE model"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.vae.load_state_dict(checkpoint["vae_state_dict"])
        self.input_dim = checkpoint["input_dim"]
        self.failure_modes = checkpoint["failure_modes"]
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")

    def get_failure_statistics(self) -> Dict[str, Dict]:
        """Get statistics about defined failure modes"""
        stats = {}

        for name, mode in self.failure_modes.items():
            stats[name] = {
                "description": mode.description,
                "severity_range": mode.severity_range,
                "affected_parameters": mode.affected_parameters,
                "symptom_count": len(mode.symptom_patterns),
            }

        return stats
