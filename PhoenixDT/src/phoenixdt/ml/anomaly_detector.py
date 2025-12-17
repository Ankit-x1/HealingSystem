"""
Real-time Anomaly Detection with Uncertainty Quantification

Implements multiple anomaly detection algorithms with uncertainty estimation
to provide reliable failure detection in industrial systems.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
from loguru import logger

from ..core.config import MLConfig


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""

    is_anomaly: bool
    anomaly_score: float
    uncertainty: float
    confidence: float
    detected_by: List[str]
    details: Dict[str, Any]


class AutoencoderDetector(nn.Module):
    """Autoencoder for anomaly detection"""

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            )
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class BayesianAnomalyDetector:
    """Bayesian approach for anomaly detection with uncertainty"""

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.prior_mean = np.zeros(input_dim)
        self.prior_cov = np.eye(input_dim)
        self.data_count = 0
        self.is_trained = False

    def update_prior(self, data: np.ndarray):
        """Update prior distribution with new data"""
        if len(data) == 0:
            return

        # Update sufficient statistics
        new_mean = np.mean(data, axis=0)
        new_cov = np.cov(data.T)

        if self.data_count == 0:
            self.prior_mean = new_mean
            self.prior_cov = new_cov
        else:
            # Bayesian update
            alpha = self.data_count / (self.data_count + len(data))
            self.prior_mean = alpha * self.prior_mean + (1 - alpha) * new_mean
            self.prior_cov = alpha * self.prior_cov + (1 - alpha) * new_cov

        self.data_count += len(data)
        self.is_trained = True

    def detect(self, sample: np.ndarray) -> Tuple[float, float]:
        """Detect anomaly with uncertainty using Mahalanobis distance"""
        if not self.is_trained:
            return 0.5, 1.0  # High uncertainty when not trained

        # Mahalanobis distance
        diff = sample - self.prior_mean
        try:
            inv_cov = np.linalg.inv(self.prior_cov + 1e-6 * np.eye(self.input_dim))
            mahal_dist = np.sqrt(diff @ inv_cov @ diff)
        except:
            mahal_dist = np.linalg.norm(diff)

        # Convert to anomaly score (0-1)
        anomaly_score = 1.0 / (1.0 + np.exp(-0.5 * (mahal_dist - 3.0)))

        # Uncertainty based on data count and covariance condition
        uncertainty = 1.0 / (1.0 + self.data_count / 100.0)
        cond_number = np.linalg.cond(self.prior_cov)
        uncertainty *= min(1.0, cond_number / 1000.0)

        return anomaly_score, uncertainty


class AnomalyDetector:
    """Multi-algorithm anomaly detector with uncertainty quantification"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize detectors
        self.input_dim = 8  # State vector dimension
        self.isolation_forest = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=100
        )

        self.autoencoder = AutoencoderDetector(
            input_dim=self.input_dim, hidden_dims=[64, 32, 16]
        ).to(self.device)

        self.bayesian_detector = BayesianAnomalyDetector(self.input_dim)

        # Data preprocessing
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Thresholds
        self.anomaly_threshold = config.anomaly_threshold
        self.uncertainty_threshold = 0.3

        # History for ensemble
        self.detection_history: List[Dict] = []

        logger.info("Anomaly detector initialized")

    def fit(self, normal_data: np.ndarray):
        """Train all detectors on normal data"""
        logger.info(f"Training anomaly detectors on {len(normal_data)} samples")

        # Preprocess data
        self.scaler.fit(normal_data)
        scaled_data = self.scaler.transform(normal_data)

        # Train isolation forest
        self.isolation_forest.fit(scaled_data)

        # Train autoencoder
        self._train_autoencoder(scaled_data)

        # Update Bayesian prior
        self.bayesian_detector.update_prior(scaled_data)

        self.is_fitted = True
        logger.info("Anomaly detectors training completed")

    def _train_autoencoder(self, data: np.ndarray, epochs: int = 50):
        """Train autoencoder for reconstruction-based detection"""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(data).to(self.device)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]

                # Forward pass
                recon = self.autoencoder(x)
                loss = criterion(recon, x)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(f"Autoencoder epoch {epoch}: Loss = {total_loss:.4f}")

    async def detect(self, sample: np.ndarray) -> List[Dict[str, Any]]:
        """Detect anomalies using ensemble of methods"""
        if not self.is_fitted:
            logger.warning("Detectors not fitted, returning empty results")
            return []

        # Preprocess sample
        scaled_sample = self.scaler.transform(sample.reshape(1, -1))[0]

        # Detection results from different methods
        results = {}

        # 1. Isolation Forest
        if_score = self.isolation_forest.decision_function([scaled_sample])[0]
        if_anomaly = if_score < 0
        if_confidence = abs(if_score)
        results["isolation_forest"] = AnomalyResult(
            is_anomaly=if_anomaly,
            anomaly_score=1.0 - (if_score + 1) / 2,  # Convert to 0-1
            uncertainty=0.1,  # Low uncertainty for IF
            confidence=if_confidence,
            detected_by=["isolation_forest"],
            details={"if_score": if_score},
        )

        # 2. Autoencoder (reconstruction error)
        with torch.no_grad():
            x = torch.FloatTensor(scaled_sample).unsqueeze(0).to(self.device)
            recon = self.autoencoder(x)
            recon_error = torch.mean((x - recon) ** 2).item()

        # Determine threshold from training data (simplified)
        ae_threshold = np.percentile([0.01, 0.02, 0.03, 0.05, 0.08], 90)
        ae_anomaly = recon_error > ae_threshold
        ae_confidence = min(1.0, recon_error / ae_threshold)

        results["autoencoder"] = AnomalyResult(
            is_anomaly=ae_anomaly,
            anomaly_score=min(1.0, recon_error / 0.1),
            uncertainty=0.15,
            confidence=ae_confidence,
            detected_by=["autoencoder"],
            details={"reconstruction_error": recon_error},
        )

        # 3. Bayesian detection
        bayes_score, bayes_uncertainty = self.bayesian_detector.detect(scaled_sample)
        bayes_anomaly = bayes_score > self.anomaly_threshold

        results["bayesian"] = AnomalyResult(
            is_anomaly=bayes_anomaly,
            anomaly_score=bayes_score,
            uncertainty=bayes_uncertainty,
            confidence=1.0 - bayes_uncertainty,
            detected_by=["bayesian"],
            details={"bayes_score": bayes_score},
        )

        # 4. Statistical detection (z-score)
        z_scores = np.abs(scaled_sample)  # Assuming zero-mean normalized data
        max_z_score = np.max(z_scores)
        stat_anomaly = max_z_score > 3.0
        stat_confidence = min(1.0, max_z_score / 5.0)

        results["statistical"] = AnomalyResult(
            is_anomaly=stat_anomaly,
            anomaly_score=min(1.0, max_z_score / 5.0),
            uncertainty=0.2,
            confidence=stat_confidence,
            detected_by=["statistical"],
            details={"max_z_score": max_z_score, "z_scores": z_scores.tolist()},
        )

        # Ensemble decision
        ensemble_result = self._ensemble_decision(results)

        # Store in history
        detection_record = {
            "timestamp": np.random.randint(0, 1000000),
            "sample": sample.tolist(),
            "results": {
                name: {
                    "is_anomaly": result.is_anomaly,
                    "anomaly_score": result.anomaly_score,
                    "uncertainty": result.uncertainty,
                    "confidence": result.confidence,
                }
                for name, result in results.items()
            },
            "ensemble": ensemble_result.__dict__,
        }
        self.detection_history.append(detection_record)

        # Return anomalies detected
        anomalies = []
        if ensemble_result.is_anomaly:
            anomalies.append(
                {
                    "type": "ensemble_anomaly",
                    "severity": ensemble_result.anomaly_score,
                    "confidence": ensemble_result.confidence,
                    "uncertainty": ensemble_result.uncertainty,
                    "detected_by": ensemble_result.detected_by,
                    "details": ensemble_result.details,
                    "timestamp": detection_record["timestamp"],
                }
            )

        return anomalies

    def _ensemble_decision(self, results: Dict[str, AnomalyResult]) -> AnomalyResult:
        """Combine results from multiple detectors"""
        # Weighted voting based on confidence and uncertainty
        weights = {}
        for name, result in results.items():
            # Higher confidence and lower uncertainty = higher weight
            weights[name] = result.confidence * (1.0 - result.uncertainty)

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(results) for k in results.keys()}

        # Weighted anomaly score
        weighted_score = sum(
            weights[name] * result.anomaly_score for name, result in results.items()
        )

        # Weighted uncertainty
        weighted_uncertainty = sum(
            weights[name] * result.uncertainty for name, result in results.items()
        )

        # Weighted confidence
        weighted_confidence = sum(
            weights[name] * result.confidence for name, result in results.items()
        )

        # Determine if anomaly (conservative approach)
        anomaly_votes = sum(
            weights[name] for name, result in results.items() if result.is_anomaly
        )

        is_anomaly = (
            weighted_score > self.anomaly_threshold
            and weighted_uncertainty < self.uncertainty_threshold
            and anomaly_votes > 0.3
        )

        # Collect all detectors that flagged anomaly
        detected_by = []
        for name, result in results.items():
            if result.is_anomaly:
                detected_by.append(name)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=weighted_score,
            uncertainty=weighted_uncertainty,
            confidence=weighted_confidence,
            detected_by=detected_by,
            details={
                "individual_scores": {
                    name: result.anomaly_score for name, result in results.items()
                },
                "weights": weights,
                "anomaly_votes": anomaly_votes,
            },
        )

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get statistics about detection performance"""
        if not self.detection_history:
            return {}

        # Extract anomaly scores and uncertainties
        ensemble_scores = [
            record["ensemble"]["anomaly_score"] for record in self.detection_history
        ]
        uncertainties = [
            record["ensemble"]["uncertainty"] for record in self.detection_history
        ]
        anomaly_count = sum(
            1 for record in self.detection_history if record["ensemble"]["is_anomaly"]
        )

        # Method-specific statistics
        method_stats = {}
        for method in ["isolation_forest", "autoencoder", "bayesian", "statistical"]:
            if method in self.detection_history[0]["results"]:
                scores = [
                    record["results"][method]["anomaly_score"]
                    for record in self.detection_history
                ]
                method_stats[method] = {
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "detection_rate": sum(
                        1
                        for record in self.detection_history
                        if record["results"][method]["is_anomaly"]
                    )
                    / len(self.detection_history),
                }

        return {
            "total_detections": len(self.detection_history),
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_count / len(self.detection_history),
            "mean_ensemble_score": np.mean(ensemble_scores),
            "mean_uncertainty": np.mean(uncertainties),
            "method_statistics": method_stats,
        }

    def update_threshold(self, new_threshold: float):
        """Update anomaly detection threshold"""
        self.anomaly_threshold = np.clip(new_threshold, 0.0, 1.0)
        logger.info(f"Anomaly threshold updated to {self.anomaly_threshold}")

    def save_model(self, filepath: str):
        """Save trained models"""
        import torch

        torch.save(
            {
                "autoencoder_state_dict": self.autoencoder.state_dict(),
                "isolation_forest": self.isolation_forest,
                "scaler": self.scaler,
                "bayesian_detector": self.bayesian_detector,
                "config": self.config,
            },
            filepath,
        )
        logger.info(f"Anomaly detector models saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained models"""
        import torch

        checkpoint = torch.load(filepath, map_location=self.device)

        self.autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
        self.isolation_forest = checkpoint["isolation_forest"]
        self.scaler = checkpoint["scaler"]
        self.bayesian_detector = checkpoint["bayesian_detector"]

        self.is_fitted = True
        logger.info(f"Anomaly detector models loaded from {filepath}")
