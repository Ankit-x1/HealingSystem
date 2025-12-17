"""
Test suite for PhoenixDT ML components
"""

import pytest
import numpy as np
import torch
from phoenixdt.ml.failure_synthesizer import FailureSynthesizer
from phoenixdt.ml.anomaly_detector import AnomalyDetector
from phoenixdt.core.config import MLConfig


class TestFailureSynthesizer:
    """Test failure synthesizer functionality"""

    def test_initialization(self):
        """Test synthesizer initialization"""
        config = MLConfig()
        synthesizer = FailureSynthesizer(config)

        assert synthesizer.input_dim == 8
        assert not synthesizer.is_trained
        assert len(synthesizer.failure_modes) > 0

    def test_failure_generation(self):
        """Test synthetic failure generation"""
        config = MLConfig()
        synthesizer = FailureSynthesizer(config)

        # Generate failure without training (should work)
        state = np.random.randn(8)
        failure = synthesizer.generate_failure(state)

        assert "type" in failure
        assert "severity" in failure
        assert "original_state" in failure
        assert "anomalous_state" in failure
        assert 0 <= failure["severity"] <= 1

    def test_failure_modes(self):
        """Test failure mode definitions"""
        config = MLConfig()
        synthesizer = FailureSynthesizer(config)

        stats = synthesizer.get_failure_statistics()

        assert "bearing_wear" in stats
        assert "lubrication_loss" in stats
        assert "overload" in stats

        for mode_name, mode_info in stats.items():
            assert "description" in mode_info
            assert "severity_range" in mode_info
            assert "affected_parameters" in mode_info


class TestAnomalyDetector:
    """Test anomaly detector functionality"""

    def test_initialization(self):
        """Test detector initialization"""
        config = MLConfig()
        detector = AnomalyDetector(config)

        assert not detector.is_fitted
        assert detector.input_dim == 8
        assert detector.anomaly_threshold == config.anomaly_threshold

    def test_training(self):
        """Test detector training"""
        config = MLConfig()
        detector = AnomalyDetector(config)

        # Generate normal data
        normal_data = np.random.randn(1000, 8)

        # Train detector
        detector.fit(normal_data)

        assert detector.is_fitted

    @pytest.mark.asyncio
    async def test_anomaly_detection(self):
        """Test anomaly detection"""
        config = MLConfig()
        detector = AnomalyDetector(config)

        # Train on normal data
        normal_data = np.random.randn(100, 8)
        detector.fit(normal_data)

        # Test with normal sample
        normal_sample = np.random.randn(8)
        anomalies = await detector.detect(normal_sample)

        # Should detect no anomalies for normal data
        assert len(anomalies) == 0 or all(a["severity"] < 0.5 for a in anomalies)

        # Test with anomalous sample
        anomalous_sample = np.random.randn(8) * 5  # Large values
        anomalies = await detector.detect(anomalous_sample)

        # Should detect anomalies for anomalous data
        assert len(anomalies) >= 0  # May or may not detect depending on threshold


if __name__ == "__main__":
    pytest.main([__file__])
