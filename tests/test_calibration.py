"""
tests/test_calibration.py — Tests unitarios para Calibración y ECE
==================================================================
"""

import pytest
import torch

from lse_recognition.evaluation.calibration import (
    ExpectedCalibrationError,
    TemperatureScaler,
)


class TestExpectedCalibrationError:
    """Tests para cálculo de ECE, MCE y Brier Score."""

    def test_ece_returns_expected_keys(self):
        meter = ExpectedCalibrationError(n_bins=10)
        logits = torch.randn(30, 5)
        labels = torch.randint(0, 5, (30,))

        stats = meter(logits, labels, is_logits=True)
        assert "ece" in stats
        assert "mce" in stats
        assert "brier_score" in stats
        assert "accuracy" in stats
        assert "avg_confidence" in stats
        assert "bin_stats" in stats

    def test_ece_bounds(self):
        meter = ExpectedCalibrationError(n_bins=15)
        logits = torch.randn(50, 4)
        labels = torch.randint(0, 4, (50,))

        stats = meter(logits, labels, is_logits=True)
        assert 0.0 <= stats["ece"] <= 100.0
        assert 0.0 <= stats["mce"] <= 100.0
        assert stats["brier_score"] >= 0.0

    def test_perfect_calibration_synthetic(self):
        meter = ExpectedCalibrationError(n_bins=10)
        # Probabilidades exactas coincidentes con etiquetas
        probs = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        stats = meter(probs, labels, is_logits=False)
        assert stats["accuracy"] == 100.0
        assert stats["ece"] == 0.0


class TestTemperatureScaler:
    """Tests para optimización de temperatura post-hoc."""

    def test_temperature_scaling_optimization(self):
        scaler = TemperatureScaler()
        # Logits sobreconfiados
        val_logits = torch.tensor([
            [10.0, -5.0],
            [8.0, -3.0],
            [-6.0, 9.0],
            [-4.0, 7.0],
        ])
        val_labels = torch.tensor([0, 0, 1, 1])

        initial_temp = float(scaler.temperature.item())
        opt_temp = scaler.fit(val_logits, val_labels, max_iter=20)
        assert opt_temp > 0.0

    def test_calibrated_probs_sum_to_one(self):
        scaler = TemperatureScaler()
        logits = torch.randn(5, 10)
        probs = scaler.calibrate_probs(logits)
        assert probs.shape == (5, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)
