"""
lse_recognition.evaluation — Métricas, Calibración de Confianza y Visualización
=============================================================================
"""

from lse_recognition.evaluation.calibration import (
    ExpectedCalibrationError,
    TemperatureScaler,
)
from lse_recognition.evaluation.metrics import (
    evaluate_model,
    plot_training_history,
)

__all__ = [
    "evaluate_model",
    "plot_training_history",
    "ExpectedCalibrationError",
    "TemperatureScaler",
]
