"""
lse_recognition — Sistema de Reconocimiento de Lengua de Signos Española
=========================================================================

Paquete principal. Expone las importaciones más comunes para facilitar el uso.

Estructura:
    lse_recognition/
    ├── config.py           → Carga de configuración YAML
    ├── data/               → Dataset, normalización y utilidades de datos
    ├── models/             → Arquitecturas TCN y LSTM
    ├── training/           → Pipeline de entrenamiento
    ├── evaluation/         → Métricas y visualización de resultados
    └── inference/          → Inferencia en tiempo real y TTS
"""

import sys

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError with emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

__version__ = "1.0.0"
__author__ = "Universidad Politécnica de Madrid"

# Re-exportaciones convenientes
from lse_recognition.config import load_config
from lse_recognition.data.dataset import SignLanguageDataset, LandmarksNormalizer, create_dataloaders
from lse_recognition.models.tcn import TCNSignClassifier
from lse_recognition.models.lstm import LSTMSignClassifier
from lse_recognition.training.trainer import Trainer, run_training
from lse_recognition.evaluation.metrics import evaluate_model

__all__ = [
    "load_config",
    "SignLanguageDataset",
    "LandmarksNormalizer",
    "create_dataloaders",
    "TCNSignClassifier",
    "LSTMSignClassifier",
    "Trainer",
    "run_training",
    "evaluate_model",
]
