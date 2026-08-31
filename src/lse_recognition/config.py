"""
config.py — Carga y acceso a la configuración del proyecto
===========================================================

Carga el archivo YAML de configuración y expone un diccionario plano
compatible con el código existente del proyecto.

Uso:
    from lse_recognition.config import load_config

    cfg = load_config()                        # usa configs/default.yaml
    cfg = load_config("configs/custom.yaml")   # usa config personalizado
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


# Ruta al config por defecto (relativa al raíz del proyecto)
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo YAML.

    Args:
        path: Ruta al archivo YAML. Si es None, usa configs/default.yaml.

    Returns:
        Diccionario plano con todas las claves de configuración,
        compatible con MODEL_CONFIG e INFERENCE_CONFIG del notebook original.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {config_path}\n"
            f"Crea un archivo YAML o usa la ruta correcta."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Aplanar la configuración en un diccionario compatible con el código original
    cfg: Dict[str, Any] = {}

    # Sección model
    model = raw.get("model", {})
    cfg["seq_length"] = model.get("seq_length", 60)
    cfg["input_features"] = model.get("input_features", 126)
    cfg["num_classes"] = model.get("num_classes", 10)
    cfg["projection_dim"] = model.get("projection_dim", 128)
    cfg["tcn_channels"] = model.get("tcn_channels", [128, 128, 128, 256])
    cfg["kernel_size"] = model.get("kernel_size", 3)
    cfg["tcn_dropout"] = model.get("tcn_dropout", 0.3)
    cfg["fc_hidden_dim"] = model.get("fc_hidden_dim", 128)
    cfg["fc_dropout"] = model.get("fc_dropout", 0.4)

    # Sección training
    training = raw.get("training", {})
    cfg["batch_size"] = training.get("batch_size", 32)
    cfg["learning_rate"] = training.get("learning_rate", 1e-3)
    cfg["weight_decay"] = training.get("weight_decay", 1e-4)
    cfg["epochs"] = training.get("epochs", 120)
    cfg["early_stopping_patience"] = training.get("early_stopping_patience", 15)
    cfg["lr_scheduler_patience"] = training.get("lr_scheduler_patience", 6)
    cfg["lr_scheduler_factor"] = training.get("lr_scheduler_factor", 0.5)
    cfg["seed"] = training.get("seed", 42)
    cfg["use_hands_only"] = training.get("use_hands_only", True)
    cfg["normalize"] = training.get("normalize", True)

    # Sección data
    data = raw.get("data", {})
    cfg["train_csv"] = data.get("train_csv", "data/metadata/train_split_augmented.csv")
    cfg["val_csv"] = data.get("val_csv", "data/metadata/val_split.csv")
    cfg["test_csv"] = data.get("test_csv", "data/metadata/test_split.csv")
    cfg["model_dir"] = data.get("model_dir", "models")

    # Sección inference
    inference = raw.get("inference", {})
    cfg["camera_index"] = inference.get("camera_index", 0)
    cfg["frame_width"] = inference.get("frame_width", 640)
    cfg["frame_height"] = inference.get("frame_height", 480)
    cfg["buffer_size"] = inference.get("buffer_size", 90)
    cfg["min_frames_to_predict"] = inference.get("min_frames_to_predict", 30)
    cfg["confidence_threshold_high"] = inference.get("confidence_threshold_high", 0.75)
    cfg["confidence_threshold_low"] = inference.get("confidence_threshold_low", 0.5)
    cfg["stabilization_votes"] = inference.get("stabilization_votes", 3)
    cfg["cooldown_duration"] = inference.get("cooldown_duration", 3.0)
    cfg["prediction_interval"] = inference.get("prediction_interval", 5)
    cfg["enable_tts"] = inference.get("enable_tts", True)
    cfg["tts_language"] = inference.get("tts_language", "es")
    cfg["tts_rate"] = inference.get("tts_rate", 150)
    cfg["show_fps"] = inference.get("show_fps", True)
    cfg["show_buffer_status"] = inference.get("show_buffer_status", True)
    cfg["mediapipe_min_detection_confidence"] = inference.get(
        "mediapipe_min_detection_confidence", 0.7
    )
    cfg["mediapipe_min_tracking_confidence"] = inference.get(
        "mediapipe_min_tracking_confidence", 0.5
    )

    return cfg


def get_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae solo las claves relevantes para crear el modelo."""
    model_keys = [
        "seq_length", "input_features", "num_classes", "projection_dim",
        "tcn_channels", "kernel_size", "tcn_dropout", "fc_hidden_dim", "fc_dropout",
        "batch_size", "learning_rate", "weight_decay", "epochs",
        "early_stopping_patience", "lr_scheduler_patience", "lr_scheduler_factor",
        "confidence_threshold_high", "confidence_threshold_low",
        "cooldown_duration", "stabilization_votes", "min_frames_to_predict",
    ]
    return {k: cfg[k] for k in model_keys if k in cfg}
