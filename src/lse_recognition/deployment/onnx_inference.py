"""
onnx_inference.py — Motor de Inferencia Optimizado con ONNX Runtime
==================================================================

Permite ejecutar inferencia de ultra-baja latencia sobre secuencias de LSE
sin depender del framework pesado de PyTorch en entornos de producción.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ONNXSignPredictor:
    """
    Motor de inferencia para modelos exportados a ONNX.

    Args:
        model_path: Ruta al archivo .onnx.
        class_names: Lista ordenada de nombres de clases (índice -> palabra).
        temperature: Temperatura para calibración de probabilidades (T=1.0 sin cambio).
        num_threads: Hilos de CPU para ejecución paralela.
    """

    def __init__(
        self,
        model_path: str | Path,
        class_names: Optional[List[str]] = None,
        temperature: float = 1.0,
        num_threads: int = 4,
    ):
        self.model_path = Path(model_path)
        self.class_names = class_names or []
        self.temperature = max(0.01, float(temperature))

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo ONNX no encontrado en: {self.model_path}")

        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape  # ej: [batch_size, 40, 126]
            logger.info(f"✅ Sesión ONNX Runtime inicializada ({self.model_path.name})")
        except ImportError:
            raise RuntimeError("onnxruntime no está instalado. Ejecute `pip install onnxruntime`.")

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Aplica Softmax con escalado de temperatura."""
        scaled_logits = logits / self.temperature
        exp_vals = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def predict_sequence(
        self, sequence: np.ndarray, top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Realiza la predicción sobre una secuencia temporal (T, D) o batch (B, T, D).

        Args:
            sequence: Array numpy con forma (T, D) o (B, T, D), dtype float32.
            top_k: Número de clases candidatas principales a retornar.

        Returns:
            Dict con:
                - 'predicted_class': Nombre de la clase ganadora.
                - 'predicted_idx': Índice de la clase ganadora.
                - 'confidence': Confianza calibrada [0.0, 1.0].
                - 'top_k': Lista de tuplas (clase, probabilidad).
                - 'latency_ms': Tiempo de inferencia en milisegundos.
                - 'all_probabilities': Array de probabilidades por clase.
        """
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)  # (1, T, D)

        t0 = time.perf_counter()
        raw_outputs = self.session.run(
            [self.output_name], {self.input_name: arr}
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = self._softmax(raw_outputs[0])
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        pred_class = (
            self.class_names[pred_idx]
            if pred_idx < len(self.class_names)
            else f"class_{pred_idx}"
        )

        # Top-K
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_k_list = [
            {
                "class": (
                    self.class_names[idx]
                    if idx < len(self.class_names)
                    else f"class_{idx}"
                ),
                "index": int(idx),
                "probability": float(probs[idx]),
            }
            for idx in top_k_indices
        ]

        return {
            "predicted_class": pred_class,
            "predicted_idx": pred_idx,
            "confidence": confidence,
            "top_k": top_k_list,
            "latency_ms": round(latency_ms, 2),
            "all_probabilities": probs.tolist(),
        }
