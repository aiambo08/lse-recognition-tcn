"""
export.py — Exportación Optimizada a ONNX Runtime y TorchScript
==============================================================

Permite exportar modelos entrenados a formatos de producción de baja latencia:
    - ONNX con ejes dinámicos de batch y validación de paridad numérica.
    - TorchScript (JIT Traced / Scripted).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    seq_length: int = 60,
    input_features: int = 126,
    batch_size: int = 1,
    dynamic_batch: bool = True,
    opset_version: int = 18,
    verify_parity: bool = True,
) -> Path:
    """
    Exporta un modelo PyTorch a formato ONNX (.onnx).

    Args:
        model: Modelo PyTorch entrenado.
        output_path: Ruta de salida del archivo .onnx.
        seq_length: Longitud temporal de la secuencia (T=60).
        input_features: Dimensión de entrada por frame (D=126).
        batch_size: Tamaño de batch de ejemplo para el trazado.
        dynamic_batch: Si es True, configura el eje 0 (batch) y eje 1 (tiempo) como dinámicos.
        opset_version: Versión de ONNX opset (default 18).
        verify_parity: Si es True, verifica con onnxruntime que las salidas coincidan.

    Returns:
        Path del archivo ONNX generado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_eval = model.cpu().eval()
    dummy_input = torch.randn(batch_size, seq_length, input_features, dtype=torch.float32)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        }

    logger.info(f"Exportando modelo a ONNX en: {output_path}")
    torch.onnx.export(
        model_eval,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("✅ Validación de grafo ONNX superada con éxito.")
    except ImportError:
        logger.warning("Paquete 'onnx' no disponible para verificación estática.")

    if verify_parity:
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outs = session.run(None, ort_inputs)[0]

            with torch.no_grad():
                torch_out = model_eval(dummy_input).numpy()

            max_diff = np.max(np.abs(torch_out - ort_outs))
            if not np.allclose(torch_out, ort_outs, atol=1e-4):
                logger.warning(f"Diferencia numérica detectable: max_diff={max_diff:.6f}")
            else:
                logger.info(f"✅ Paridad PyTorch vs ONNX perfecta (max_diff={max_diff:.6e})")
        except ImportError:
            logger.warning("onnxruntime no disponible para comprobación de paridad.")

    return output_path


def export_to_torchscript(
    model: nn.Module,
    output_path: str | Path,
    seq_length: int = 60,
    input_features: int = 126,
    batch_size: int = 1,
) -> Path:
    """
    Exporta un modelo a TorchScript (.pt) mediante torch.jit.trace.

    Args:
        model: Modelo PyTorch entrenado.
        output_path: Ruta de salida del archivo TorchScript.
        seq_length: Longitud temporal de la secuencia.
        input_features: Dimensión de entrada por frame.
        batch_size: Tamaño de batch.

    Returns:
        Path del archivo TorchScript generado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_eval = model.cpu().eval()
    dummy_input = torch.randn(batch_size, seq_length, input_features, dtype=torch.float32)

    traced_model = torch.jit.trace(model_eval, dummy_input)
    traced_model.save(str(output_path))
    logger.info(f"✅ Modelo exportado a TorchScript en: {output_path}")

    return output_path
