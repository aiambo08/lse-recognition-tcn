"""
lse_recognition.models — Suite de Arquitecturas para Reconocimiento de LSE
========================================================================

Modelos disponibles:
    - TCNSignClassifier: Temporal Convolutional Network causal dilatada estándar.
    - MSTCNSignClassifier: Multi-Scale TCN con convoluciones piramidales y Channel Attention.
    - LSTMSignClassifier: LSTM estándar / Bidireccional.
    - AttentionLSTMSignClassifier: BiLSTM con mecanismo de Temporal Self-Attention.
    - STGCNSignClassifier: Spatial-Temporal Graph Convolutional Network sobre esqueleto de manos.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from lse_recognition.models.attention_lstm import (
    AttentionLSTMSignClassifier,
    TemporalAttention,
)
from lse_recognition.models.lstm import LSTMSignClassifier
from lse_recognition.models.mstcn import (
    ChannelAttention,
    MSTCNSignClassifier,
    MultiScaleTCNBlock,
)
from lse_recognition.models.stgcn import (
    STGCNBlock,
    STGCNSignClassifier,
    SpatialGraphConv,
    build_hand_adjacency_matrix,
)
from lse_recognition.models.tcn import (
    TCNResidualBlock,
    TCNSignClassifier,
)


def create_model(
    config: Dict,
    model_type: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory unificada para instanciar cualquier modelo de la suite.

    Args:
        config: Diccionario de configuración con hiperparámetros.
        model_type: Tipo de modelo ('tcn', 'mstcn', 'lstm', 'attention_lstm', 'stgcn').
                    Si es None, se lee de config['model_type'] o por defecto 'tcn'.
        device: Dispositivo PyTorch (CPU o CUDA).

    Returns:
        Instancia de nn.Module transferida al dispositivo.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_type = (model_type or config.get("model_type", "tcn")).lower()

    if m_type in ("tcn", "standard_tcn"):
        model = TCNSignClassifier(config)
    elif m_type in ("mstcn", "ms_tcn", "multi_scale_tcn"):
        model = MSTCNSignClassifier(config)
    elif m_type in ("lstm", "bilstm"):
        model = LSTMSignClassifier(config)
    elif m_type in ("attention_lstm", "attn_lstm", "attention_bilstm"):
        model = AttentionLSTMSignClassifier(config)
    elif m_type in ("stgcn", "st_gcn", "graph_tcn"):
        model = STGCNSignClassifier(config)
    else:
        raise ValueError(
            f"Tipo de modelo desconocido: '{m_type}'. "
            f"Opciones válidas: 'tcn', 'mstcn', 'lstm', 'attention_lstm', 'stgcn'"
        )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"✅ Modelo [{m_type.upper()}] instanciado: {total_params:,} parámetros totales, "
        f"{trainable_params:,} entrenables en [{device}]"
    )

    return model


__all__ = [
    "TCNResidualBlock",
    "TCNSignClassifier",
    "MultiScaleTCNBlock",
    "ChannelAttention",
    "MSTCNSignClassifier",
    "LSTMSignClassifier",
    "TemporalAttention",
    "AttentionLSTMSignClassifier",
    "SpatialGraphConv",
    "STGCNBlock",
    "STGCNSignClassifier",
    "build_hand_adjacency_matrix",
    "create_model",
]
