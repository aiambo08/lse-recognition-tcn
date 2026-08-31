"""
attention_lstm.py — BiLSTM con Temporal Self-Attention para LSE
==============================================================

Arquitectura con mecanismo de atención temporal:
    1. Proyección lineal: input_features → hidden_dim
    2. Stack Bi-direccional LSTM de N capas
    3. Mecanismo de Atención Temporal:
       Calcula un vector de pesos de importancia \\alpha_t \\in [0, 1] para cada frame t,
       permitiendo al modelo focalizarse en los momentos clave del signo y descartar
       los frames de transición o descanso.
    4. Clasificador MLP sobre el vector de contexto ponderado c = \\sum_t \\alpha_t h_t.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Mecanismo de Atención Temporal Aditiva (Bahdanau-style).
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch, seq_len, hidden_dim)

        Returns:
            context: (batch, hidden_dim)
            weights: (batch, seq_len)
        """
        # Calcular scores de energía: (batch, seq_len, 1)
        energy = self.projection(lstm_outputs)

        # Normalizar con softmax a lo largo del tiempo
        weights = F.softmax(energy.squeeze(-1), dim=1)  # (batch, seq_len)

        # Context vector: combinación lineal ponderada de estados ocultos
        context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)  # (batch, hidden_dim)

        return context, weights


class AttentionLSTMSignClassifier(nn.Module):
    """
    Clasificador BiLSTM con Atención Temporal para reconocimiento de señas LSE.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        input_dim = config.get("input_features", 126)
        hidden_dim = config.get("lstm_hidden_dim", 128)
        num_layers = config.get("lstm_num_layers", 2)
        num_classes = config.get("num_classes", 10)
        dropout = config.get("lstm_dropout", 0.3)
        bidirectional = config.get("lstm_bidirectional", True)

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 1. Proyección previa
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Stack LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. Capa de Atención Temporal
        self.attention = TemporalAttention(
            hidden_dim=lstm_output_dim,
            attention_dim=config.get("attention_dim", 128),
        )

        # 4. Clasificador MLP
        fc_hidden = config.get("fc_hidden_dim", 256)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Dropout(config.get("fc_dropout", 0.3)),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_features)
            return_attention: Si es True, retorna (logits, attention_weights)

        Returns:
            logits: (batch, num_classes)
        """
        # 1. Proyección
        h = self.input_projection(x)  # (batch, seq_len, hidden_dim)

        # 2. LSTM
        lstm_out, _ = self.lstm(h)  # (batch, seq_len, lstm_output_dim)

        # 3. Atención temporal
        context, attn_weights = self.attention(lstm_out)

        # 4. Clasificación
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilidades de clase con Softmax."""
        with torch.no_grad():
            logits = self.forward(x, return_attention=False)
            return torch.softmax(logits, dim=1)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
