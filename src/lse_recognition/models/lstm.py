"""
lstm.py — LSTM alternativo para clasificación de LSE
====================================================

Implementado como alternativa al TCN para comparación y benchmarking.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class LSTMSignClassifier(nn.Module):
    """
    Clasificador de secuencias con proyección inicial y capas LSTM.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        input_dim = config.get("input_features", 126)
        proj_dim = config.get("projection_dim", 128)
        hidden_size = config.get("lstm_hidden_size", config.get("lstm_hidden_dim", 128))
        num_layers = config.get("lstm_num_layers", 2)
        lstm_dropout = config.get("lstm_dropout", 0.3)
        fc_hidden = config.get("fc_hidden_dim", 256)
        fc_dropout = config.get("fc_dropout", 0.3)
        num_classes = config.get("num_classes", 10)

        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_features)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, feat_dim = x.shape
        x = x.view(batch_size * seq_len, feat_dim)
        x = self.projection(x)
        x = x.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_step = lstm_out[:, -1, :]  # último frame

        return self.classifier(last_step)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
