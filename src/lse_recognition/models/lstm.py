"""
lstm.py — LSTM alternativo para clasificación de LSE
====================================================

Implementado como alternativa al TCN para comparación.
El checkpoint guardado incluye la clave 'lstm_hidden_size' en el config,
lo que permite detectar automáticamente el tipo de modelo.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class LSTMSignClassifier(nn.Module):
    """
    Clasificador de secuencias con proyección inicial y capas LSTM.

    Args:
        config: Diccionario con al menos:
            - input_features    (int)
            - projection_dim    (int)
            - lstm_hidden_size  (int)   ← distingue de TCN al cargar checkpoint
            - lstm_num_layers   (int)
            - lstm_dropout      (float)
            - fc_hidden_dim     (int)
            - fc_dropout        (float)
            - num_classes       (int)
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(config["input_features"], config["projection_dim"]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.lstm = nn.LSTM(
            input_size=config["projection_dim"],
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_num_layers"],
            batch_first=True,
            dropout=(
                config["lstm_dropout"] if config["lstm_num_layers"] > 1 else 0.0
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(config["lstm_hidden_size"], config["fc_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["fc_dropout"]),
            nn.Linear(config["fc_hidden_dim"], config["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_features)

        Returns:
            logits: (batch, num_classes)
        """
        bsz, seq_len, feat_dim = x.shape

        # Proyección
        x = x.view(bsz * seq_len, feat_dim)
        x = self.projection(x)
        x = x.view(bsz, seq_len, -1)

        # LSTM — usa solo el último hidden state
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]

        return self.classifier(last_hidden)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción con probabilidades (softmax). Sin gradientes."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
