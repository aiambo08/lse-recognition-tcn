"""
tcn.py — Temporal Convolutional Network para clasificación de LSE
=================================================================

Arquitectura:
    1. Proyección lineal: input_features → projection_dim
    2. Stack de 4 bloques TCN residuales con dilaciones [1, 2, 4, 8]
    3. Global Average Pooling temporal
    4. Clasificador denso: channels[-1] → fc_hidden_dim → num_classes

Campo receptivo con k=3, dilaciones [1,2,4,8]:
    RF = 1 + 2*(1*2 + 2*2 + 4*2 + 8*2) = 61 frames

Ventajas sobre LSTM:
    - Paralelizable (más rápido en entrenamiento)
    - No sufre vanishing gradients
    - Menor memoria GPU
    - 677K parámetros, 38.5 MFLOPs por forward pass
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class TCNResidualBlock(nn.Module):
    """
    Bloque residual de TCN con convolución dilatada 1D.

    Pipeline:
        Conv1D(dilation) → BN → ReLU → Dropout →
        Conv1D(dilation) → BN → ReLU → Dropout  + shortcut

    La conexión residual es identidad si in_channels == out_channels,
    o una convolución 1×1 si cambia el número de canales.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        # Padding causal: mantiene la longitud temporal
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Proyección residual si cambia la dimensión de canales
        self.residual_projection: Optional[nn.Module] = None
        if in_channels != out_channels:
            self.residual_projection = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            out: (batch, out_channels, seq_len)
        """
        residual = x

        # Primera convolución dilatada
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # recortar padding para mantener seq_len
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Segunda convolución dilatada
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Conexión residual
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)

        return out + residual


class TCNSignClassifier(nn.Module):
    """
    Temporal Convolutional Network para clasificación de signos LSE.

    Args:
        config: Diccionario con al menos las claves:
            - input_features (int)
            - projection_dim (int)
            - tcn_channels   (List[int])
            - kernel_size    (int)
            - tcn_dropout    (float)
            - fc_hidden_dim  (int)
            - fc_dropout     (float)
            - num_classes    (int)
    """

    def __init__(self, config: Dict):
        super().__init__()

        # 1. Proyección inicial
        input_dim = config.get("input_features", 126)
        proj_dim = config.get("projection_dim", 128)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. Stack de bloques TCN con dilaciones exponenciales
        self.tcn_blocks = nn.ModuleList()
        channels = config.get("tcn_channels", [128, 128, 256, 256])
        in_channels = proj_dim
        kernel_size = config.get("kernel_size", 3)
        tcn_dropout = config.get("tcn_dropout", 0.2)

        for i, out_channels in enumerate(channels):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            self.tcn_blocks.append(
                TCNResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=tcn_dropout,
                )
            )
            in_channels = out_channels

        # 3. Global Average Pooling temporal
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 4. Clasificador
        fc_hidden = config.get("fc_hidden_dim", 256)
        fc_dropout = config.get("fc_dropout", 0.3)
        num_classes = config.get("num_classes", 10)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], fc_hidden),
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

        # 1. Proyección frame-by-frame
        x = x.view(batch_size * seq_len, feat_dim)
        x = self.projection(x)
        x = x.view(batch_size, seq_len, -1)

        # 2. Transponer para Conv1d: (batch, seq_len, ch) → (batch, ch, seq_len)
        x = x.permute(0, 2, 1)

        # 3. Bloques TCN
        for block in self.tcn_blocks:
            x = block(x)

        # 4. Global Average Pooling: (batch, ch, seq_len) → (batch, ch)
        x = self.global_pool(x).squeeze(-1)

        # 5. Clasificador
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción con probabilidades (softmax). Sin gradientes."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def get_receptive_field(self) -> int:
        """Calcula el campo receptivo total de la red."""
        rf = 1
        for i in range(len(self.tcn_blocks)):
            dilation = 2 ** i
            rf += 2 * dilation * (3 - 1)  # kernel_size = 3
        return rf


def create_model(config: Dict, device: Optional[torch.device] = None) -> TCNSignClassifier:
    """
    Crea un TCNSignClassifier y lo transfiere al dispositivo indicado.

    Imprime estadísticas del modelo (parámetros, campo receptivo, forward pass).

    Args:
        config: Diccionario de configuración completo.
        device: Dispositivo PyTorch. Si None, auto-detecta CUDA.

    Returns:
        Modelo creado y en el dispositivo correcto.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCNSignClassifier(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"✅ Modelo TCN creado: {total_params:,} parámetros totales, "
        f"{trainable_params:,} entrenables"
    )

    rf = model.get_receptive_field()
    print(f"   Campo receptivo: {rf} frames (de {config['seq_length']} totales)")

    # Smoke test con tensor aleatorio
    dummy = torch.randn(2, config["seq_length"], config["input_features"]).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"   Dummy input: {tuple(dummy.shape)} → output: {tuple(out.shape)}")

    return model
