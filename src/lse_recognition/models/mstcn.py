"""
mstcn.py — Multi-Scale Temporal Convolutional Network (MS-TCN) para LSE
=====================================================================

Arquitectura Avanzada para Reconocimiento de Lengua de Signos:
    1. Proyección lineal inicial: input_features → hidden_channels
    2. Stack de bloques Multi-Scale TCN residuales:
       - Cada bloque ejecuta en paralelo convoluciones causales dilatadas con
         múltiples tamaños de kernel (k=3 para micro-movimientos de dedos,
         k=5 para transiciones intermedias, k=7 para macro-movimientos de brazos).
       - Mecanismo de Squeeze-and-Excitation (Channel Attention) para reponderar
         dinámicamente las escalas temporales más informativas.
       - Conexión residual con proyección 1x1.
    3. Global Temporal Pooling (Avg + Max pooling concatenado).
    4. Clasificador MLP profundo con Dropout y Normalización.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block para reponderación adaptativa de canales.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_ch = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_ch),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        w = self.fc(x).unsqueeze(-1)  # (batch, channels, 1)
        return x * w


class MultiScaleTCNBlock(nn.Module):
    """
    Bloque TCN Multi-Escala con ramas temporales paralelas y Channel Attention.

    Ejecuta ramas con diferentes kernels (k=3, k=5, k=7) a la misma tasa de
    dilatación, fusiona sus salidas y aplica atención de canal residual.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.dilation = dilation
        branch_channels = max(out_channels // len(kernel_sizes), 16)
        total_branch_ch = branch_channels * len(kernel_sizes)

        # Ramas paralelas con diferentes tamaños de kernel
        self.branches = nn.ModuleList()
        self.paddings = []

        for k in kernel_sizes:
            pad = (k - 1) * dilation
            self.paddings.append(pad)
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    branch_channels,
                    kernel_size=k,
                    dilation=dilation,
                    padding=pad,
                ),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.branches.append(branch)

        # Fusión y proyección de salida
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(total_branch_ch, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Squeeze-and-Excitation
        self.se = ChannelAttention(out_channels)

        # Conexión residual
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            out: (batch, out_channels, seq_len)
        """
        branch_outs = []
        for branch, pad in zip(self.branches, self.paddings):
            b_out = branch(x)
            if pad > 0:
                b_out = b_out[:, :, :-pad]  # causal trimming
            branch_outs.append(b_out)

        # Concatenar todas las escalas temporales
        fused = torch.cat(branch_outs, dim=1)
        fused = self.fuse_conv(fused)
        fused = self.se(fused)

        # Residual shortcut
        res = self.residual(x)
        return F.gelu(fused + res)


class MSTCNSignClassifier(nn.Module):
    """
    Clasificador Multi-Scale TCN completo para Sign Language Recognition.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        input_dim = config.get("input_features", 126)
        num_classes = config.get("num_classes", 10)
        proj_dim = config.get("projection_dim", 128)
        dropout = config.get("tcn_dropout", 0.2)
        kernel_sizes = tuple(config.get("ms_kernel_sizes", (3, 5, 7)))

        # Canales por capa
        channels = config.get("tcn_channels", [128, 128, 256, 256])

        # 1. Proyección lineal de entrada
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Bloques MS-TCN con dilataciones exponenciales
        self.blocks = nn.ModuleList()
        current_ch = proj_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            self.blocks.append(
                MultiScaleTCNBlock(
                    in_channels=current_ch,
                    out_channels=out_ch,
                    dilation=dilation,
                    kernel_sizes=kernel_sizes,
                    dropout=dropout,
                )
            )
            current_ch = out_ch

        # 3. Clasificador denso con pooling dual (Avg + Max)
        self.classifier = nn.Sequential(
            nn.Linear(current_ch * 2, config.get("fc_hidden_dim", 256)),
            nn.LayerNorm(config.get("fc_hidden_dim", 256)),
            nn.GELU(),
            nn.Dropout(config.get("fc_dropout", 0.3)),
            nn.Linear(config.get("fc_hidden_dim", 256), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_features)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, feat_dim = x.shape

        # Proyección frame por frame
        h = self.input_projection(x)  # (batch, seq_len, proj_dim)

        # Transponer para Conv1d: (batch, proj_dim, seq_len)
        h = h.permute(0, 2, 1)

        # Pasar por bloques MS-TCN
        for block in self.blocks:
            h = block(h)

        # Pooling dual temporal (Avg + Max)
        avg_pool = torch.mean(h, dim=2)  # (batch, current_ch)
        max_pool, _ = torch.max(h, dim=2)  # (batch, current_ch)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, current_ch * 2)

        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilidades softmax sin cálculo de gradiente."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
