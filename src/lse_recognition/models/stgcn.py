"""
stgcn.py — Spatial-Temporal Graph Convolutional Network (ST-GCN) para LSE
========================================================================

Modela las manos como grafos biomecánicos naturales:
    - Nodos: 42 landmarks articulares (21 mano izquierda + 21 mano derecha).
    - Aristas: Conexiones óseas anatómicas (muñeca → articulaciones → yemas).
    - Convolución espacial sobre el grafo esquelético:
      H^{(l+1)} = \\sigma(\\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2} H^{(l)} W^{(l)})
    - Convolución temporal 1D a lo largo de los frames temporales.
    - Bloques ST-GCN residuales apilados con Global Pooling y Clasificador MLP.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conexiones anatómicas MediaPipe Hands (21 articulaciones por mano)
HAND_EDGES_SINGLE = [
    # Pulgar
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Índice
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Medio
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Anular
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Meñique
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Conexiones transversales de la palma (MCPs)
    (5, 9), (9, 13), (13, 17),
]


def build_hand_adjacency_matrix(num_nodes: int = 42) -> torch.Tensor:
    """
    Construye la matriz de adyacencia normalizada para 2 manos (42 nodos).
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Mano izquierda (0-20)
    for u, v in HAND_EDGES_SINGLE:
        if u < num_nodes and v < num_nodes:
            A[u, v] = 1.0
            A[v, u] = 1.0

    # Mano derecha (21-41)
    if num_nodes >= 42:
        for u, v in HAND_EDGES_SINGLE:
            ur, vr = u + 21, v + 21
            A[ur, vr] = 1.0
            A[vr, ur] = 1.0
        # Conexión virtual entre muñecas para interacción bimanual
        A[0, 21] = 0.5
        A[21, 0] = 0.5

    # Añadir auto-bucles (A_tilde = A + I)
    A_tilde = A + np.eye(num_nodes, dtype=np.float32)

    # Grado normalizado D^(-1/2) * A_tilde * D^(-1/2)
    degrees = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_mat = np.diag(d_inv_sqrt)

    A_norm = D_mat @ A_tilde @ D_mat
    return torch.from_numpy(A_norm).float()


class SpatialGraphConv(nn.Module):
    """
    Capa de convolución espacial sobre el grafo esquelético.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 42):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer("A_norm", build_hand_adjacency_matrix(num_nodes))

        # Matriz de pesos adaptativa aprendible (Edge Attention / Adaptive Graph)
        self.adaptive_weight = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_nodes, in_channels)

        Returns:
            out: (batch, seq_len, num_nodes, out_channels)
        """
        batch_size, seq_len, num_nodes, in_ch = x.shape

        # Adyacencia combinada estática normalizada + adaptativa aprendible
        A = self.A_norm + torch.tanh(self.adaptive_weight)

        # Multiplicación de grafo: (batch * seq_len, num_nodes, in_channels)
        x_flat = x.view(batch_size * seq_len, num_nodes, in_ch)
        ax = torch.matmul(A, x_flat)  # (batch * seq_len, num_nodes, in_ch)

        # Proyección lineal
        out = self.linear(ax)  # (batch * seq_len, num_nodes, out_ch)

        # BatchNorm sobre out_channels
        out = out.permute(0, 2, 1)  # (batch * seq_len, out_ch, num_nodes)
        out = self.bn(out)
        out = out.permute(0, 2, 1)  # (batch * seq_len, num_nodes, out_ch)

        return out.view(batch_size, seq_len, num_nodes, -1)


class STGCNBlock(nn.Module):
    """
    Bloque espacio-temporal completo (Spatial GCN + Temporal Conv1D + Residual).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int = 42,
        temporal_kernel: int = 5,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, num_nodes)

        # Convolución temporal 1D sobre el eje de tiempo
        pad = (temporal_kernel - 1) * dilation
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                out_channels * num_nodes,
                out_channels * num_nodes,
                kernel_size=temporal_kernel,
                dilation=dilation,
                padding=pad,
                groups=num_nodes,  # convolución por articulación
            ),
            nn.BatchNorm1d(out_channels * num_nodes),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pad = pad
        self.num_nodes = num_nodes
        self.out_channels = out_channels

        # Shortcut residual
        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_nodes, in_channels)
        """
        batch_size, seq_len, num_nodes, in_ch = x.shape
        res = self.residual(x)

        # 1. Spatial GCN
        h = self.gcn(x)  # (batch, seq_len, num_nodes, out_channels)

        # 2. Temporal Conv
        # Reshape a (batch, num_nodes * out_channels, seq_len)
        h = h.permute(0, 2, 3, 1).contiguous().view(batch_size, num_nodes * self.out_channels, seq_len)
        h = self.temporal_conv(h)
        if self.pad > 0:
            h = h[:, :, :-self.pad]  # causal trimming

        h = h.view(batch_size, num_nodes, self.out_channels, seq_len).permute(0, 3, 1, 2).contiguous()

        return F.gelu(h + res)


class STGCNSignClassifier(nn.Module):
    """
    Clasificador Spatial-Temporal Graph Convolutional Network (ST-GCN) para LSE.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        num_classes = config.get("num_classes", 10)
        num_nodes = config.get("num_nodes", 42)
        node_features = config.get("node_features", 3)  # x, y, z por landmark
        hidden_dim = config.get("stgcn_hidden_dim", 64)
        dropout = config.get("stgcn_dropout", 0.2)

        # 1. Proyección inicial de nodo
        self.node_proj = nn.Linear(node_features, hidden_dim)

        # 2. Bloques ST-GCN apilados con dilatación creciente
        self.blocks = nn.ModuleList([
            STGCNBlock(hidden_dim, hidden_dim, num_nodes=num_nodes, dilation=1, dropout=dropout),
            STGCNBlock(hidden_dim, hidden_dim * 2, num_nodes=num_nodes, dilation=2, dropout=dropout),
            STGCNBlock(hidden_dim * 2, hidden_dim * 2, num_nodes=num_nodes, dilation=4, dropout=dropout),
        ])

        final_channels = hidden_dim * 2

        # 3. Clasificador con Global Joint & Temporal Pooling
        self.classifier = nn.Sequential(
            nn.Linear(final_channels * 2, config.get("fc_hidden_dim", 256)),
            nn.LayerNorm(config.get("fc_hidden_dim", 256)),
            nn.GELU(),
            nn.Dropout(config.get("fc_dropout", 0.3)),
            nn.Linear(config.get("fc_hidden_dim", 256), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_features) donde input_features = num_nodes * 3
        """
        batch_size, seq_len, feat_dim = x.shape
        num_nodes = 42
        node_features = feat_dim // num_nodes if feat_dim % num_nodes == 0 else 3

        # Reshape a representación de grafo: (batch, seq_len, num_nodes, node_features)
        h = x.view(batch_size, seq_len, num_nodes, node_features)
        h = self.node_proj(h)

        for block in self.blocks:
            h = block(h)

        # Global Joint Pooling (promedio sobre articulaciones)
        h_joints = torch.mean(h, dim=2)  # (batch, seq_len, final_channels)

        # Global Temporal Pooling (Avg + Max)
        avg_pool = torch.mean(h_joints, dim=1)  # (batch, final_channels)
        max_pool, _ = torch.max(h_joints, dim=1)  # (batch, final_channels)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, final_channels * 2)

        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
