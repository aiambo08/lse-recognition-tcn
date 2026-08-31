"""
kinematics.py — Extracción de Features Cinemáticas y Espaciales para LSE
========================================================================

Enriquece las secuencias de landmarks 3D con descriptores de movimiento y
configuración articular de alto nivel:
    1. Coordenadas de posición normalizadas: P_t
    2. Velocidades temporales (1ª diferencia): V_t = P_t - P_{t-1}
    3. Aceleraciones temporales (2ª diferencia): A_t = V_t - V_{t-1}
    4. Distancias euclidianas inter-digitales clave:
       - Puntas de dedos entre sí (apertura/cierre: pulgar-índice, pulgar-medio, etc.)
       - Distancia muñeca a yemas (extensión/flexión)
       - Amplitud de la palma (ancho de nudillos)

Compatible como transform en `SignLanguageDataset` o como paso de preprocesamiento.
"""

from __future__ import annotations

from typing import List, Tuple, Union
import numpy as np


class KinematicFeatureExtractor:
    """
    Extractor de características cinemáticas y distancias anatómicas sobre secuencias de landmarks.

    Estructura de landmarks de manos (21 puntos por mano):
        0: WRIST
        1-4: THUMB (CMC, MCP, IP, TIP)
        5-8: INDEX (MCP, PIP, DIP, TIP)
        9-12: MIDDLE (MCP, PIP, DIP, TIP)
        13-16: RING (MCP, PIP, DIP, TIP)
        17-20: PINKY (MCP, PIP, DIP, TIP)

    Args:
        include_position: Incluir coordenadas 3D de posición (default: True)
        include_velocity: Incluir velocidades 3D temporales (default: True)
        include_acceleration: Incluir aceleraciones 3D temporales (default: False)
        include_distances: Incluir distancias inter-digitales clave (default: True)
        num_points: Número de landmarks por frame (default: 42 para 2 manos)
    """

    # Índices clave dentro de una mano de 21 puntos
    WRIST = 0
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_TIP = 20

    # Pares de distancias inter-digitales clave por mano (11 distancias por mano activa)
    KEY_DISTANCE_PAIRS: List[Tuple[int, int]] = [
        (THUMB_TIP, INDEX_TIP),     # Pinza pulgar-índice
        (THUMB_TIP, MIDDLE_TIP),    # Contacto pulgar-medio
        (THUMB_TIP, RING_TIP),      # Contacto pulgar-anular
        (THUMB_TIP, PINKY_TIP),     # Contacto pulgar-meñique
        (INDEX_TIP, MIDDLE_TIP),    # Separación índice-medio (V sign)
        (MIDDLE_TIP, RING_TIP),     # Separación medio-anular
        (RING_TIP, PINKY_TIP),      # Separación anular-meñique
        (WRIST, THUMB_TIP),         # Extensión pulgar
        (WRIST, INDEX_TIP),         # Extensión índice
        (WRIST, MIDDLE_TIP),        # Extensión medio
        (WRIST, PINKY_TIP),         # Extensión meñique
        (INDEX_MCP, PINKY_MCP),     # Ancho de la palma
    ]

    def __init__(
        self,
        include_position: bool = True,
        include_velocity: bool = True,
        include_acceleration: bool = False,
        include_distances: bool = True,
        num_points: int = 42,
    ):
        self.include_position = include_position
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.include_distances = include_distances
        self.num_points = num_points

    def compute_distances(self, lm_3d: np.ndarray) -> np.ndarray:
        """
        Calcula distancias inter-digitales para cada frame.

        Args:
            lm_3d: Array de shape (seq_len, num_points, 3)

        Returns:
            dist: Array de shape (seq_len, n_distances)
        """
        seq_len = lm_3d.shape[0]
        n_hands = self.num_points // 21  # Típicamente 2 manos
        n_pairs_per_hand = len(self.KEY_DISTANCE_PAIRS)
        total_pairs = n_pairs_per_hand * n_hands

        distances = np.zeros((seq_len, total_pairs), dtype=np.float32)

        for h in range(n_hands):
            offset = h * 21
            col_offset = h * n_pairs_per_hand

            for pair_idx, (p1, p2) in enumerate(self.KEY_DISTANCE_PAIRS):
                pt1 = lm_3d[:, offset + p1, :]  # (seq_len, 3)
                pt2 = lm_3d[:, offset + p2, :]  # (seq_len, 3)

                # Máscara para frames donde ambos puntos son distintos de cero
                active_mask = (np.linalg.norm(pt1, axis=1) > 1e-6) & (
                    np.linalg.norm(pt2, axis=1) > 1e-6
                )

                diff = pt1 - pt2
                dist = np.linalg.norm(diff, axis=1)
                dist[~active_mask] = 0.0

                distances[:, col_offset + pair_idx] = dist

        return distances

    def compute_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Calcula la velocidad temporal (1ª diferencia con padding forward).

        Args:
            positions: Array de shape (seq_len, D)

        Returns:
            velocities: Array de shape (seq_len, D)
        """
        vel = np.zeros_like(positions)
        if positions.shape[0] > 1:
            vel[1:] = positions[1:] - positions[:-1]
            vel[0] = vel[1]  # Padding suave en borde inicial
        return vel

    def compute_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        """
        Calcula la aceleración temporal (2ª diferencia).

        Args:
            velocities: Array de shape (seq_len, D)

        Returns:
            accelerations: Array de shape (seq_len, D)
        """
        acc = np.zeros_like(velocities)
        if velocities.shape[0] > 1:
            acc[1:] = velocities[1:] - velocities[:-1]
            acc[0] = acc[1]
        return acc

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Transforma una secuencia de landmarks en un vector plano de features enriquecidas.

        Args:
            landmarks: Array (seq_len, num_points * 3) o (seq_len, num_points, 3)

        Returns:
            features: Array (seq_len, total_features)
        """
        seq_len = landmarks.shape[0]

        if landmarks.ndim == 2:
            lm_3d = landmarks.reshape(seq_len, self.num_points, 3)
            pos_flat = landmarks.astype(np.float32)
        else:
            lm_3d = landmarks.astype(np.float32)
            pos_flat = landmarks.reshape(seq_len, -1).astype(np.float32)

        feature_blocks = []

        # 1. Posición
        if self.include_position:
            feature_blocks.append(pos_flat)

        # 2. Velocidad
        if self.include_velocity:
            vel = self.compute_velocity(pos_flat)
            feature_blocks.append(vel)

        # 3. Aceleración
        if self.include_acceleration:
            vel = self.compute_velocity(pos_flat) if not self.include_velocity else vel
            acc = self.compute_acceleration(vel)
            feature_blocks.append(acc)

        # 4. Distancias anatómicas
        if self.include_distances:
            dist = self.compute_distances(lm_3d)
            feature_blocks.append(dist)

        if not feature_blocks:
            return pos_flat

        return np.concatenate(feature_blocks, axis=1)

    def get_output_dim(self) -> int:
        """Calcula la dimensión total de salida por frame."""
        dim = 0
        raw_dim = self.num_points * 3
        if self.include_position:
            dim += raw_dim
        if self.include_velocity:
            dim += raw_dim
        if self.include_acceleration:
            dim += raw_dim
        if self.include_distances:
            n_hands = self.num_points // 21
            dim += len(self.KEY_DISTANCE_PAIRS) * n_hands
        return dim
