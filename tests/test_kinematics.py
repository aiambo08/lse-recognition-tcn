"""
tests/test_kinematics.py — Tests del extractor de features cinemáticas
=====================================================================
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lse_recognition.data.kinematics import KinematicFeatureExtractor
from lse_recognition.data.dataset import ComposeTransforms, LandmarksNormalizer


class TestKinematicFeatureExtractor:
    def test_output_dimension_calculation(self):
        """Calcula correctamente la dimensión de salida configurada."""
        # 42 puntos x 3 = 126
        # Distancias: 12 pares x 2 manos = 24
        extractor = KinematicFeatureExtractor(
            include_position=True,
            include_velocity=True,
            include_acceleration=False,
            include_distances=True,
            num_points=42,
        )
        expected_dim = 126 + 126 + 24  # 276
        assert extractor.get_output_dim() == expected_dim

    def test_feature_extraction_shape_2d_input(self):
        """Procesa arrays 2D (seq_len, 126) y genera la dimensión correcta."""
        extractor = KinematicFeatureExtractor(
            include_position=True,
            include_velocity=True,
            include_acceleration=True,
            include_distances=True,
            num_points=42,
        )
        seq = np.random.randn(60, 126).astype(np.float32)
        out = extractor(seq)

        expected_dim = 126 * 3 + 24  # pos(126) + vel(126) + acc(126) + dist(24) = 402
        assert out.shape == (60, expected_dim)

    def test_feature_extraction_shape_3d_input(self):
        """Procesa arrays 3D (seq_len, 42, 3) directamente."""
        extractor = KinematicFeatureExtractor(
            include_position=True,
            include_velocity=True,
            include_acceleration=False,
            include_distances=False,
            num_points=42,
        )
        seq = np.random.randn(60, 42, 3).astype(np.float32)
        out = extractor(seq)
        assert out.shape == (60, 252)  # pos(126) + vel(126)

    def test_velocity_zero_for_static_sequence(self):
        """La velocidad debe ser 0 para una secuencia estática idéntica frame a frame."""
        extractor = KinematicFeatureExtractor(
            include_position=False,
            include_velocity=True,
            include_acceleration=False,
            include_distances=False,
            num_points=42,
        )
        static_frame = np.ones((1, 126), dtype=np.float32) * 0.5
        seq = np.repeat(static_frame, 30, axis=0)  # (30, 126)

        vel = extractor(seq)
        np.testing.assert_allclose(vel, np.zeros((30, 126)), atol=1e-6)

    def test_distances_invariance_under_translation(self):
        """Las distancias inter-digitales deben ser invariantes ante traslación global."""
        extractor = KinematicFeatureExtractor(
            include_position=False,
            include_velocity=False,
            include_acceleration=False,
            include_distances=True,
            num_points=42,
        )
        # Secuencia base
        seq1 = np.random.rand(10, 42, 3).astype(np.float32) + 0.1
        # Secuencia trasladada en +5.0 en todos los ejes
        seq2 = seq1 + 5.0

        dist1 = extractor(seq1)
        dist2 = extractor(seq2)

        np.testing.assert_allclose(dist1, dist2, atol=1e-5)

    def test_compose_transforms_pipeline(self):
        """Prueba ComposeTransforms combinando normalización y cinemática."""
        pipeline = ComposeTransforms([
            LandmarksNormalizer(),
            KinematicFeatureExtractor(
                include_position=True,
                include_velocity=True,
                include_distances=True,
            )
        ])
        seq = np.random.rand(60, 126).astype(np.float32)
        out = pipeline(seq)
        assert out.shape == (60, 276)
        assert not np.isnan(out).any()
