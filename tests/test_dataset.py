"""
tests/test_dataset.py — Tests del módulo data.dataset
======================================================
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Permite importar sin instalación del paquete
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lse_recognition.data.dataset import LandmarksNormalizer, SignLanguageDataset


# ------------------------------------------------------------------ #
# LandmarksNormalizer                                                 #
# ------------------------------------------------------------------ #

class TestLandmarksNormalizer:
    """Tests de la normalización geométrica."""

    def test_output_shape_preserved(self):
        """La normalización no debe cambiar el shape del array."""
        normalizer = LandmarksNormalizer()
        seq = np.random.rand(60, 126).astype(np.float32)
        out = normalizer(seq)
        assert out.shape == (60, 126), f"Shape incorrecto: {out.shape}"

    def test_zero_frame_not_modified(self):
        """Frames completamente a cero deben permanecer a cero."""
        normalizer = LandmarksNormalizer()
        seq = np.zeros((60, 126), dtype=np.float32)
        out = normalizer(seq)
        np.testing.assert_array_equal(out, seq)

    def test_normalization_centers_wrist(self):
        """
        Después de normalizar, la muñeca de la mano activa debe estar
        cerca del origen (valores próximos a cero).
        """
        normalizer = LandmarksNormalizer()
        seq = np.zeros((1, 126), dtype=np.float32)

        # Solo mano izquierda activa (landmarks 0-20)
        # Muñeca en (0.5, 0.5, 0), middle_mcp en (0.6, 0.5, 0)
        lm = seq.reshape(1, 42, 3)
        lm[0, 0] = [0.5, 0.5, 0.0]    # WRIST
        lm[0, 9] = [0.6, 0.5, 0.0]    # MIDDLE_MCP (scale = 0.1)
        # Otros puntos con valores no cero
        for i in range(1, 21):
            lm[0, i] = [0.5 + i * 0.01, 0.5, 0.0]

        seq = lm.reshape(1, 126)
        out = normalizer(seq)
        out_lm = out.reshape(1, 42, 3)

        # La muñeca normalizada debe ser (0, 0, 0)
        np.testing.assert_allclose(out_lm[0, 0], [0.0, 0.0, 0.0], atol=1e-5)

    def test_right_hand_zeroed_when_left_active(self):
        """Si solo la mano izquierda está activa, la derecha debe ser 0."""
        normalizer = LandmarksNormalizer()
        seq = np.zeros((1, 126), dtype=np.float32)
        lm = seq.reshape(1, 42, 3)

        # Solo mano izquierda (índices 0-20)
        lm[0, 0] = [0.5, 0.5, 0.0]   # WRIST
        lm[0, 9] = [0.6, 0.5, 0.0]   # MIDDLE_MCP
        lm[0, 5] = [0.55, 0.52, 0.0]

        seq = lm.reshape(1, 126)
        out = normalizer(seq)
        out_lm = out.reshape(1, 42, 3)

        # Mano derecha (índices 21-41) debe permanecer cero
        np.testing.assert_array_equal(out_lm[0, 21:], np.zeros((21, 3)))


# ------------------------------------------------------------------ #
# SignLanguageDataset                                                  #
# ------------------------------------------------------------------ #

class TestSignLanguageDataset:
    """Tests del Dataset con datos sintéticos."""

    @pytest.fixture
    def synthetic_dataset(self, tmp_path):
        """Crea un dataset sintético con archivos .npy y CSV."""
        import pandas as pd

        words = ["HOLA", "GRACIAS", "NO"]
        records = []

        for word in words:
            for sample_idx in range(3):
                # Crear landmark .npy (60 frames, 42 puntos, 3 coordenadas)
                lm = np.random.rand(60, 42, 3).astype(np.float32)
                npy_path = tmp_path / f"{word}_{sample_idx}.npy"
                np.save(npy_path, lm)

                records.append({
                    "word": word,
                    "landmarks_hands_only_file": str(npy_path),
                })

        csv_path = tmp_path / "metadata.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False)
        return csv_path, words

    def test_dataset_length(self, synthetic_dataset):
        """El dataset debe tener el número correcto de muestras."""
        csv_path, words = synthetic_dataset
        ds = SignLanguageDataset(str(csv_path), seq_length=60)
        assert len(ds) == len(words) * 3

    def test_getitem_returns_correct_shapes(self, synthetic_dataset):
        """__getitem__ debe devolver tensores con shapes correctos."""
        csv_path, words = synthetic_dataset
        ds = SignLanguageDataset(str(csv_path), seq_length=60)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (60, 126), f"Shape X incorrecto: {x.shape}"
        assert y.shape == (), f"Shape Y incorrecto: {y.shape}"

    def test_word_to_idx_has_all_classes(self, synthetic_dataset):
        """El mapping debe contener todas las clases del CSV."""
        csv_path, words = synthetic_dataset
        ds = SignLanguageDataset(str(csv_path), seq_length=60)
        for word in words:
            assert word in ds.word_to_idx, f"Clase '{word}' no en word_to_idx"

    def test_resample_shorter_sequence(self, tmp_path):
        """Secuencias más cortas deben ser re-muestreadas a seq_length."""
        import pandas as pd

        # Crear archivo con solo 30 frames (mitad de 60)
        lm = np.random.rand(30, 42, 3).astype(np.float32)
        npy_path = tmp_path / "short_seq.npy"
        np.save(npy_path, lm)

        records = [{"word": "TEST", "landmarks_hands_only_file": str(npy_path)}]
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False)

        ds = SignLanguageDataset(str(csv_path), seq_length=60)
        x, y = ds[0]
        assert x.shape == (60, 126)
