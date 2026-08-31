"""
tests/test_ingestion.py — Tests de ingesta y estandarización de manifiestos
===========================================================================
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lse_recognition.data.ingestion import DatasetManifestBuilder


class TestDatasetManifestBuilder:
    def test_create_mock_multisigner_dataset(self, tmp_path):
        """Genera un dataset sintético multi-signante funcional en disco."""
        builder = DatasetManifestBuilder(base_data_dir=tmp_path)
        words = ["HOLA", "ADIOS"]
        df, manifest_path = builder.create_mock_multisigner_dataset(
            words=words,
            num_signers=3,
            reps_per_signer=2,
            seq_length=30,
        )

        assert manifest_path.exists()
        assert len(df) == 2 * 3 * 2  # 12 muestras
        assert set(df["word"].unique()) == {"HOLA", "ADIOS"}
        assert set(df["signer_id"].unique()) == {"signer_01", "signer_02", "signer_03"}

        # Verificar que los archivos .npy se crearon realmente
        for lm_file in df["landmarks_hands_only_file"]:
            p = Path(lm_file)
            assert p.exists(), f"Archivo .npy no encontrado: {p}"

    def test_scan_directory_tree(self, tmp_path):
        """Escanea correctamente una jerarquía de carpetas de vídeos."""
        videos_root = tmp_path / "raw_videos"
        # Crear estructura: raw_videos/signer_01/HOLA/video1.mp4
        s1_hola = videos_root / "signer_01" / "HOLA"
        s1_hola.mkdir(parents=True, exist_ok=True)
        (s1_hola / "vid1.mp4").write_text("dummy video")
        (s1_hola / "vid2.mp4").write_text("dummy video")

        # Estructura: raw_videos/signer_02/NO/video3.mp4
        s2_no = videos_root / "signer_02" / "NO"
        s2_no.mkdir(parents=True, exist_ok=True)
        (s2_no / "vid3.mp4").write_text("dummy video")

        builder = DatasetManifestBuilder(base_data_dir=tmp_path)
        df = builder.scan_directory_tree(videos_root=videos_root)

        assert len(df) == 3
        assert set(df["word"].unique()) == {"HOLA", "NO"}
        assert set(df["signer_id"].unique()) == {"signer_01", "signer_02"}
