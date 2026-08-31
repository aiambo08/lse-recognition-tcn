"""
ingestion.py — Motor de Ingesta y Estandarización de Datasets de LSE
===================================================================

Estandariza cualquier fuente de datos (vídeos locales, carpetas por signante,
DILSE, o subconjuntos del Corpus LSE) en un manifiesto CSV unificado con:
    ['sample_id', 'word', 'signer_id', 'video_path', 'landmarks_hands_only_file', 'duration_frames', 'hand_type']
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class DatasetManifestBuilder:
    """
    Construye y valida manifiestos estandarizados para entrenamiento y evaluación.

    Soporta dos estructuras de directorios comunes:
        1. Jerárquica: `raw_videos/<signer_id>/<word>/<video>.mp4`
        2. Plana con metadatos: `raw_videos/<word>_<signer_id>_<idx>.mp4`
    """

    def __init__(self, base_data_dir: str | Path = "data"):
        self.base_dir = Path(base_data_dir)
        self.metadata_dir = self.base_dir / "metadata"
        self.raw_videos_dir = self.base_dir / "raw_videos"
        self.landmarks_dir = self.base_dir / "landmarks_hands_only"

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.landmarks_dir.mkdir(parents=True, exist_ok=True)

    def scan_directory_tree(
        self,
        videos_root: Optional[str | Path] = None,
        video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    ) -> pd.DataFrame:
        """
        Escanea el árbol de directorios de vídeos y extrae metadatos.

        Formato esperado:
            `<root>/<signer_id>/<word>/<file>.mp4` o `<root>/<word>/<signer_id>_<file>.mp4`
        """
        root = Path(videos_root) if videos_root else self.raw_videos_dir
        if not root.exists():
            raise FileNotFoundError(f"Directorio de vídeos no encontrado: {root}")

        records = []
        sample_counter = 0

        for video_file in root.rglob("*"):
            if video_file.suffix.lower() not in video_extensions:
                continue

            rel_parts = video_file.relative_to(root).parts
            if len(rel_parts) >= 3:
                # Estructura: signer_id / word / filename
                signer_id = rel_parts[0]
                word = rel_parts[1].upper()
            elif len(rel_parts) == 2:
                # Estructura: word / filename (con signer en filename o default)
                word = rel_parts[0].upper()
                name_parts = video_file.stem.split("_")
                signer_id = name_parts[0] if len(name_parts) > 1 else "signer_01"
            else:
                name_parts = video_file.stem.split("_")
                word = name_parts[0].upper() if name_parts else "UNKNOWN"
                signer_id = name_parts[1] if len(name_parts) > 1 else "signer_01"

            sample_id = f"{word}_{signer_id}_{sample_counter:04d}"
            landmarks_path = self.landmarks_dir / word / f"{sample_id}.npy"

            records.append({
                "sample_id": sample_id,
                "word": word,
                "signer_id": signer_id,
                "video_path": str(video_file),
                "landmarks_hands_only_file": str(landmarks_path),
            })
            sample_counter += 1

        df = pd.DataFrame(records)
        print(f"✅ Escaneo completado: {len(df)} vídeos encontrados | "
              f"{df['word'].nunique() if not df.empty else 0} palabras | "
              f"{df['signer_id'].nunique() if not df.empty else 0} signantes")
        return df

    def create_mock_multisigner_dataset(
        self,
        words: List[str] = ("HOLA", "GRACIAS", "POR_FAVOR", "SI", "NO"),
        num_signers: int = 4,
        reps_per_signer: int = 5,
        seq_length: int = 60,
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Genera un dataset sintético multi-signante para testing inmediato del pipeline.

        Crea archivos `.npy` reales en disco y un CSV de manifiesto.
        """
        records = []
        sample_idx = 0

        for s_idx in range(1, num_signers + 1):
            signer_id = f"signer_{s_idx:02d}"

            for word in words:
                word_dir = self.landmarks_dir / word
                word_dir.mkdir(parents=True, exist_ok=True)

                for rep in range(reps_per_signer):
                    sample_id = f"{word}_{signer_id}_rep{rep:02d}"
                    lm_path = word_dir / f"{sample_id}.npy"

                    # Simular landmarks de manos (seq_length, 42, 3) con ligera variación por signante
                    base_offset = s_idx * 0.05
                    lm_data = (
                        np.random.randn(seq_length, 42, 3).astype(np.float32) * 0.1
                        + base_offset
                    )
                    np.save(lm_path, lm_data)

                    records.append({
                        "sample_id": sample_id,
                        "word": word,
                        "signer_id": signer_id,
                        "landmarks_hands_only_file": str(lm_path),
                    })
                    sample_idx += 1

        df = pd.DataFrame(records)
        manifest_path = self.metadata_dir / "multisigner_manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"✅ Dataset sintético multi-signante creado en {manifest_path} ({len(df)} muestras)")
        return df, manifest_path
