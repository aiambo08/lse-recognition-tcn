"""
extraction.py — Extractor por Lotes de Landmarks de Vídeos LSE
==============================================================

Procesa vídeos MP4/AVI en lote extrayendo secuencias temporales de landmarks
articulares con MediaPipe Hands y guardándolas en formato .npy estándar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class BatchLandmarkExtractor:
    """
    Extractor de landmarks por lotes desde archivos de vídeo.

    Args:
        min_detection_confidence: Umbral de detección de MediaPipe
        min_tracking_confidence: Umbral de seguimiento de MediaPipe
        max_num_hands: Número máximo de manos a detectar (default: 2)
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe no está instalado en este entorno Python. "
                "Para extracción de vídeo instala: pip install mediapipe"
            )
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_from_video(self, video_path: str | Path) -> np.ndarray:
        """
        Extrae landmarks de manos de todos los frames de un vídeo.

        Returns:
            Array de shape (num_frames, 42, 3) con coordenadas (x, y, z).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el archivo de vídeo: {video_path}")

        frames_landmarks = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            frame_lm = np.zeros((42, 3), dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = hand_info.classification[0].label
                    offset = 0 if label == "Left" else 21

                    for idx, lm in enumerate(hand_lm.landmark):
                        frame_lm[offset + idx] = [lm.x, lm.y, lm.z]

            frames_landmarks.append(frame_lm)

        cap.release()

        if not frames_landmarks:
            return np.zeros((1, 42, 3), dtype=np.float32)

        return np.array(frames_landmarks, dtype=np.float32)

    def process_manifest(
        self,
        manifest_df: pd.DataFrame,
        output_dir: Optional[str | Path] = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Procesa todos los vídeos referenciados en un DataFrame de manifiesto.

        Args:
            manifest_df: DataFrame con columnas 'video_path' y opcionalmente 'landmarks_hands_only_file'.
            output_dir: Directorio de destino para los .npy.
            overwrite: Si False, omite vídeos cuyos .npy ya existan.

        Returns:
            DataFrame enriquecido con 'num_frames' y 'landmarks_extracted'.
        """
        out_base = Path(output_dir) if output_dir else Path("data/landmarks_hands_only")
        updated_records = []

        print(f"🎬 Iniciando extracción por lotes sobre {len(manifest_df)} vídeos...")

        for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extrayendo landmarks"):
            video_path = Path(row["video_path"])
            word = str(row.get("word", "UNKNOWN")).upper()
            sample_id = str(row.get("sample_id", video_path.stem))

            dest_dir = out_base / word
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{sample_id}.npy"

            row_dict = row.to_dict()
            row_dict["landmarks_hands_only_file"] = str(dest_path)

            if dest_path.exists() and not overwrite:
                lm_data = np.load(dest_path)
                row_dict["num_frames"] = len(lm_data)
                row_dict["landmarks_extracted"] = True
            else:
                try:
                    lm_data = self.extract_from_video(video_path)
                    np.save(dest_path, lm_data)
                    row_dict["num_frames"] = len(lm_data)
                    row_dict["landmarks_extracted"] = True
                except Exception as e:
                    print(f"⚠️ Error al procesar {video_path}: {e}")
                    row_dict["num_frames"] = 0
                    row_dict["landmarks_extracted"] = False

            updated_records.append(row_dict)

        result_df = pd.DataFrame(updated_records)
        print(f"✅ Extracción completada. {result_df['landmarks_extracted'].sum()}/{len(result_df)} vídeos procesados.")
        return result_df

    def close(self) -> None:
        """Libera los recursos de MediaPipe."""
        self.hands.close()
