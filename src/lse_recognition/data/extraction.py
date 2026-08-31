"""
extraction.py — Extractor Universal de Landmarks de Vídeos LSE
==============================================================

Compatible con todas las versiones de MediaPipe:
    - Legacy API: `mediapipe.solutions.hands` (MediaPipe <= 0.10.14)
    - Modern Tasks API: `mediapipe.tasks.vision.HandLandmarker` (MediaPipe >= 0.10.15)
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


class BatchLandmarkExtractor:
    """
    Extractor universal de landmarks de manos desde archivos de vídeo.

    Descarga automáticamente el modelo `hand_landmarker.task` si es necesario.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
        model_path: Optional[str | Path] = None,
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe no está instalado en este entorno Python. "
                "Para extracción de vídeo instala: pip install mediapipe"
            )

        self.mode = "legacy"
        self.detector = None

        # Intentar legacy API primero
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mode = "legacy"
            self.mp_hands = mp.solutions.hands
            self.detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        # Fallback a modern Tasks API
        elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            self.mode = "tasks"
            if model_path is None:
                model_dir = Path("models")
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "hand_landmarker.task"
                if not model_path.exists():
                    print(f"📥 Descargando modelo MediaPipe Tasks desde {MODEL_URL}...")
                    urllib.request.urlretrieve(MODEL_URL, str(model_path))

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.detector = HandLandmarker.create_from_options(options)
        else:
            raise RuntimeError("No se pudo inicializar ningún backend de MediaPipe.")

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
            frame_lm = np.zeros((42, 3), dtype=np.float32)

            if self.mode == "legacy":
                results = self.detector.process(frame_rgb)
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_lm, hand_info in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        label = hand_info.classification[0].label
                        offset = 0 if label == "Left" else 21
                        for idx, lm in enumerate(hand_lm.landmark):
                            frame_lm[offset + idx] = [lm.x, lm.y, lm.z]

            elif self.mode == "tasks":
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = self.detector.detect(mp_image)

                if results.hand_landmarks and results.handedness:
                    for hand_lm, hand_info in zip(
                        results.hand_landmarks, results.handedness
                    ):
                        label = hand_info[0].category_name
                        offset = 0 if label == "Left" else 21
                        for idx, lm in enumerate(hand_lm):
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
        """Libera los recursos."""
        if hasattr(self.detector, "close"):
            self.detector.close()
