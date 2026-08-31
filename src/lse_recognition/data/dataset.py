"""
dataset.py — Dataset y normalización para LSE
=============================================

Contiene:
    - LandmarksNormalizer : Normalización geométrica frame-by-frame
    - SignLanguageDataset  : torch.utils.data.Dataset para landmarks LSE
    - create_dataloaders   : Factory para train/val/test DataLoaders
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class LandmarksNormalizer:
    """
    Normalización geométrica por frame.

    Hace el modelo invariante a:
        - Escala   → divide por distancia wrist → middle_mcp
        - Traslación → resta coordenadas de la muñeca

    Diseñado para hands-only landmarks (42 puntos × 3 coords = 126 features).

    Referencia MediaPipe Hands:
        - Landmark 0  : WRIST
        - Landmark 9  : MIDDLE_MCP
        - Landmarks 0-20  : mano izquierda
        - Landmarks 21-41 : mano derecha
    """

    WRIST = 0
    MIDDLE_MCP = 9

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Args:
            landmarks: Array de shape (seq_len, n_features) donde
                       n_features = n_points * 3

        Returns:
            landmarks normalizados, mismo shape que la entrada
        """
        seq_len, feat_dim = landmarks.shape
        lm = landmarks.reshape(seq_len, -1, 3)  # (T, 42, 3)

        for t in range(seq_len):
            frame = lm[t]

            # Detectar qué mano está activa
            left = frame[:21]
            right = frame[21:]
            left_active = np.any(left != 0)
            right_active = np.any(right != 0)

            if left_active:
                hand = left
            elif right_active:
                hand = right
            else:
                continue  # frame vacío, no normalizar

            # Centrado por muñeca
            wrist = hand[self.WRIST].copy()
            ref = hand[self.MIDDLE_MCP].copy()
            scale = np.linalg.norm(ref - wrist)

            if scale < 1e-6:
                continue  # evitar división por cero

            # Centrar y escalar solo la mano activa
            hand[:] = (hand - wrist) / scale

            # Anular la mano no activa
            if left_active and not right_active:
                frame[21:] = 0
            elif right_active and not left_active:
                frame[:21] = 0

            lm[t] = frame

        return lm.reshape(seq_len, -1)


class ComposeTransforms:
    """
    Aplica una lista de transformaciones en secuencia sobre arrays de landmarks.

    Ejemplo:
        transform = ComposeTransforms([
            LandmarksNormalizer(),
            KinematicFeatureExtractor(include_velocity=True)
        ])
    """

    def __init__(self, transforms: List):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


class SignLanguageDataset(Dataset):
    """
    Dataset para secuencias de landmarks de LSE.

    Args:
        metadata_csv: Ruta al CSV con columnas 'word' y
                      'landmarks_hands_only_file' (o 'landmarks_file').
        seq_length:   Longitud fija de secuencia en frames.
        transform:    Transform opcional (e.g. LandmarksNormalizer).
        word_to_idx:  Mapping palabra→índice externo. Si None, se deriva
                      del CSV (útil para el split de train).
        use_hands_only: Si True, usa la columna de hands-only (126 features),
                        si False, usa la columna de landmarks completos.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        seq_length: int = 60,
        transform=None,
        word_to_idx: Optional[Dict[str, int]] = None,
        use_hands_only: bool = True,
    ):
        self.df = pd.read_csv(metadata_csv)
        self.seq_length = seq_length
        self.transform = transform
        self.use_hands_only = use_hands_only

        # Crear o usar mapping existente
        if word_to_idx is None:
            words = sorted(self.df["word"].unique())
            self.word_to_idx: Dict[str, int] = {w: i for i, w in enumerate(words)}
        else:
            self.word_to_idx = dict(word_to_idx)

        self.idx_to_word: Dict[int, str] = {
            i: w for w, i in self.word_to_idx.items()
        }

        # Filtrar filas fuera del mapping (seguridad)
        self.df = self.df[
            self.df["word"].isin(self.word_to_idx.keys())
        ].reset_index(drop=True)

        # Seleccionar columna de landmarks
        hands_col = "landmarks_hands_only_file"
        full_col = "landmarks_file"
        if self.use_hands_only and hands_col in self.df.columns:
            self.landmark_column = hands_col
        else:
            self.landmark_column = full_col

        print(
            f"✅ Dataset cargado: {metadata_csv} | "
            f"muestras: {len(self.df)} | "
            f"clases: {len(self.word_to_idx)}"
        )

    # ------------------------------------------------------------------ #
    # Métodos internos                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resample_to_len(seq: np.ndarray, target_len: int) -> Optional[np.ndarray]:
        """Re-muestreo lineal a longitud fija."""
        T = len(seq)
        if T == 0:
            return None
        idx = np.linspace(0, T - 1, target_len).astype(int)
        return seq[idx]

    def _load_landmarks(self, row: pd.Series) -> np.ndarray:
        """Carga el archivo .npy, buscando rutas relativas si es necesario."""
        lm_path = Path(row[self.landmark_column])
        if not lm_path.exists():
            lm_path = Path("data") / lm_path
        if not lm_path.exists():
            raise FileNotFoundError(f"No se encuentra el fichero: {lm_path}")
        return np.load(lm_path).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Interfaz Dataset                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        landmarks = self._load_landmarks(row)

        # Asegurar longitud correcta
        if len(landmarks) != self.seq_length:
            resampled = self._resample_to_len(landmarks, self.seq_length)
            if resampled is None:
                n_pts = landmarks.shape[1] if landmarks.ndim == 3 else 42
                landmarks = np.zeros(
                    (self.seq_length, n_pts, 3), dtype=np.float32
                )
            else:
                landmarks = resampled

        # Aplanar: (seq_len, n_points, 3) → (seq_len, n_points*3)
        seq_flat = landmarks.reshape(self.seq_length, -1)

        if self.transform is not None:
            seq_flat = self.transform(seq_flat)

        label = self.word_to_idx[row["word"]]
        return torch.from_numpy(seq_flat).float(), torch.tensor(label, dtype=torch.long)


# ------------------------------------------------------------------ #
# Factory de DataLoaders                                              #
# ------------------------------------------------------------------ #

def create_dataloaders(
    config: Dict,
    use_hands_only: bool = True,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    Lee los CSV de train/val/test y crea DataLoaders.

    Detecta automáticamente el CSV augmentado de train si existe.
    Actualiza config['num_classes'] e config['input_features'] en el lugar.

    Args:
        config:        Diccionario de configuración (se modifica in-place).
        use_hands_only: Usar solo landmarks de manos.
        normalize:     Aplicar LandmarksNormalizer.

    Returns:
        (train_loader, val_loader, test_loader, word_to_idx, idx_to_word)
    """
    normalizer = LandmarksNormalizer() if normalize else None

    base_meta = Path("data/metadata")
    train_candidates = ["train_split_augmented.csv", "train_split.csv"]
    val_candidates = ["val_split.csv"]
    test_candidates = ["test_split.csv"]

    def _pick_csv(candidates: List[str]) -> Optional[str]:
        for name in candidates:
            p = base_meta / name
            if p.exists():
                return str(p)
        return None

    train_csv = _pick_csv(train_candidates)
    val_csv = _pick_csv(val_candidates)
    test_csv = _pick_csv(test_candidates)

    if not all([train_csv, val_csv, test_csv]):
        missing = [
            name
            for name, csv in [("train", train_csv), ("val", val_csv), ("test", test_csv)]
            if csv is None
        ]
        raise FileNotFoundError(
            f"No se encontraron los CSV necesarios en {base_meta}. "
            f"Faltan: {missing}. "
            "Ejecuta fase1.ipynb para generarlos."
        )

    print(f"📄 Usando splits:")
    print(f"   train: {train_csv}")
    print(f"   val:   {val_csv}")
    print(f"   test:  {test_csv}")

    # Dataset de train (define el word_to_idx)
    train_dataset = SignLanguageDataset(
        train_csv,
        seq_length=config["seq_length"],
        transform=normalizer,
        word_to_idx=None,
        use_hands_only=use_hands_only,
    )
    word_to_idx = train_dataset.word_to_idx
    idx_to_word = train_dataset.idx_to_word

    # Actualizar config con dimensiones reales
    config["num_classes"] = len(word_to_idx)
    first_path = Path(train_dataset.df.iloc[0][train_dataset.landmark_column])
    if not first_path.exists():
        first_path = Path("data") / first_path
    sample = np.load(first_path)
    n_points = sample.shape[1]  # 42 para hands-only, 75 para full
    config["input_features"] = n_points * 3

    # Datasets de val y test con el mismo mapping
    val_dataset = SignLanguageDataset(
        val_csv,
        seq_length=config["seq_length"],
        transform=normalizer,
        word_to_idx=word_to_idx,
        use_hands_only=use_hands_only,
    )
    test_dataset = SignLanguageDataset(
        test_csv,
        seq_length=config["seq_length"],
        transform=normalizer,
        word_to_idx=word_to_idx,
        use_hands_only=use_hands_only,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    # Guardar mapping para inferencia posterior
    model_dir = Path(config.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "label_mapping.json", "w") as f:
        json.dump(word_to_idx, f, indent=2)

    print(
        f"✅ DataLoaders creados. "
        f"Train: {len(train_dataset)} | "
        f"Clases: {config['num_classes']} | "
        f"Input dim: {config['input_features']}"
    )
    return train_loader, val_loader, test_loader, word_to_idx, idx_to_word
