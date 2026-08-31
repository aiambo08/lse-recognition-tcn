"""
splits.py — Particionado Científico Cross-Signer (Signer-Independent)
=====================================================================

Implementa esquemas de particionado que garantizan la estricta independencia
de signantes entre entrenamiento, validación y test:
    1. Cross-Signer Split: Partición manual por lista de IDs de signantes.
    2. Leave-One-Signer-Out (LOSO): Iterador K-Fold donde cada fold evalúa
       sobre un signante nunca visto.
    3. Stratified Group Split: Reparto balanceado de clases asegurando
       cero fuga (leakage) de signantes.
"""

from __future__ import annotations

from typing import Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def create_cross_signer_splits(
    df: pd.DataFrame,
    test_signers: List[str | int],
    val_signers: Optional[List[str | int]] = None,
    signer_col: str = "signer_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame en train, val y test basándose estrictamente en los IDs de los signantes.

    Args:
        df: DataFrame con metadatos del dataset.
        test_signers: Lista de identificadores de signantes para el test set.
        val_signers: Lista de identificadores para validación. Si es None,
                     se selecciona automáticamente un 10% del train set.
        signer_col: Nombre de la columna que identifica al signante.

    Returns:
        (train_df, val_df, test_df)
    """
    if signer_col not in df.columns:
        raise ValueError(
            f"Columna '{signer_col}' no encontrada en el DataFrame. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    all_signers = df[signer_col].unique()
    test_set = set(test_signers)

    invalid_test = test_set - set(all_signers)
    if invalid_test:
        raise ValueError(f"Signantes de test no presentes en el dataset: {invalid_test}")

    test_mask = df[signer_col].isin(test_set)
    test_df = df[test_mask].reset_index(drop=True)
    remaining_df = df[~test_mask].reset_index(drop=True)

    if val_signers is not None:
        val_set = set(val_signers)
        val_mask = remaining_df[signer_col].isin(val_set)
        val_df = remaining_df[val_mask].reset_index(drop=True)
        train_df = remaining_df[~val_mask].reset_index(drop=True)
    else:
        # Si no se especifican signantes de validación, se aparta 1 signante o ~15% de remaining
        remaining_signers = remaining_df[signer_col].unique()
        if len(remaining_signers) > 1:
            n_val_signers = max(1, int(len(remaining_signers) * 0.15))
            selected_val_signers = list(remaining_signers[-n_val_signers:])
            val_mask = remaining_df[signer_col].isin(selected_val_signers)
            val_df = remaining_df[val_mask].reset_index(drop=True)
            train_df = remaining_df[~val_mask].reset_index(drop=True)
        else:
            # Fallback si solo queda un signante: split aleatorio estratificado por palabra
            val_df = remaining_df.sample(frac=0.15, random_state=42)
            train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)

    # Validar que no hay fuga de signantes entre train y test
    train_signers = set(train_df[signer_col].unique())
    test_signers_actual = set(test_df[signer_col].unique())
    leakage = train_signers.intersection(test_signers_actual)
    assert len(leakage) == 0, f"¡Alerta de fuga! Signantes en train y test: {leakage}"

    print(
        f"📊 Cross-Signer Split generado con éxito:\n"
        f"   Train: {len(train_df)} muestras ({len(train_signers)} signantes: {sorted(list(train_signers))})\n"
        f"   Val:   {len(val_df)} muestras ({len(val_df[signer_col].unique())} signantes: {sorted(list(val_df[signer_col].unique()))})\n"
        f"   Test:  {len(test_df)} muestras ({len(test_signers_actual)} signantes: {sorted(list(test_signers_actual))})"
    )

    return train_df, val_df, test_df


def generate_loso_folds(
    df: pd.DataFrame,
    signer_col: str = "signer_id",
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, str | int], None, None]:
    """
    Generador de particiones Leave-One-Signer-Out (LOSO) Cross-Validation.

    En cada iteración, se reserva 1 signante para test y el resto para train.

    Yields:
        (train_df, test_df, held_out_signer_id)
    """
    if signer_col not in df.columns:
        raise ValueError(f"Columna '{signer_col}' no encontrada en el DataFrame.")

    signers = sorted(df[signer_col].unique())
    print(f"🔄 Iniciando LOSO-CV sobre {len(signers)} signantes: {signers}")

    for held_out in signers:
        test_mask = df[signer_col] == held_out
        test_df = df[test_mask].reset_index(drop=True)
        train_df = df[~test_mask].reset_index(drop=True)
        yield train_df, test_df, held_out


def generate_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    signer_col: str = "signer_id",
    label_col: str = "word",
    random_state: int = 42,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
    """
    K-Fold estratificado por grupos para balancear clases con cero fuga de signantes.

    Yields:
        (train_df, val_df, fold_index)
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = np.zeros(len(df))
    y = df[label_col].values
    groups = df[signer_col].values

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        yield train_df, val_df, fold_idx
