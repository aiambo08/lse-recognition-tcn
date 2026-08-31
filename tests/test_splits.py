"""
tests/test_splits.py — Tests del particionado Cross-Signer y LOSO-CV
=====================================================================
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lse_recognition.data.splits import (
    create_cross_signer_splits,
    generate_loso_folds,
    generate_stratified_group_folds,
)


@pytest.fixture
def multisigner_df():
    """Genera un DataFrame sintético con 4 signantes y 3 palabras."""
    records = []
    words = ["HOLA", "GRACIAS", "NO"]
    signers = ["S1", "S2", "S3", "S4"]

    for signer in signers:
        for word in words:
            for rep in range(4):
                records.append({
                    "sample_id": f"{word}_{signer}_{rep}",
                    "word": word,
                    "signer_id": signer,
                })
    return pd.DataFrame(records)


class TestCrossSignerSplits:
    def test_zero_signer_leakage(self, multisigner_df):
        """Verifica que ningún signante de test esté presente en train o val."""
        train_df, val_df, test_df = create_cross_signer_splits(
            multisigner_df,
            test_signers=["S4"],
            val_signers=["S3"],
            signer_col="signer_id",
        )

        train_signers = set(train_df["signer_id"].unique())
        val_signers = set(val_df["signer_id"].unique())
        test_signers = set(test_df["signer_id"].unique())

        assert test_signers == {"S4"}
        assert val_signers == {"S3"}
        assert train_signers == {"S1", "S2"}

        # Cero intersección
        assert len(train_signers.intersection(test_signers)) == 0
        assert len(train_signers.intersection(val_signers)) == 0
        assert len(val_signers.intersection(test_signers)) == 0

    def test_total_sample_conservation(self, multisigner_df):
        """La suma de muestras de todos los splits debe ser igual al total original."""
        train_df, val_df, test_df = create_cross_signer_splits(
            multisigner_df,
            test_signers=["S4"],
            val_signers=["S3"],
        )
        total_split = len(train_df) + len(val_df) + len(test_df)
        assert total_split == len(multisigner_df)

    def test_invalid_signer_raises_error(self, multisigner_df):
        """Lanza ValueError si se solicita un signante inexistente."""
        with pytest.raises(ValueError, match="Signantes de test no presentes"):
            create_cross_signer_splits(
                multisigner_df,
                test_signers=["S99_NO_EXISTE"],
            )

    def test_loso_folds_count(self, multisigner_df):
        """LOSO-CV debe generar exactamente tantos folds como signantes únicos."""
        folds = list(generate_loso_folds(multisigner_df, signer_col="signer_id"))
        assert len(folds) == 4

        for train_df, test_df, held_out in folds:
            assert held_out in ["S1", "S2", "S3", "S4"]
            assert set(test_df["signer_id"].unique()) == {held_out}
            assert held_out not in train_df["signer_id"].unique()
            assert len(train_df) + len(test_df) == len(multisigner_df)

    def test_stratified_group_folds_no_leakage(self, multisigner_df):
        """StratifiedGroupKFold debe mantener grupos de signantes disjuntos."""
        folds = list(generate_stratified_group_folds(
            multisigner_df, n_splits=2, signer_col="signer_id", label_col="word"
        ))
        assert len(folds) == 2

        for train_df, val_df, fold_idx in folds:
            train_signers = set(train_df["signer_id"].unique())
            val_signers = set(val_df["signer_id"].unique())
            assert len(train_signers.intersection(val_signers)) == 0
