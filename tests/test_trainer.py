"""
tests/test_trainer.py — Smoke test del ciclo de entrenamiento
=============================================================

Usa un dataset sintético mínimo (2 clases, 10 muestras) para verificar
que el pipeline de entrenamiento corre sin errores.
"""
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lse_recognition.data.dataset import SignLanguageDataset
from lse_recognition.models.tcn import TCNSignClassifier
from lse_recognition.training.trainer import Trainer


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def synthetic_dataloaders(tmp_path_factory):
    """
    Crea un dataset sintético temporal con 2 clases y 10 muestras por clase.
    Devuelve (train_loader, val_loader, word_to_idx, config).
    """
    tmp = tmp_path_factory.mktemp("data")
    words = ["HOLA", "ADIOS"]
    records = []

    for word in words:
        for i in range(10):
            lm = np.random.rand(60, 42, 3).astype(np.float32)
            npy_path = tmp / f"{word}_{i}.npy"
            np.save(npy_path, lm)
            records.append({
                "word": word,
                "landmarks_hands_only_file": str(npy_path),
            })

    csv_path = tmp / "meta.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    word_to_idx = {"HOLA": 0, "ADIOS": 1}

    config = {
        "seq_length": 60,
        "input_features": 126,
        "num_classes": 2,
        "projection_dim": 32,
        "tcn_channels": [32, 32],
        "kernel_size": 3,
        "tcn_dropout": 0.1,
        "fc_hidden_dim": 32,
        "fc_dropout": 0.1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 3,
        "early_stopping_patience": 2,
        "lr_scheduler_patience": 1,
        "lr_scheduler_factor": 0.5,
        "seed": 42,
        "model_dir": str(tmp / "models"),
    }

    ds = SignLanguageDataset(
        str(csv_path), seq_length=60, word_to_idx=word_to_idx
    )
    # 80/20 split
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    return train_loader, val_loader, word_to_idx, config


# ------------------------------------------------------------------ #
# Tests                                                               #
# ------------------------------------------------------------------ #

class TestTrainer:
    def test_trainer_runs_without_error(self, synthetic_dataloaders):
        """El Trainer debe ejecutar epochs sin lanzar excepciones."""
        train_loader, val_loader, _, config = synthetic_dataloaders
        device = torch.device("cpu")
        model = TCNSignClassifier(config).to(device)

        trainer = Trainer(model, train_loader, val_loader, config, device)
        history = trainer.train(num_epochs=2)

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert len(history["val_f1"]) == 2

    def test_history_values_are_finite(self, synthetic_dataloaders):
        """Las métricas del historial no deben contener NaN o Inf."""
        train_loader, val_loader, _, config = synthetic_dataloaders
        device = torch.device("cpu")
        model = TCNSignClassifier(config).to(device)

        trainer = Trainer(model, train_loader, val_loader, config, device)
        history = trainer.train(num_epochs=2)

        import math
        for key, values in history.items():
            for v in values:
                assert math.isfinite(v), f"Valor no finito en history['{key}']: {v}"

    def test_best_model_checkpoint_saved(self, synthetic_dataloaders):
        """El Trainer debe guardar el checkpoint del mejor modelo."""
        train_loader, val_loader, _, config = synthetic_dataloaders
        device = torch.device("cpu")
        model = TCNSignClassifier(config).to(device)

        trainer = Trainer(model, train_loader, val_loader, config, device)
        trainer.train(num_epochs=2)

        checkpoint_path = Path(config["model_dir"]) / "best_model.pth"
        assert checkpoint_path.exists(), "Checkpoint no fue creado"

        # Verificar que el checkpoint es válido
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        assert "model_state_dict" in ckpt
        assert "val_f1" in ckpt
        assert "config" in ckpt

    def test_validate_returns_correct_tuple(self, synthetic_dataloaders):
        """validate() debe devolver una tupla de 3 floats."""
        train_loader, val_loader, _, config = synthetic_dataloaders
        device = torch.device("cpu")
        model = TCNSignClassifier(config).to(device)

        trainer = Trainer(model, train_loader, val_loader, config, device)
        val_loss, val_acc, val_f1 = trainer.validate()

        assert 0.0 <= val_acc <= 1.0, f"Accuracy fuera de rango: {val_acc}"
        assert 0.0 <= val_f1 <= 1.0, f"F1 fuera de rango: {val_f1}"
        assert val_loss >= 0.0, f"Loss negativa: {val_loss}"
