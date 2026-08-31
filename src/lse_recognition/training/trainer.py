"""
trainer.py — Pipeline completo de entrenamiento
================================================

Contiene:
    - Trainer       : Clase que encapsula el ciclo de entrenamiento
    - run_training  : Función de alto nivel que orquesta el pipeline completo

El Trainer implementa:
    - Optimizador: Adam (lr=1e-3, weight_decay=1e-4)
    - Loss: CrossEntropyLoss
    - Scheduler: ReduceLROnPlateau (factor=0.5, patience=6)
    - Early Stopping (patience=15, métrica: F1 macro en validación)
    - Guardado automático del mejor checkpoint

Formato del checkpoint (models/best_model.pth):
    {
        'epoch': int,
        'model_state_dict': dict,
        'optimizer_state_dict': dict,
        'val_f1': float,
        'config': dict,
    }
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from lse_recognition.data.dataset import create_dataloaders
from lse_recognition.models.tcn import TCNSignClassifier, create_model


def _set_seed(seed: int) -> None:
    """Fija semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Encapsula el ciclo de entrenamiento para modelos de clasificación de secuencias.

    Args:
        model:        Modelo PyTorch.
        train_loader: DataLoader de entrenamiento.
        val_loader:   DataLoader de validación.
        config:       Diccionario de configuración.
        device:       Dispositivo PyTorch.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config["lr_scheduler_factor"],
            patience=config["lr_scheduler_patience"],
        )

        self.history: Dict = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "learning_rates": [],
        }
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.best_model_path = str(
            Path(config.get("model_dir", "models")) / "best_model.pth"
        )

    # ------------------------------------------------------------------ #
    # Entrenamiento / validación por epoch                                 #
    # ------------------------------------------------------------------ #

    def train_epoch(self) -> Tuple[float, float]:
        """Ejecuta un epoch de entrenamiento. Devuelve (loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float, float]:
        """Evalúa en validación. Devuelve (loss, accuracy, f1_macro)."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return epoch_loss, epoch_acc, epoch_f1

    # ------------------------------------------------------------------ #
    # Bucle principal                                                      #
    # ------------------------------------------------------------------ #

    def train(self, num_epochs: int) -> Dict:
        """
        Ejecuta el bucle de entrenamiento completo con early stopping.

        Args:
            num_epochs: Número máximo de epochs.

        Returns:
            Historial de métricas.
        """
        print("\n" + "=" * 60)
        print("Iniciando entrenamiento")
        print("=" * 60)

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Guardar historial
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["learning_rates"].append(current_lr)

            print(
                f"Epoch {epoch:3d}: "
                f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
                f"Val loss {val_loss:.4f}, acc {val_acc:.4f}, F1 {val_f1:.4f}"
            )

            # Guardar mejor modelo
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                Path(self.best_model_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_f1": val_f1,
                        "config": self.config,
                    },
                    self.best_model_path,
                )
                print(f"  ✅ Mejor modelo actualizado (F1={val_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config["early_stopping_patience"]:
                    print(f"  ⏹️  Early stopping en epoch {epoch}")
                    break

        print("=" * 60)
        print("✅ Entrenamiento completado")
        print("=" * 60)
        return self.history

    # ------------------------------------------------------------------ #
    # Visualización                                                        #
    # ------------------------------------------------------------------ #

    def plot_history(self) -> None:
        """Dibuja curvas de pérdida y métricas."""
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], label="Train loss")
        axes[0].plot(epochs, self.history["val_loss"], label="Val loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Curvas de pérdida")
        axes[0].legend()

        # Accuracy / F1
        axes[1].plot(epochs, self.history["train_acc"], label="Train acc")
        axes[1].plot(epochs, self.history["val_acc"], label="Val acc")
        axes[1].plot(epochs, self.history["val_f1"], label="Val F1", linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Métrica")
        axes[1].set_title("Accuracy y F1")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------ #
# Pipeline de alto nivel                                              #
# ------------------------------------------------------------------ #

def run_training(
    config: Dict,
    use_hands_only: bool = True,
    normalize: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """
    Pipeline completo: DataLoaders → Modelo → Entrenamiento → Historia.

    Args:
        config:         Diccionario de configuración (se modifica in-place con
                        num_classes e input_features detectados automáticamente).
        use_hands_only: Usar solo landmarks de manos (126 features).
        normalize:      Aplicar normalización geométrica.
        seed:           Semilla aleatoria. Si None, usa config.get('seed', 42).
        device:         Dispositivo PyTorch. Si None, auto-detecta CUDA.

    Returns:
        (trainer, history, test_loader, idx_to_word)
    """
    # Reproducibilidad
    _seed = seed if seed is not None else config.get("seed", 42)
    _set_seed(_seed)

    # Dispositivo
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # DataLoaders
    train_loader, val_loader, test_loader, word_to_idx, idx_to_word = (
        create_dataloaders(config, use_hands_only=use_hands_only, normalize=normalize)
    )

    # Modelo
    model = create_model(config, device=device)

    # Entrenamiento
    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train(config["epochs"])
    trainer.plot_history()

    # Guardar historial
    model_dir = Path(config.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return trainer, history, test_loader, idx_to_word
