"""
metrics.py — Evaluación y visualización de resultados
======================================================

Contiene:
    - evaluate_model        : Evaluación completa en test set
    - plot_training_history : Curvas de aprendizaje desde historial JSON
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from lse_recognition.models.tcn import TCNSignClassifier
from lse_recognition.models.lstm import LSTMSignClassifier


def _detect_model_class(config: Dict) -> Type[nn.Module]:
    """Detecta automáticamente la clase del modelo por las claves del config."""
    if "tcn_channels" in config:
        return TCNSignClassifier
    elif "lstm_hidden_size" in config:
        return LSTMSignClassifier
    else:
        raise ValueError(
            "No se pudo detectar el tipo de modelo. "
            "Especifica model_class manualmente."
        )


def evaluate_model(
    model_path: str | Path,
    test_loader: DataLoader,
    idx_to_word: Dict[int, str],
    device: Optional[torch.device] = None,
    model_class: Optional[Type[nn.Module]] = None,
    show_plots: bool = True,
) -> Tuple[float, float]:
    """
    Evalúa un checkpoint guardado en el conjunto de test.

    Args:
        model_path:   Ruta al checkpoint .pth.
        test_loader:  DataLoader del conjunto de test.
        idx_to_word:  Mapeo índice → palabra.
        device:       Dispositivo PyTorch. Si None, auto-detecta CUDA.
        model_class:  Clase del modelo. Si None, se detecta automáticamente.
        show_plots:   Si True, muestra matrices de confusión.

    Returns:
        (accuracy, f1_macro)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]

    if model_class is None:
        model_class = _detect_model_class(config)

    model = model_class(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\n{'='*60}")
    print("EVALUACIÓN EN TEST SET")
    print(f"{'='*60}")
    print(f"Modelo:           {model_class.__name__}")
    print(f"Checkpoint:       {model_path}")
    print(f"Epoch guardado:   {checkpoint.get('epoch', 'N/A')}")
    val_f1 = checkpoint.get("val_f1", 0.0)
    print(f"Val F1 en ese epoch: {val_f1:.4f}")
    print(f"{'='*60}\n")

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Métricas globales
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"{'='*60}")
    print("RESULTADOS")
    print(f"{'='*60}")
    print(f"✅ Test Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
    print(f"✅ F1 Macro:          {f1:.4f}")
    print(f"✅ F1 Weighted:       {f1_weighted:.4f}")
    print(f"{'='*60}\n")

    # Reporte por clase
    target_names = [idx_to_word[i] for i in range(len(idx_to_word))]
    print(f"{'='*60}")
    print("REPORTE POR CLASE")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    # Análisis de errores
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n{'='*60}")
    print("⚠️  CLASES CON MÁS ERRORES")
    print(f"{'='*60}")
    errors_per_class = {}
    for i in range(len(idx_to_word)):
        total = cm[i].sum()
        correct = cm[i, i]
        errors = total - correct
        if total > 0:
            errors_per_class[idx_to_word[i]] = (errors, total, errors / total)

    for word, (errors, total, rate) in sorted(
        errors_per_class.items(), key=lambda x: x[1][2], reverse=True
    )[:5]:
        print(f"{word:15s}: {errors}/{total} errores ({rate*100:.1f}%)")

    # Matrices de confusión
    if show_plots:
        _plot_confusion_matrices(cm, target_names, model_class.__name__, acc)

    return acc, f1


def _plot_confusion_matrices(
    cm: np.ndarray, target_names: list, model_name: str, acc: float
) -> None:
    """Dibuja matrices de confusión absoluta y normalizada."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Absoluta
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
        cbar_kws={"label": "Número de muestras"},
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicción", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Verdadero", fontsize=12, fontweight="bold")
    axes[0].set_title(
        f"Matriz de Confusión — {model_name}\nAccuracy: {acc:.4f}",
        fontsize=13, fontweight="bold"
    )

    # Normalizada
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="RdYlGn_r",
        xticklabels=target_names, yticklabels=target_names,
        vmin=0, vmax=1,
        cbar_kws={"label": "Tasa de error"},
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicción", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Verdadero", fontsize=12, fontweight="bold")
    axes[1].set_title(
        f"Matriz de Confusión Normalizada — {model_name}",
        fontsize=13, fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


def plot_training_history(history: Dict) -> None:
    """
    Dibuja curvas de aprendizaje a partir del diccionario de historial.

    Args:
        history: Diccionario con claves train_loss, val_loss, train_acc,
                 val_acc, val_f1, learning_rates.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train acc")
    axes[1].plot(epochs, history["val_acc"], label="Val acc")
    axes[1].plot(epochs, history["val_f1"], label="Val F1", linestyle="--")
    axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Learning Rate
    axes[2].plot(epochs, history["learning_rates"], color="orange")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")

    plt.suptitle("Historial de entrenamiento", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
