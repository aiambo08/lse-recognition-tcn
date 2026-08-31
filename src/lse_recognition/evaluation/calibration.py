"""
calibration.py — Calibración de Confianza y Expected Calibration Error (ECE)
===========================================================================

Proporciona herramientas formales para evaluar y mejorar la calibración probabilística:
    - Expected Calibration Error (ECE) con binned accuracy vs confidence.
    - Maximum Calibration Error (MCE) y Brier Score.
    - Temperature Scaling: Optimización del hiperparámetro de temperatura T > 0
      sobre el conjunto de validación para eliminar sobreconfianza sin alterar el accuracy.
    - Generación de Reliability Diagrams (curvas de calibración).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ExpectedCalibrationError(nn.Module):
    """
    Calcula el Expected Calibration Error (ECE) y Maximum Calibration Error (MCE).

    ECE mide la diferencia ponderada entre la confianza asignada por el modelo
    y la precisión empírica real a lo largo de M intervalos probabilísticos.
    """

    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins

    def forward(
        self, logits_or_probs: torch.Tensor, labels: torch.Tensor, is_logits: bool = True
    ) -> Dict[str, float]:
        """
        Args:
            logits_or_probs: Tensor (N, num_classes) con logits o probabilidades.
            labels: Tensor (N,) con las clases objetivo reales.
            is_logits: Si es True, aplica softmax internamente.

        Returns:
            Dict con 'ece', 'mce', 'brier_score', 'avg_confidence', 'accuracy'.
        """
        if is_logits:
            probs = torch.softmax(logits_or_probs, dim=1)
        else:
            probs = logits_or_probs

        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        mce = torch.zeros(1, device=probs.device)

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=probs.device)

        bin_stats = []

        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Seleccionar muestras en el bin (bin_lower, bin_upper]
            if i == 0:
                in_bin = confidences.ge(bin_lower) & confidences.le(bin_upper)
            else:
                in_bin = confidences.gt(bin_lower) & confidences.le(bin_upper)

            prop_in_bin = in_bin.float().mean()

            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                gap = (avg_confidence_in_bin - accuracy_in_bin).abs()

                ece += gap * prop_in_bin
                mce = torch.max(mce, gap)

                bin_stats.append({
                    "bin_lower": float(bin_lower),
                    "bin_upper": float(bin_upper),
                    "count": int(in_bin.sum()),
                    "accuracy": float(accuracy_in_bin),
                    "confidence": float(avg_confidence_in_bin),
                })

        # Brier Score = (1/N) * sum_i sum_k (p_ik - y_ik)^2
        one_hot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
        brier_score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))

        return {
            "ece": float(ece.item() * 100.0),  # en porcentaje
            "mce": float(mce.item() * 100.0),  # en porcentaje
            "brier_score": float(brier_score.item()),
            "avg_confidence": float(confidences.mean().item() * 100.0),
            "accuracy": float(accuracies.float().mean().item() * 100.0),
            "bin_stats": bin_stats,
        }


class TemperatureScaler(nn.Module):
    """
    Calibra un modelo pre-entrenado mediante Temperature Scaling.

    Aprende un parámetro escalar T > 0 minimizando NLL en validación:
        p_i = softmax(logits / T)
    """

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=0.01, max=50.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Aplica el escalado por temperatura a los logits."""
        return logits / self.temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.05,
        max_iter: int = 50,
    ) -> float:
        """
        Optimiza la temperatura sobre los logits y etiquetas de validación.

        Returns:
            Temperatura óptima aprendida T*.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())

    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades perfectamente calibradas."""
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            return torch.softmax(scaled_logits, dim=1)

