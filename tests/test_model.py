"""
tests/test_model.py — Smoke tests del modelo TCN y LSTM
=========================================================
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lse_recognition.models.tcn import TCNSignClassifier, TCNResidualBlock, create_model
from lse_recognition.models.lstm import LSTMSignClassifier


# ------------------------------------------------------------------ #
# Configuración mínima para tests                                     #
# ------------------------------------------------------------------ #

TCN_CONFIG = {
    "seq_length": 60,
    "input_features": 126,
    "num_classes": 10,
    "projection_dim": 64,     # reducido para que los tests sean rápidos
    "tcn_channels": [64, 64, 128],
    "kernel_size": 3,
    "tcn_dropout": 0.1,
    "fc_hidden_dim": 64,
    "fc_dropout": 0.1,
    # Para Trainer (no usado en model, pero presente en config completo)
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 5,
    "early_stopping_patience": 3,
    "lr_scheduler_patience": 2,
    "lr_scheduler_factor": 0.5,
    "confidence_threshold_high": 0.75,
    "confidence_threshold_low": 0.5,
    "cooldown_duration": 3.0,
    "stabilization_votes": 3,
    "min_frames_to_predict": 30,
}

LSTM_CONFIG = {
    **TCN_CONFIG,
    "lstm_hidden_size": 128,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.3,
}


# ------------------------------------------------------------------ #
# TCNResidualBlock                                                    #
# ------------------------------------------------------------------ #

class TestTCNResidualBlock:
    def test_output_shape_same_channels(self):
        """Mismo número de canales: shape de salida igual al de entrada."""
        block = TCNResidualBlock(
            in_channels=64, out_channels=64,
            kernel_size=3, dilation=1, dropout=0.0
        )
        x = torch.randn(2, 64, 60)  # (batch, channels, seq_len)
        out = block(x)
        assert out.shape == (2, 64, 60), f"Shape incorrecto: {out.shape}"

    def test_output_shape_different_channels(self):
        """Canales distintos: se aplica la proyección residual."""
        block = TCNResidualBlock(
            in_channels=32, out_channels=64,
            kernel_size=3, dilation=2, dropout=0.0
        )
        x = torch.randn(2, 32, 60)
        out = block(x)
        assert out.shape == (2, 64, 60), f"Shape incorrecto: {out.shape}"

    def test_residual_projection_exists_when_needed(self):
        """residual_projection debe existir solo cuando cambian los canales."""
        block_same = TCNResidualBlock(64, 64, 3, 1, 0.0)
        block_diff = TCNResidualBlock(32, 64, 3, 1, 0.0)

        assert block_same.residual_projection is None
        assert block_diff.residual_projection is not None


# ------------------------------------------------------------------ #
# TCNSignClassifier                                                   #
# ------------------------------------------------------------------ #

class TestTCNSignClassifier:
    def test_forward_output_shape(self):
        """Forward pass debe devolver (batch, num_classes)."""
        model = TCNSignClassifier(TCN_CONFIG)
        model.eval()
        x = torch.randn(4, TCN_CONFIG["seq_length"], TCN_CONFIG["input_features"])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, TCN_CONFIG["num_classes"])

    def test_predict_proba_sums_to_one(self):
        """predict_proba debe devolver probabilidades que suman 1."""
        model = TCNSignClassifier(TCN_CONFIG)
        model.eval()
        x = torch.randn(2, TCN_CONFIG["seq_length"], TCN_CONFIG["input_features"])
        probs = model.predict_proba(x)
        assert probs.shape == (2, TCN_CONFIG["num_classes"])
        torch.testing.assert_close(
            probs.sum(dim=1), torch.ones(2), atol=1e-5, rtol=0
        )

    def test_receptive_field_calculation(self):
        """
        El campo receptivo para dilaciones [1,2,4] con k=3 debe ser:
        RF = 1 + 2*(1*2 + 2*2 + 4*2) = 1 + 2*(2+4+8) = 1 + 28 = 29
        """
        model = TCNSignClassifier(TCN_CONFIG)
        rf = model.get_receptive_field()
        expected = 1 + sum(2 * (2 ** i) * 2 for i in range(len(TCN_CONFIG["tcn_channels"])))
        assert rf == expected, f"RF incorrecto: {rf} (esperado {expected})"

    def test_no_nan_in_output(self):
        """El modelo no debe producir NaN en la salida."""
        model = TCNSignClassifier(TCN_CONFIG)
        model.eval()
        x = torch.randn(4, TCN_CONFIG["seq_length"], TCN_CONFIG["input_features"])
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "La salida contiene NaN"

    def test_single_sample_batch(self):
        """El modelo debe funcionar con batch_size=1."""
        model = TCNSignClassifier(TCN_CONFIG)
        model.eval()
        x = torch.randn(1, TCN_CONFIG["seq_length"], TCN_CONFIG["input_features"])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, TCN_CONFIG["num_classes"])


# ------------------------------------------------------------------ #
# LSTMSignClassifier                                                  #
# ------------------------------------------------------------------ #

class TestLSTMSignClassifier:
    def test_forward_output_shape(self):
        """Forward pass LSTM debe devolver (batch, num_classes)."""
        model = LSTMSignClassifier(LSTM_CONFIG)
        model.eval()
        x = torch.randn(4, LSTM_CONFIG["seq_length"], LSTM_CONFIG["input_features"])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, LSTM_CONFIG["num_classes"])

    def test_predict_proba_sums_to_one(self):
        """predict_proba LSTM debe devolver probabilidades que suman 1."""
        model = LSTMSignClassifier(LSTM_CONFIG)
        model.eval()
        x = torch.randn(2, LSTM_CONFIG["seq_length"], LSTM_CONFIG["input_features"])
        probs = model.predict_proba(x)
        torch.testing.assert_close(
            probs.sum(dim=1), torch.ones(2), atol=1e-5, rtol=0
        )


# ------------------------------------------------------------------ #
# create_model                                                        #
# ------------------------------------------------------------------ #

class TestCreateModel:
    def test_create_model_returns_tcn(self):
        """create_model debe devolver una instancia de TCNSignClassifier."""
        device = torch.device("cpu")
        model = create_model(TCN_CONFIG, device=device)
        assert isinstance(model, TCNSignClassifier)

    def test_model_on_correct_device(self):
        """El modelo debe estar en el dispositivo especificado."""
        device = torch.device("cpu")
        model = create_model(TCN_CONFIG, device=device)
        param = next(model.parameters())
        assert param.device.type == "cpu"
