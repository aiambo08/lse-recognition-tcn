"""
tests/test_advanced_models.py — Tests unitarios para MS-TCN, Attention-BiLSTM y ST-GCN
======================================================================================
"""

import pytest
import torch

from lse_recognition.models import (
    AttentionLSTMSignClassifier,
    MSTCNSignClassifier,
    STGCNSignClassifier,
    build_hand_adjacency_matrix,
    create_model,
)

CONFIG_BASE = {
    "input_features": 126,
    "num_classes": 10,
    "seq_length": 40,
    "projection_dim": 64,
    "tcn_channels": [64, 64, 128],
    "ms_kernel_sizes": [3, 5],
    "tcn_dropout": 0.1,
    "fc_hidden_dim": 64,
    "fc_dropout": 0.1,
    "lstm_hidden_dim": 64,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.1,
    "lstm_bidirectional": True,
    "attention_dim": 32,
    "stgcn_hidden_dim": 32,
    "stgcn_dropout": 0.1,
    "num_nodes": 42,
}


class TestMSTCN:
    """Tests para Multi-Scale TCN."""

    def test_output_shape(self):
        model = MSTCNSignClassifier(CONFIG_BASE)
        x = torch.randn(4, 40, 126)
        out = model(x)
        assert out.shape == (4, 10)

    def test_single_sample_batch(self):
        model = MSTCNSignClassifier(CONFIG_BASE)
        x = torch.randn(1, 40, 126)
        out = model(x)
        assert out.shape == (1, 10)

    def test_predict_proba_sums_to_one(self):
        model = MSTCNSignClassifier(CONFIG_BASE)
        x = torch.randn(2, 40, 126)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_backward_pass(self):
        model = MSTCNSignClassifier(CONFIG_BASE)
        x = torch.randn(2, 40, 126)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestAttentionLSTM:
    """Tests para BiLSTM con Temporal Attention."""

    def test_output_shape(self):
        model = AttentionLSTMSignClassifier(CONFIG_BASE)
        x = torch.randn(4, 40, 126)
        out = model(x)
        assert out.shape == (4, 10)

    def test_attention_weights_return(self):
        model = AttentionLSTMSignClassifier(CONFIG_BASE)
        x = torch.randn(3, 40, 126)
        logits, weights = model(x, return_attention=True)
        assert logits.shape == (3, 10)
        assert weights.shape == (3, 40)
        assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_predict_proba(self):
        model = AttentionLSTMSignClassifier(CONFIG_BASE)
        x = torch.randn(2, 40, 126)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


class TestSTGCN:
    """Tests para Spatial-Temporal Graph Convolutional Network."""

    def test_adjacency_matrix(self):
        A = build_hand_adjacency_matrix(42)
        assert A.shape == (42, 42)
        assert not torch.isnan(A).any()
        assert not torch.isinf(A).any()

    def test_output_shape(self):
        model = STGCNSignClassifier(CONFIG_BASE)
        x = torch.randn(4, 40, 126)
        out = model(x)
        assert out.shape == (4, 10)

    def test_predict_proba(self):
        model = STGCNSignClassifier(CONFIG_BASE)
        x = torch.randn(2, 40, 126)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


class TestModelFactory:
    """Tests para la factory unificada create_model."""

    @pytest.mark.parametrize("model_type", ["tcn", "mstcn", "lstm", "attention_lstm", "stgcn"])
    def test_create_all_model_types(self, model_type):
        device = torch.device("cpu")
        model = create_model(CONFIG_BASE, model_type=model_type, device=device)
        assert model is not None
        x = torch.randn(2, 40, 126)
        out = model(x)
        assert out.shape == (2, 10)

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Tipo de modelo desconocido"):
            create_model(CONFIG_BASE, model_type="non_existent_transformer")
