"""
tests/test_deployment.py — Tests de Integración para Despliegue, ONNX y FastAPI
==============================================================================
"""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from lse_recognition.config import load_config
from lse_recognition.deployment.export import export_to_onnx, export_to_torchscript
from lse_recognition.deployment.onnx_inference import ONNXSignPredictor
from lse_recognition.models.mstcn import MSTCNSignClassifier
from lse_recognition.models.tcn import TCNSignClassifier
from lse_recognition.server.app import app


@pytest.fixture
def sample_config():
    return {
        "seq_length": 40,
        "input_features": 126,
        "num_classes": 5,
        "tcn_channels": [32, 64],
        "kernel_size": 3,
        "dropout": 0.1,
    }


class TestModelExport:
    """Tests para exportación a ONNX y TorchScript."""

    def test_export_tcn_to_onnx(self, tmp_path, sample_config):
        model = TCNSignClassifier(sample_config)
        onnx_file = tmp_path / "test_tcn.onnx"

        exported = export_to_onnx(
            model,
            onnx_file,
            seq_length=sample_config["seq_length"],
            input_features=sample_config["input_features"],
            verify_parity=True,
        )
        assert exported.exists()
        assert exported.stat().st_size > 1000

    def test_export_mstcn_to_onnx(self, tmp_path, sample_config):
        sample_config["ms_kernel_sizes"] = [3, 5]
        model = MSTCNSignClassifier(sample_config)
        onnx_file = tmp_path / "test_mstcn.onnx"

        exported = export_to_onnx(
            model,
            onnx_file,
            seq_length=sample_config["seq_length"],
            input_features=sample_config["input_features"],
            verify_parity=True,
        )
        assert exported.exists()

    def test_export_to_torchscript(self, tmp_path, sample_config):
        model = TCNSignClassifier(sample_config)
        ts_file = tmp_path / "test_tcn.pt"

        exported = export_to_torchscript(
            model,
            ts_file,
            seq_length=sample_config["seq_length"],
            input_features=sample_config["input_features"],
        )
        assert exported.exists()


class TestONNXPredictor:
    """Tests para el motor de inferencia ONNX Runtime."""

    @pytest.fixture
    def onnx_model_path(self, tmp_path, sample_config):
        model = TCNSignClassifier(sample_config)
        onnx_file = tmp_path / "fixture_model.onnx"
        export_to_onnx(
            model,
            onnx_file,
            seq_length=sample_config["seq_length"],
            input_features=sample_config["input_features"],
        )
        return onnx_file

    def test_predictor_single_sequence(self, onnx_model_path, sample_config):
        class_names = [f"sign_{i}" for i in range(sample_config["num_classes"])]
        predictor = ONNXSignPredictor(onnx_model_path, class_names=class_names, temperature=1.0)

        dummy_seq = np.random.randn(sample_config["seq_length"], sample_config["input_features"]).astype(np.float32)
        res = predictor.predict_sequence(dummy_seq, top_k=3)

        assert "predicted_class" in res
        assert "confidence" in res
        assert "latency_ms" in res
        assert len(res["top_k"]) == 3
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["latency_ms"] > 0.0

    def test_predictor_temperature_calibration(self, onnx_model_path, sample_config):
        predictor = ONNXSignPredictor(onnx_model_path, temperature=2.5)
        assert predictor.temperature == 2.5


class TestFastAPIServer:
    """Tests para la API REST y validación de endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "classes_count" in data

    def test_classes_endpoint(self, client):
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert isinstance(data["classes"], list)

    def test_calibrate_endpoint(self, client):
        response = client.post("/calibrate", json={"temperature": 1.25})
        assert response.status_code == 200
        assert response.json()["new_temperature"] == 1.25

    def test_predict_sequence_endpoint(self, client):
        # Generar secuencia válida de 40 x 126
        dummy_seq = [[0.0] * 126 for _ in range(40)]
        response = client.post(
            "/predict/sequence",
            json={"sequence": dummy_seq, "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert len(data["top_k"]) == 3
