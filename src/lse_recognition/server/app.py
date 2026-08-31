"""
server/app.py — Servidor FastAPI de Inferencia y Streaming en Tiempo Real (WebSocket/REST)
========================================================================================

Proporciona una API de nivel de producción para el reconocimiento de Lengua de Signos Española:
    - Endpoints REST para inferencia en batch, consulta de vocabulario y calibración dinámica.
    - Endpoint WebSocket para streaming interactivo de landmarks a 30-60 FPS con ventana deslizante.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lse_recognition.config import load_config
from lse_recognition.data import LandmarksNormalizer
from lse_recognition.deployment.export import export_to_onnx
from lse_recognition.deployment.onnx_inference import ONNXSignPredictor
from lse_recognition.models import create_model

logger = logging.getLogger("lse_api")
logging.basicConfig(level=logging.INFO)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load_or_initialize()
    yield

app = FastAPI(
    title="LSE Real-Time Recognition API",
    description="API de Producción para Reconocimiento de Lengua de Signos Española en Tiempo Real",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Esquemas Pydantic
# -----------------------------------------------------------------------------
class SequencePayload(BaseModel):
    sequence: List[List[float]] = Field(
        ..., description="Secuencia temporal de landmarks (T x 126)"
    )
    top_k: Optional[int] = Field(3, description="Número de predicciones top-k")


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    latency_ms: float
    top_k: List[Dict[str, Any]]


class TemperaturePayload(BaseModel):
    temperature: float = Field(..., gt=0.0, description="Temperatura de calibración T > 0")


# -----------------------------------------------------------------------------
# Estado Global del Modelo
# -----------------------------------------------------------------------------
class ModelServerState:
    def __init__(self):
        self.predictor: Optional[ONNXSignPredictor] = None
        self.normalizer = LandmarksNormalizer()
        self.class_names: List[str] = []
        self.config: Dict[str, Any] = {}
        self.seq_length: int = 40
        self.input_features: int = 126

    def ensure_initialized(self):
        if self.predictor is None:
            self.load_or_initialize()

    def load_or_initialize(self, onnx_path: Optional[str] = None):
        self.config = load_config()
        self.seq_length = self.config.get("seq_length", 40)
        self.input_features = self.config.get("input_features", 126)

        # Cargar vocabulario si existe
        manifest_path = Path("data/metadata/dilse_manifest.csv")
        if manifest_path.exists():
            import pandas as pd
            df = pd.read_csv(manifest_path)
            col = "label" if "label" in df.columns else "word"
            self.class_names = sorted(df[col].dropna().unique().tolist())
        else:
            self.class_names = [f"sign_{i}" for i in range(self.config.get("num_classes", 10))]

        # Resolver modelo ONNX
        default_onnx = Path("models/lse_model.onnx")
        if onnx_path and Path(onnx_path).exists():
            target_path = Path(onnx_path)
        elif default_onnx.exists():
            target_path = default_onnx
        else:
            # Exportar modelo base por defecto
            logger.info("Generando modelo ONNX inicial...")
            model = create_model(self.config, model_type="mstcn", device="cpu")
            export_to_onnx(
                model,
                default_onnx,
                seq_length=self.seq_length,
                input_features=self.input_features,
            )
            target_path = default_onnx

        self.predictor = ONNXSignPredictor(
            model_path=target_path,
            class_names=self.class_names,
            temperature=1.0,
        )
        logger.info(f"🚀 Servidor listo con {len(self.class_names)} clases léxicas.")


state = ModelServerState()


# -----------------------------------------------------------------------------
# Endpoints REST
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    state.ensure_initialized()
    return {
        "status": "healthy",
        "service": "LSE-Recognition-API",
        "version": "2.0.0",
        "classes_count": len(state.class_names),
        "temperature": state.predictor.temperature if state.predictor else 1.0,
        "input_shape": [1, state.seq_length, state.input_features],
    }


@app.get("/classes")
def get_classes():
    state.ensure_initialized()
    return {
        "count": len(state.class_names),
        "classes": state.class_names,
    }


@app.post("/predict/sequence", response_model=PredictionResponse)
def predict_sequence(payload: SequencePayload):
    state.ensure_initialized()
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    seq_arr = np.array(payload.sequence, dtype=np.float32)
    if seq_arr.shape[-1] != state.input_features:
        raise HTTPException(
            status_code=400,
            detail=f"Dimensión incorrecta: esperado (*, {state.input_features}), recibido {seq_arr.shape}",
        )

    # Normalizar si no viene normalizado
    if seq_arr.ndim == 2 and seq_arr.shape[0] != state.seq_length:
        # Interpolación temporal a seq_length
        indices = np.linspace(0, seq_arr.shape[0] - 1, state.seq_length).astype(int)
        seq_arr = seq_arr[indices]

    res = state.predictor.predict_sequence(seq_arr, top_k=payload.top_k or 3)
    return PredictionResponse(
        predicted_class=res["predicted_class"],
        confidence=res["confidence"],
        latency_ms=res["latency_ms"],
        top_k=res["top_k"],
    )


@app.post("/calibrate")
def set_temperature(payload: TemperaturePayload):
    state.ensure_initialized()
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    state.predictor.temperature = payload.temperature
    return {"message": "Temperatura actualizada", "new_temperature": payload.temperature}


# -----------------------------------------------------------------------------
# Endpoint WebSocket para Streaming en Tiempo Real
# -----------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Cliente WebSocket conectado para streaming de LSE.")

    buffer = deque(maxlen=state.seq_length)
    cooldown_counter = 0
    cooldown_frames = 15
    confidence_threshold = 0.70

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)

            # Espera landmarks del frame: lista de 126 floats
            landmarks = data.get("landmarks", None)
            if landmarks is None or len(landmarks) != state.input_features:
                await websocket.send_text(
                    json.dumps({"error": f"Formato inválido: se esperaban {state.input_features} floats"})
                )
                continue

            buffer.append(landmarks)

            if len(buffer) < state.seq_length:
                await websocket.send_text(
                    json.dumps({
                        "status": "buffering",
                        "buffer_size": len(buffer),
                        "buffer_target": state.seq_length,
                    })
                )
                continue

            if cooldown_counter > 0:
                cooldown_counter -= 1
                continue

            # Realizar predicción sobre la ventana deslizante
            seq_array = np.array(buffer, dtype=np.float32)
            pred_res = state.predictor.predict_sequence(seq_array, top_k=3)

            is_new_detection = False
            if pred_res["confidence"] >= confidence_threshold:
                is_new_detection = True
                cooldown_counter = cooldown_frames

            response_payload = {
                "status": "active",
                "detected_sign": pred_res["predicted_class"],
                "confidence": pred_res["confidence"],
                "is_new_detection": is_new_detection,
                "top_k": pred_res["top_k"],
                "latency_ms": pred_res["latency_ms"],
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(response_payload))

    except WebSocketDisconnect:
        logger.info("Cliente WebSocket desconectado.")
    except Exception as e:
        logger.error(f"Error en WebSocket stream: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
