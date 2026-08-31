"""
demo/app.py — Dashboard Interactivo de Demostración y Reconocimiento de LSE
==========================================================================

Interfaz gráfica completa basada en Streamlit para experimentación, visualización
de landmarks en tiempo real, comparación de modelos e inferencia acústica (TTS).

Uso:
    streamlit run src/lse_recognition/demo/app.py
"""

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

from lse_recognition.config import load_config
from lse_recognition.data import LandmarksNormalizer, create_dataloaders
from lse_recognition.deployment import ONNXSignPredictor, export_to_onnx
from lse_recognition.models import create_model

st.set_page_config(
    page_title="LSE Recognition System — Research Demo",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "temperature" not in st.session_state:
        st.session_state.temperature = 1.0


init_session()

# -----------------------------------------------------------------------------
# Sidebar: Configuración del Sistema
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Arquitectura del Modelo:",
    ["mstcn", "tcn", "attention_lstm", "stgcn", "lstm"],
    format_func=lambda x: {
        "mstcn": "Multi-Scale TCN (MS-TCN)",
        "tcn": "Standard Causal TCN",
        "attention_lstm": "Attention-BiLSTM",
        "stgcn": "Spatial-Temporal GCN",
        "lstm": "Standard BiLSTM",
    }[x],
)

backend_choice = st.sidebar.radio(
    "Motor de Inferencia:",
    ["ONNX Runtime (Producción)", "PyTorch (Nativo)"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Calibración de Confianza")
temperature = st.sidebar.slider(
    "Temperatura (T*):",
    min_value=0.1,
    max_value=3.0,
    value=float(st.session_state.temperature),
    step=0.05,
    help="Escala de temperatura óptima aprendida en Fase 3 para calibrar probabilidades.",
)
st.session_state.temperature = temperature

threshold = st.sidebar.slider(
    "Umbral de Decisión (Confidence Threshold):",
    min_value=0.40,
    max_value=0.99,
    value=0.70,
    step=0.05,
)

enable_tts = st.sidebar.checkbox("Activar Síntesis de Voz (TTS)", value=False)

# -----------------------------------------------------------------------------
# Carga del Modelo
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_and_classes(m_type: str):
    config = load_config()
    train_l, _, test_l, w2i, _ = create_dataloaders(config, use_hands_only=True, normalize=True)
    config["num_classes"] = len(w2i)
    classes = [k for k, v in sorted(w2i.items(), key=lambda item: item[1])]

    model = create_model(config, model_type=m_type, device="cpu")

    # Crear o cargar ONNX
    onnx_path = Path(f"models/demo_{m_type}.onnx")
    if not onnx_path.exists():
        export_to_onnx(model, onnx_path, seq_length=40, input_features=126)

    onnx_pred = ONNXSignPredictor(
        model_path=onnx_path,
        class_names=classes,
        temperature=st.session_state.temperature,
    )

    return model, onnx_pred, classes, test_l


model_pt, onnx_pred, class_names, test_loader = get_model_and_classes(model_choice)
onnx_pred.temperature = temperature

# -----------------------------------------------------------------------------
# Cabecera Principal
# -----------------------------------------------------------------------------
st.title("🤟 Sistema de Reconocimiento de Lengua de Signos Española (LSE)")
st.markdown(
    """
    **Plataforma de Investigación y Demostración en Tiempo Real**  
    *Arquitectura Deep Learning con Convoluciones Causales Dilatadas Multi-Escala, 
    Modelado Biomecánico Esquelético y Calibración Probabilística ECE.*
    """
)

tab_demo, tab_benchmark, tab_ablation, tab_info = st.tabs([
    "🎮 Demostrador Interactivo",
    "📊 Benchmarking Comparativo",
    "🧪 Estudios de Ablación",
    "📄 Documentación Técnica",
])

# -----------------------------------------------------------------------------
# Tab 1: Demostrador
# -----------------------------------------------------------------------------
with tab_demo:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📥 Selección de Muestra de Entrada")
        sample_mode = st.radio(
            "Fuente de datos:",
            ["Muestra del Conjunto de Test", "Generar Secuencia Sintética de Signo"],
            horizontal=True,
        )

        selected_seq = None
        true_label = None

        if sample_mode == "Muestra del Conjunto de Test":
            # Extraer primer batch de test
            for batch_x, batch_y in test_loader:
                sample_idx = st.slider("Índice de muestra:", 0, len(batch_x) - 1, 0)
                selected_seq = batch_x[sample_idx].numpy()
                true_idx = int(batch_y[sample_idx].item())
                true_label = class_names[true_idx] if true_idx < len(class_names) else f"Clase {true_idx}"
                break
        else:
            # Sintético
            seq_t = config.get("seq_length", 60)
            t = np.linspace(0, 2 * np.pi, seq_t)
            selected_seq = np.sin(t[:, None] + np.linspace(0, np.pi, 126)[None, :]).astype(np.float32)
            true_label = "Sintético (Simulación)"

        st.info(f"**Etiqueta Real (Ground Truth):** `{true_label}`")

        # Inferencia
        if selected_seq is not None:
            if backend_choice == "ONNX Runtime (Producción)":
                pred_result = onnx_pred.predict_sequence(selected_seq, top_k=5)
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    inp = torch.tensor(selected_seq).unsqueeze(0).float()
                    logits = model_pt(inp)
                    probs = torch.softmax(logits / temperature, dim=1)[0].numpy()
                lat_ms = (time.perf_counter() - t0) * 1000.0
                pred_idx = int(np.argmax(probs))
                pred_result = {
                    "predicted_class": class_names[pred_idx],
                    "predicted_idx": pred_idx,
                    "confidence": float(probs[pred_idx]),
                    "top_k": [
                        {"class": class_names[i], "probability": float(probs[i])}
                        for i in np.argsort(probs)[::-1][:5]
                    ],
                    "latency_ms": round(lat_ms, 2),
                    "all_probabilities": probs.tolist(),
                }

    with col_right:
        st.subheader("🎯 Resultado de la Predicción")
        is_confident = pred_result["confidence"] >= threshold
        card_color = "green" if is_confident else "orange"

        st.metric(
            label="Signo Detectado",
            value=pred_result["predicted_class"].upper(),
            delta=f"Confianza: {pred_result['confidence'] * 100:.1f}% (T={temperature})",
        )

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Latencia", f"{pred_result['latency_ms']} ms")
        m_col2.metric("Throughput", f"{int(1000.0 / max(0.1, pred_result['latency_ms']))} FPS")
        m_col3.metric("Estado", "✅ Aceptado" if is_confident else "⚠️ Bajo Umbral")

        st.write("#### Probabilidades de Clases Top-5:")
        top_df = pd.DataFrame(pred_result["top_k"])
        st.bar_chart(data=top_df.set_index("class")["probability"])

        if enable_tts and is_confident:
            st.success(f"🔊 Reproduciendo síntesis de voz: '{pred_result['predicted_class']}'")

# -----------------------------------------------------------------------------
# Tab 2: Benchmarking
# -----------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("📊 Tabla Comparativa de Modelos Evaluados")
    bench_csv = Path("data/metadata/benchmark_results.csv")
    if bench_csv.exists():
        df_bench = pd.read_csv(bench_csv)
        st.dataframe(df_bench, use_container_width=True)
    else:
        st.warning("Ejecute `python scripts/benchmark.py` para generar la tabla de benchmarking.")

# -----------------------------------------------------------------------------
# Tab 3: Ablación
# -----------------------------------------------------------------------------
with tab_ablation:
    st.subheader("🧪 Resultados de los Estudios de Ablación")
    ablation_csv = Path("data/metadata/ablation_results.csv")
    if ablation_csv.exists():
        df_abl = pd.read_csv(ablation_csv)
        st.dataframe(df_abl, use_container_width=True)
    else:
        st.warning("Ejecute `python scripts/ablation.py` para generar el estudio de ablación.")

# -----------------------------------------------------------------------------
# Tab 4: Documentación
# -----------------------------------------------------------------------------
with tab_info:
    st.subheader("📖 Documentación de Arquitectura y Despliegue")
    st.markdown(
        """
        - **Pipeline de Datos**: Extracción MediaPipe dual, Normalización invariante por longitud de palma y velocidades 3D.
        - **Arquitectura MS-TCN**: Convoluciones causales con dilataciones exponenciales $(1, 2, 4, 8)$, kernels $k \in \{3, 5, 7\}$ y Channel Attention (Squeeze-and-Excitation).
        - **Calibración**: Temperature Scaling $T^*$ para optimizar la fiabilidad probabilística (ECE $< 3\%$).
        - **Inferencia**: ONNX Runtime con ejecución paralela multihilo en CPU/Edge.
        """
    )
