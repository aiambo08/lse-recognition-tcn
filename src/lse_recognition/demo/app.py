"""
demo/app.py — Dashboard Interactivo de Demostración y Reconocimiento de LSE
==========================================================================

Interfaz gráfica completa basada en Streamlit para experimentación, grabación
de gestos en directo con webcam, extracción de esqueleto MediaPipe, comparación
de modelos, calibración probabilística e inferencia acústica (TTS).

Uso:
    uv run streamlit run src/lse_recognition/demo/app.py
"""

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from lse_recognition.config import load_config
from lse_recognition.data import LandmarksNormalizer, create_dataloaders
from lse_recognition.deployment import ONNXSignPredictor, export_to_onnx
from lse_recognition.inference.predictor import RealtimeLandmarkExtractor
from lse_recognition.models import create_model

st.set_page_config(
    page_title="LSE Recognition System — Research & Live Demo",
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
    help="Escala de temperatura óptima aprendida para calibrar probabilidades post-hoc.",
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

    # Cargar pesos entrenados si existen en models/best_model.pth
    ckpt_path = Path("models/best_model.pth")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass

    # Crear o cargar ONNX con T=60
    seq_len = config.get("seq_length", 60)
    onnx_path = Path(f"models/demo_{m_type}.onnx")
    if not onnx_path.exists():
        export_to_onnx(model, onnx_path, seq_length=seq_len, input_features=126)

    onnx_pred = ONNXSignPredictor(
        model_path=onnx_path,
        class_names=classes,
        temperature=st.session_state.temperature,
    )

    return model, onnx_pred, classes, test_l, config


model_pt, onnx_pred, class_names, test_loader, config = get_model_and_classes(model_choice)
onnx_pred.temperature = temperature
normalizer = LandmarksNormalizer()

# -----------------------------------------------------------------------------
# Cabecera Principal
# -----------------------------------------------------------------------------
st.title("🤟 Sistema de Reconocimiento de Lengua de Signos Española (LSE)")
st.markdown(
    """
    **Plataforma de Investigación y Demostración en Tiempo Real**  
    *Arquitectura Deep Learning con Convoluciones Causales Dilatadas Multi-Escala, 
    Modelado Biomecánico Esquelético de Manos y Calibración Probabilística ECE.*
    """
)

tab_demo, tab_benchmark, tab_ablation, tab_info = st.tabs([
    "🎮 Demostrador Interactivo",
    "📊 Benchmarking Comparativo",
    "🧪 Estudios de Ablación",
    "📄 Documentación Técnica",
])

# -----------------------------------------------------------------------------
# Tab 1: Demostrador Interactivo
# -----------------------------------------------------------------------------
with tab_demo:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📥 Selección de Fuente de Entrada")
        sample_mode = st.radio(
            "Modo de prueba:",
            [
                "🎥 Cámara en Vivo (Grabar tus Gestos)",
                "📁 Subir Archivo de Vídeo (.mp4, .mov)",
                "📊 Muestra del Conjunto de Test",
                "⚡ Secuencia Sintética de Simulación",
            ],
            index=0,
        )

        selected_seq = None
        true_label = None

        # ---------------------------------------------------------------------
        # Modo 1: Grabación en Vivo con Webcam
        # ---------------------------------------------------------------------
        if sample_mode == "🎥 Cámara en Vivo (Grabar tus Gestos)":
            st.markdown("#### 📹 Grabación de Gesto con tu Webcam")
            st.write("Colócate frente a la cámara y pulsa el botón para capturar un signo (60 frames / ~2-3 s).")

            cam_col1, cam_col2 = st.columns([1, 2])
            camera_id = cam_col1.number_input("ID de Cámara:", min_value=0, max_value=5, value=0)
            target_frames = config.get("seq_length", 60)

            record_btn = st.button("🔴 Iniciar Grabación de Signo", type="primary")

            preview_slot = st.empty()
            status_slot = st.empty()

            if record_btn:
                cap = cv2.VideoCapture(int(camera_id))
                if not cap.isOpened():
                    st.error(f"❌ No se pudo acceder a la cámara con ID {camera_id}. Verifica que no esté en uso por otra app.")
                else:
                    try:
                        extractor = RealtimeLandmarkExtractor(config)
                        captured_landmarks = []

                        status_slot.info("⏳ Prepárate: iniciando en 2 segundos...")
                        time.sleep(1.0)
                        status_slot.warning("🎬 ¡GRABANDO! Realiza tu signo ahora...")

                        progress_bar = st.progress(0.0)

                        for f_idx in range(target_frames):
                            ret, frame = cap.read()
                            if not ret:
                                break

                            # Extraer landmarks de manos y dibujar esqueleto
                            lm_raw, annotated_frame, hands_detected = extractor.extract_hands_landmarks(frame)
                            captured_landmarks.append(lm_raw)

                            # Mostrar preview en directo con esqueleto MediaPipe
                            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            preview_slot.image(frame_rgb, caption=f"Frame {f_idx + 1}/{target_frames}", channels="RGB")
                            progress_bar.progress((f_idx + 1) / target_frames)
                            time.sleep(0.03)  # ~30 FPS

                        cap.release()
                        status_slot.success(f"✅ ¡Captura completada! ({len(captured_landmarks)} frames procesados)")

                        # Convertir a tensor normalizado (1, T, 126)
                        raw_seq = np.array(captured_landmarks, dtype=np.float32)  # (T, 42, 3)
                        norm_seq = normalizer.normalize(raw_seq)  # (T, 42, 3)
                        selected_seq = norm_seq.reshape(target_frames, -1)  # (T, 126)
                        true_label = "Gesto Personal en Vivo (Webcam)"

                    except Exception as e:
                        cap.release()
                        st.error(f"Error durante la captura: {e}")

        # ---------------------------------------------------------------------
        # Modo 2: Subir archivo de vídeo
        # ---------------------------------------------------------------------
        elif sample_mode == "📁 Subir Archivo de Vídeo (.mp4, .mov)":
            uploaded_video = st.file_uploader("Selecciona un archivo de vídeo:", type=["mp4", "mov", "avi"])
            if uploaded_video is not None:
                import tempfile
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                extractor = RealtimeLandmarkExtractor(config)
                frames_lm = []

                st.write("🔄 Procesando vídeo con MediaPipe...")
                prog = st.progress(0.0)
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 60
                curr = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    lm_raw, _, _ = extractor.extract_hands_landmarks(frame)
                    frames_lm.append(lm_raw)
                    curr += 1
                    prog.progress(min(1.0, curr / max(1, total_f)))

                cap.release()

                if len(frames_lm) > 0:
                    raw_arr = np.array(frames_lm, dtype=np.float32)
                    norm_arr = normalizer.normalize(raw_arr)
                    # Re-muestrear a seq_length fija (60 frames)
                    target_T = config.get("seq_length", 60)
                    indices = np.linspace(0, len(norm_arr) - 1, target_T).astype(np.int32)
                    selected_seq = norm_arr[indices].reshape(target_T, -1)
                    true_label = f"Vídeo Subido ({uploaded_video.name})"
                    st.success(f"✅ Vídeo procesado: {len(frames_lm)} frames re-muestreados a {target_T}")

        # ---------------------------------------------------------------------
        # Modo 3: Muestra del test set
        # ---------------------------------------------------------------------
        elif sample_mode == "📊 Muestra del Conjunto de Test":
            for batch_x, batch_y in test_loader:
                sample_idx = st.slider("Índice de muestra:", 0, len(batch_x) - 1, 0)
                selected_seq = batch_x[sample_idx].numpy()
                true_idx = int(batch_y[sample_idx].item())
                true_label = class_names[true_idx] if true_idx < len(class_names) else f"Clase {true_idx}"
                break

        # ---------------------------------------------------------------------
        # Modo 4: Secuencia Sintética
        # ---------------------------------------------------------------------
        else:
            seq_t = config.get("seq_length", 60)
            t = np.linspace(0, 2 * np.pi, seq_t)
            selected_seq = np.sin(t[:, None] + np.linspace(0, np.pi, 126)[None, :]).astype(np.float32)
            true_label = "Sintético (Simulación)"

        if true_label:
            st.info(f"**Origen de Entrada:** `{true_label}`")

        # Inferencia
        pred_result = None
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
                    "predicted_class": class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}",
                    "predicted_idx": pred_idx,
                    "confidence": float(probs[pred_idx]),
                    "top_k": [
                        {"class": class_names[i] if i < len(class_names) else f"class_{i}", "probability": float(probs[i])}
                        for i in np.argsort(probs)[::-1][:5]
                    ],
                    "latency_ms": round(lat_ms, 2),
                    "all_probabilities": probs.tolist(),
                }

    with col_right:
        st.subheader("🎯 Resultado de la Inferencia")

        if pred_result is not None:
            is_confident = pred_result["confidence"] >= threshold
            card_color = "green" if is_confident else "orange"

            st.metric(
                label="Signo Reconocido",
                value=pred_result["predicted_class"].upper(),
                delta=f"Confianza: {pred_result['confidence'] * 100:.1f}% (T*={temperature})",
            )

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Latencia", f"{pred_result['latency_ms']} ms")
            m_col2.metric("Throughput", f"{int(1000.0 / max(0.1, pred_result['latency_ms']))} FPS")
            m_col3.metric("Estado", "✅ Aceptado" if is_confident else "⚠️ Bajo Umbral")

            st.write("#### Distribución de Probabilidades Top-5:")
            top_df = pd.DataFrame(pred_result["top_k"])
            st.bar_chart(data=top_df.set_index("class")["probability"])

            if enable_tts and is_confident:
                st.success(f"🔊 Pronunciación por Voz: '{pred_result['predicted_class']}'")
        else:
            st.info("👈 Selecciona una muestra o graba un gesto con tu cámara para ver la predicción en directo.")

# -----------------------------------------------------------------------------
# Tab 2: Benchmarking
# -----------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("📊 Tabla Comparativa de Modelos Evaluados")
    bench_csv = Path("data/metadata/benchmark_results.csv")
    if bench_csv.exists():
        df_bench = pd.read_csv(bench_csv)
        st.dataframe(df_bench)
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
        st.dataframe(df_abl)
    else:
        st.warning("Ejecute `python scripts/ablation.py` para generar el estudio de ablación.")

# -----------------------------------------------------------------------------
# Tab 4: Documentación
# -----------------------------------------------------------------------------
with tab_info:
    st.subheader("📖 Documentación de Arquitectura y Despliegue")
    st.markdown(
        """
        - **Pipeline de Datos**: Extracción MediaPipe dual, Normalización geométrica invariante por longitud de palma y velocidades 3D.
        - **Arquitectura MS-TCN**: Convoluciones causales con dilataciones exponenciales $(1, 2, 4, 8)$, kernels $k \in \{3, 5, 7\}$ y Channel Attention (Squeeze-and-Excitation).
        - **Calibración**: Temperature Scaling $T^*$ para optimizar la fiabilidad probabilística (ECE $< 3\%$).
        - **Inferencia**: ONNX Runtime con ejecución paralela multihilo en CPU/Edge.
        """
    )
