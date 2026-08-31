"""
predictor.py — Inferencia en tiempo real con webcam
====================================================

Contiene:
    - RealtimeLandmarkExtractor : Extrae landmarks de manos con MediaPipe
    - RealtimeSignPredictor     : Buffer + voting + cooldown → predicción estable
    - LSERealtimeSystem         : Sistema completo (webcam + predicción + TTS)
    - run_realtime_recognition  : Punto de entrada de alto nivel
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

from lse_recognition.data.dataset import LandmarksNormalizer
from lse_recognition.inference.tts import TextToSpeech
from lse_recognition.models.lstm import LSTMSignClassifier
from lse_recognition.models.tcn import TCNSignClassifier


# ------------------------------------------------------------------ #
# Extractor de landmarks                                              #
# ------------------------------------------------------------------ #

class RealtimeLandmarkExtractor:
    """
    Extrae landmarks de ambas manos por frame usando MediaPipe Hands.

    Salida por frame: array (42, 3) — 21 puntos mano izq. + 21 puntos mano der.
    Las coordenadas de la mano no detectada se ponen a cero.
    """

    N_HAND_POINTS = 21

    def __init__(self, config: Dict):
        self.config = config
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.get("mediapipe_min_detection_confidence", 0.7),
            min_tracking_confidence=config.get("mediapipe_min_tracking_confidence", 0.5),
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp_hands

    def extract_hands_landmarks(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Extrae landmarks de manos en un frame.

        Args:
            frame: Frame BGR de OpenCV.

        Returns:
            (landmarks, annotated_frame, hands_detected)
            - landmarks:       ndarray shape (42, 3)
            - annotated_frame: frame con anotaciones dibujadas
            - hands_detected:  True si al menos una mano fue detectada
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        landmarks = np.zeros((42, 3), dtype=np.float32)
        annotated_frame = frame.copy()
        hands_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # MediaPipe devuelve 'Left'/'Right' desde la perspectiva del sujeto
                # (espejo), por eso se invierten:
                label = hand_info.classification[0].label
                if label == "Left":
                    offset = 0   # mano derecha real → slot izquierdo
                else:
                    offset = 21  # mano izquierda real → slot derecho

                for idx, lm in enumerate(hand_lm.landmark):
                    landmarks[offset + idx] = [lm.x, lm.y, lm.z]

                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                )
                hands_detected = True

        return landmarks, annotated_frame, hands_detected

    def release(self) -> None:
        """Libera recursos de MediaPipe."""
        self.hands.close()


# ------------------------------------------------------------------ #
# Predictor con buffer + voting                                       #
# ------------------------------------------------------------------ #

class RealtimeSignPredictor:
    """
    Sistema de predicción de signos en tiempo real.

    Gestiona:
        - Buffer circular de frames
        - Sistema de estabilización (voting)
        - Cooldown selectivo por palabra
        - Top-K predicciones

    Args:
        model:       Modelo PyTorch en eval mode.
        config:      Diccionario de configuración.
        normalizer:  LandmarksNormalizer (puede ser None).
        idx_to_word: Mapeo índice → palabra.
        device:      Dispositivo PyTorch.
    """

    def __init__(
        self,
        model,
        config: Dict,
        normalizer: Optional[LandmarksNormalizer],
        idx_to_word: Dict[int, str],
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.normalizer = normalizer
        self.idx_to_word = idx_to_word
        self.device = device

        self.frame_buffer: deque = deque(maxlen=config["buffer_size"])
        self.prediction_history: deque = deque(maxlen=config["stabilization_votes"])
        self.confidence_history: deque = deque(maxlen=30)

        self.last_prediction: Optional[str] = None
        self.last_prediction_time: float = 0.0
        self.frame_count: int = 0
        self.predictions_made: int = 0

        self.model.eval()

    def add_frame(self, landmarks: np.ndarray) -> None:
        """Añade landmarks de un frame al buffer."""
        self.frame_buffer.append(landmarks)
        self.frame_count += 1

    def can_predict(self) -> bool:
        """True si hay suficientes frames con señal para intentar predicción."""
        if len(self.frame_buffer) < self.config["min_frames_to_predict"]:
            return False
        if self.frame_count % self.config["prediction_interval"] != 0:
            return False

        recent = list(self.frame_buffer)[-self.config["min_frames_to_predict"]:]
        non_empty = sum(1 for f in recent if np.any(f != 0))
        return non_empty >= self.config["min_frames_to_predict"] * 0.7

    def predict(self) -> Tuple[Optional[str], float, Optional[List]]:
        """
        Realiza predicción sobre el buffer actual.

        Returns:
            (palabra_confirmada, confianza, top3)
            - palabra_confirmada: str si se confirma un signo, None en caso contrario
            - confianza: float [0, 1]
            - top3: lista de (palabra, prob) o None
        """
        if not self.can_predict():
            return None, 0.0, None

        seq_len = self.config["seq_length"]

        # Extraer y resamplear secuencia
        sequence = list(self.frame_buffer)
        sequence = self._resample_sequence(sequence, seq_len)
        sequence = np.array(sequence, dtype=np.float32)

        # Aplanar: (seq_len, 42, 3) → (seq_len, 126)
        sequence_flat = sequence.reshape(seq_len, -1)

        if self.normalizer:
            sequence_flat = self.normalizer(sequence_flat)

        input_tensor = (
            torch.from_numpy(sequence_flat).unsqueeze(0).float().to(self.device)
        )

        with torch.no_grad():
            probs = self.model.predict_proba(input_tensor)
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred_idx = pred_idx.item()

        # Top-3
        k = min(3, len(self.idx_to_word))
        top_probs, top_indices = torch.topk(probs, k=k, dim=1)
        top3 = [
            (self.idx_to_word[idx.item()], prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]

        self.confidence_history.append(confidence)
        predicted_word = self.idx_to_word[pred_idx]

        # Cooldown selectivo (bloquea solo la misma palabra)
        time_since_last = time.time() - self.last_prediction_time
        in_cooldown = time_since_last < self.config["cooldown_duration"]
        if in_cooldown and predicted_word == self.last_prediction:
            return None, confidence, top3

        # Estabilización
        if confidence >= self.config["confidence_threshold_high"]:
            self.prediction_history.append(predicted_word)

            if len(self.prediction_history) >= self.config["stabilization_votes"]:
                word_counts = Counter(self.prediction_history)
                stable_word, count = word_counts.most_common(1)[0]

                if count >= self.config["stabilization_votes"]:
                    self.last_prediction = stable_word
                    self.last_prediction_time = time.time()
                    self.predictions_made += 1
                    self.prediction_history.clear()
                    return stable_word, confidence, top3

        return None, confidence, top3

    def _resample_sequence(self, sequence: List, target_length: int) -> List:
        """Re-muestrea una lista de frames a la longitud objetivo."""
        current_length = len(sequence)
        if current_length == 0:
            return [np.zeros((42, 3), dtype=np.float32)] * target_length
        indices = np.linspace(0, current_length - 1, target_length).astype(int)
        return [sequence[i] for i in indices]

    def get_buffer_status(self) -> Dict:
        """Estado del buffer para la UI."""
        return {
            "buffer_fill": len(self.frame_buffer),
            "buffer_capacity": self.config["buffer_size"],
            "can_predict": self.can_predict(),
            "avg_confidence": (
                float(np.mean(self.confidence_history))
                if self.confidence_history else 0.0
            ),
            "predictions_made": self.predictions_made,
        }

    def reset_buffer(self) -> None:
        """Limpia el buffer y el historial de predicciones."""
        self.frame_buffer.clear()
        self.prediction_history.clear()


# ------------------------------------------------------------------ #
# Sistema completo                                                    #
# ------------------------------------------------------------------ #

class LSERealtimeSystem:
    """
    Sistema completo de reconocimiento LSE en tiempo real.

    Integra: webcam → MediaPipe → normalización → TCN → voting → TTS → UI.

    Args:
        model_path:       Ruta al checkpoint del modelo.
        model_config:     Config de arquitectura (claves del MODEL_CONFIG).
        inference_config: Config de inferencia (claves del INFERENCE_CONFIG).
        device:           Dispositivo PyTorch.
    """

    def __init__(
        self,
        model_path: str | Path,
        model_config: Dict,
        inference_config: Dict,
        device: torch.device,
    ):
        print("\n" + "=" * 60)
        print("INICIALIZANDO SISTEMA LSE EN TIEMPO REAL")
        print("=" * 60)

        self.inference_config = inference_config
        self.device = device

        # Cargar modelo
        print("Cargando modelo...")
        checkpoint = torch.load(model_path, map_location=device)
        self.saved_config = checkpoint["config"]

        if "tcn_channels" in self.saved_config:
            model_class = TCNSignClassifier
            print("   Tipo: TCN")
        else:
            model_class = LSTMSignClassifier
            print("   Tipo: LSTM")

        self.model = model_class(self.saved_config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"   ✅ Modelo cargado (Val F1: {checkpoint.get('val_f1', 0):.4f})")

        # Cargar mapeo de clases
        label_mapping_path = Path("models") / "label_mapping.json"
        with open(label_mapping_path, "r") as f:
            word_to_idx = json.load(f)
        self.idx_to_word = {v: k for k, v in word_to_idx.items()}
        print(f"   ✅ {len(self.idx_to_word)} clases cargadas")

        # Componentes
        print("\nInicializando componentes...")
        self.normalizer = LandmarksNormalizer()
        self.landmark_extractor = RealtimeLandmarkExtractor(inference_config)
        self.predictor = RealtimeSignPredictor(
            self.model, inference_config, self.normalizer,
            self.idx_to_word, device
        )
        self.tts = TextToSpeech(inference_config)
        self.tts.start()

        print("   ✅ Extractor de landmarks")
        print("   ✅ Predictor")
        print(f"   ✅ Text-to-Speech ({'activado' if self.tts.enabled else 'desactivado'})")

        # Estado
        self.running = False
        self.paused = False
        self.cap = None
        self.fps = 0.0
        self.detected_words: List[Dict] = []
        self.start_time: Optional[float] = None

        print("\n" + "=" * 60)
        print("✅ SISTEMA LISTO")
        print("=" * 60 + "\n")

    def start(self) -> None:
        """Inicia la captura de webcam y el bucle principal."""
        print("Iniciando cámara...")
        self.cap = cv2.VideoCapture(self.inference_config["camera_index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.inference_config["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.inference_config["frame_height"])

        if not self.cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara.")
            return

        self.running = True
        self.start_time = time.time()

        print("\n" + "=" * 60)
        print("SISTEMA EN EJECUCIÓN")
        print("=" * 60)
        print("\nControles:")
        print("  [ESPACIO] - Pausar/Reanudar")
        print("  [R]       - Limpiar buffer")
        print("  [Q/ESC]   - Salir")
        print("\n" + "=" * 60 + "\n")

        self._run_loop()

    def _run_loop(self) -> None:
        """Bucle principal de captura y predicción."""
        frame_times: deque = deque(maxlen=30)
        last_frame_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️  Error al capturar frame")
                break

            # Calcular FPS
            current_time = time.time()
            frame_times.append(current_time - last_frame_time)
            last_frame_time = current_time
            self.fps = 1.0 / float(np.mean(frame_times)) if frame_times else 0.0

            # Efecto espejo
            frame = cv2.flip(frame, 1)

            if not self.paused:
                landmarks, annotated_frame, hands_detected = (
                    self.landmark_extractor.extract_hands_landmarks(frame)
                )

                if hands_detected:
                    self.predictor.add_frame(landmarks)

                predicted_word, confidence, top3 = self.predictor.predict()

                if predicted_word:
                    self.detected_words.append({
                        "word": predicted_word,
                        "confidence": confidence,
                        "timestamp": datetime.now(),
                    })
                    print(
                        f"✅ [{datetime.now().strftime('%H:%M:%S')}] "
                        f"Detectado: {predicted_word.upper()} "
                        f"(confianza: {confidence:.2%})"
                    )
                    self.tts.speak(predicted_word)

                display_frame = self._draw_interface(
                    annotated_frame, predicted_word, confidence, top3
                )
            else:
                display_frame = frame.copy()
                cv2.putText(
                    display_frame, "PAUSADO", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3
                )

            cv2.imshow("LSE — Reconocimiento en Tiempo Real", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q o ESC
                break
            elif key == ord(" "):
                self.paused = not self.paused
                print("⏸️  PAUSADO" if self.paused else "▶️  REANUDADO")
            elif key == ord("r"):
                self.predictor.reset_buffer()
                print("🔄 Buffer limpiado")

        self.stop()

    def _draw_interface(
        self,
        frame: np.ndarray,
        predicted_word: Optional[str],
        confidence: float,
        top3: Optional[List],
    ) -> np.ndarray:
        """Dibuja overlays informativos sobre el frame."""
        height, width = frame.shape[:2]

        # Panel superior semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # FPS
        if self.inference_config.get("show_fps"):
            cv2.putText(
                frame, f"FPS: {self.fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Predicción actual
        if predicted_word:
            cv2.putText(
                frame, f"DETECTADO: {predicted_word.upper()}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
            )
            cv2.putText(
                frame, f"Confianza: {confidence:.1%}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Top-3
        if top3 and self.inference_config.get("show_buffer_status"):
            y_offset = 180
            cv2.putText(
                frame, "Top-3:", (width - 250, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            for i, (word, prob) in enumerate(top3):
                y_offset += 30
                color = (
                    (0, 255, 0) if prob >= 0.75
                    else (0, 165, 255) if prob >= 0.5
                    else (128, 128, 128)
                )
                cv2.putText(
                    frame, f"{i+1}. {word}: {prob:.1%}",
                    (width - 250, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        # Estado del buffer
        if self.inference_config.get("show_buffer_status"):
            status = self.predictor.get_buffer_status()
            buffer_pct = status["buffer_fill"] / max(status["buffer_capacity"], 1)
            bar_w, bar_h = 200, 20
            bar_x = width - bar_w - 20
            bar_y = height - 80

            cv2.rectangle(
                frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 2
            )
            fill_w = int(bar_w * buffer_pct)
            color = (0, 255, 0) if status["can_predict"] else (0, 165, 255)
            cv2.rectangle(
                frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1
            )
            cv2.putText(
                frame,
                f"Buffer: {status['buffer_fill']}/{status['buffer_capacity']}",
                (bar_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            cv2.putText(
                frame,
                f"Conf. promedio: {status['avg_confidence']:.1%}",
                (bar_x, bar_y + bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Historial
        if self.detected_words:
            cv2.putText(
                frame, "Historial:", (20, height - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            for i, word_info in enumerate(reversed(self.detected_words[-5:])):
                y_pos = height - 90 + i * 25
                elapsed = (datetime.now() - word_info["timestamp"]).total_seconds()
                alpha = max(0, 1 - elapsed / 10)
                color = tuple(int(c * alpha) for c in (255, 255, 255))
                cv2.putText(
                    frame, word_info["word"], (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        return frame

    def stop(self) -> None:
        """Detiene el sistema limpiamente."""
        print("\n🛑 Deteniendo sistema...")
        self.running = False

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_extractor.release()
        self.tts.stop()

        # Resumen de sesión
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE SESIÓN")
        print("=" * 60)

        if self.start_time:
            duration = time.time() - self.start_time
            print(f"Duración: {duration:.1f} segundos")

        print(f"Palabras detectadas: {len(self.detected_words)}")

        if self.detected_words:
            print("\nPalabras reconocidas:")
            word_counts = Counter(w["word"] for w in self.detected_words)
            for word, count in word_counts.most_common():
                print(f"  - {word}: {count} veces")

        print("\n✅ Sistema detenido correctamente")
        print("=" * 60 + "\n")


# ------------------------------------------------------------------ #
# Punto de entrada                                                    #
# ------------------------------------------------------------------ #

def run_realtime_recognition(
    model_path: str = "models/best_model.pth",
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Lanza el sistema de reconocimiento en tiempo real.

    Args:
        model_path: Ruta al checkpoint del modelo.
        config:     Diccionario de configuración. Si None, carga default.yaml.
        device:     Dispositivo PyTorch. Si None, auto-detecta CUDA.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config is None:
        from lse_recognition.config import load_config
        config = load_config()

    try:
        system = LSERealtimeSystem(
            model_path=model_path,
            model_config=config,
            inference_config=config,
            device=device,
        )
        system.start()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupción por teclado")
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
