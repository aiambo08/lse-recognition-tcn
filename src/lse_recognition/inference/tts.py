"""
tts.py — Síntesis de voz asíncrona (Text-to-Speech)
====================================================

Soporta dos backends:
    - pyttsx3 (offline, recomendado): sin dependencias de red
    - gTTS (online): requiere conexión a internet y playsound

El motor se inicializa en cada llamada para evitar problemas de
thread-safety en Windows con pyttsx3 (patrón "create-use-destroy").
"""

from __future__ import annotations

import os
import tempfile
import threading
from queue import Empty, Queue
from typing import Dict

# Detección opcional de backends TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from gtts import gTTS
    from playsound import playsound
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class TextToSpeech:
    """
    Sistema de síntesis de voz con múltiples backends.

    Procesa las palabras en un hilo de fondo para no bloquear
    el hilo principal de captura de video.

    Args:
        config: Diccionario con las claves:
            - enable_tts   (bool)
            - tts_language (str, e.g. 'es')
            - tts_rate     (int, palabras/minuto)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get("enable_tts", True)
        self.language = config.get("tts_language", "es")
        self.rate = config.get("tts_rate", 150)

        self.tts_queue: Queue = Queue()
        self.tts_thread: threading.Thread | None = None
        self.running = False
        self.backend: str | None = None

        if not self.enabled:
            return

        # Intentar pyttsx3 primero (offline)
        if TTS_AVAILABLE:
            try:
                test_engine = pyttsx3.init()
                test_engine.stop()
                del test_engine
                self.backend = "pyttsx3"
                print("✅ TTS inicializado con pyttsx3 (offline)")
            except Exception as e:
                print(f"⚠️  Error al inicializar pyttsx3: {e}")

        # Fallback a gTTS
        if self.backend is None and GTTS_AVAILABLE:
            self.backend = "gtts"
            print("✅ TTS usando gTTS (requiere conexión a internet)")
        elif self.backend is None:
            self.enabled = False
            print("❌ TTS no disponible. Instala pyttsx3 o gtts.")

    # ------------------------------------------------------------------ #
    # Interfaz pública                                                     #
    # ------------------------------------------------------------------ #

    def speak(self, text: str) -> None:
        """Añade texto a la cola de síntesis (no bloqueante)."""
        if self.enabled and text:
            self.tts_queue.put(text)

    def start(self) -> None:
        """Inicia el hilo de procesamiento de TTS."""
        if self.enabled and not self.running:
            self.running = True
            self.tts_thread = threading.Thread(
                target=self._tts_worker, daemon=True
            )
            self.tts_thread.start()
            print("✅ TTS thread iniciado")

    def stop(self) -> None:
        """Detiene el hilo de TTS."""
        print("🛑 Deteniendo TTS thread...")
        self.running = False
        if self.tts_thread:
            self.tts_thread.join(timeout=2.0)
        print("✅ TTS thread detenido")

    # ------------------------------------------------------------------ #
    # Worker interno                                                       #
    # ------------------------------------------------------------------ #

    def _tts_worker(self) -> None:
        """Hilo de fondo que procesa la cola de TTS."""
        print("🎤 TTS Worker: Thread iniciado")

        while self.running:
            try:
                text = self.tts_queue.get(timeout=0.5)

                if self.backend == "pyttsx3":
                    self._speak_pyttsx3(text)
                elif self.backend == "gtts":
                    self._speak_gtts(text)

                self.tts_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                import traceback
                print(f"⚠️  TTS Worker: Error inesperado: {type(e).__name__}: {e}")
                traceback.print_exc()

        print("🎤 TTS Worker: Thread detenido")

    def _speak_pyttsx3(self, text: str) -> None:
        """Sintetiza con pyttsx3 (crea y destruye el engine en cada llamada)."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)

            # Buscar voz en español
            voices = engine.getProperty("voices")
            for voice in voices:
                name = voice.name.lower()
                if "spanish" in name or "español" in name or "espanol" in name:
                    engine.setProperty("voice", voice.id)
                    break

            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            import traceback
            print(f"   ❌ Error en pyttsx3: {type(e).__name__}: {e}")
            traceback.print_exc()

    def _speak_gtts(self, text: str) -> None:
        """Sintetiza con gTTS y playsound."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_file = fp.name

            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(temp_file)
            playsound(temp_file)

            try:
                os.remove(temp_file)
            except OSError:
                pass
        except Exception as e:
            import traceback
            print(f"   ❌ Error en gTTS: {type(e).__name__}: {e}")
            traceback.print_exc()
