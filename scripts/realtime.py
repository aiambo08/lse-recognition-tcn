#!/usr/bin/env python3
"""
scripts/realtime.py — CLI de inferencia en tiempo real
========================================================

Lanza el sistema de reconocimiento LSE con webcam en tiempo real.

Uso:
    python scripts/realtime.py
    python scripts/realtime.py --checkpoint models/best_model.pth
    python scripts/realtime.py --camera 1 --no-tts
    python scripts/realtime.py --config configs/default.yaml

Opciones:
    --checkpoint PATH   Ruta al checkpoint .pth (default: models/best_model.pth)
    --config PATH       Ruta al YAML de configuración (default: configs/default.yaml)
    --camera INT        Índice de cámara (default: 0)
    --no-tts            Deshabilitar síntesis de voz
    --no-fps            No mostrar FPS en pantalla

Controles durante la ejecución:
    [ESPACIO]  Pausar/Reanudar
    [R]        Limpiar buffer
    [Q/ESC]    Salir
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

# Añadir src/ al path para poder importar sin instalación
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from lse_recognition.config import load_config
from lse_recognition.inference.predictor import run_realtime_recognition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia en tiempo real para reconocimiento de LSE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default="models/best_model.pth",
        help="Ruta al checkpoint del modelo"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Ruta al archivo YAML de configuración"
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Índice de cámara a usar (sobreescribe el config)"
    )
    parser.add_argument(
        "--no-tts", dest="tts", action="store_false", default=True,
        help="Deshabilitar síntesis de voz"
    )
    parser.add_argument(
        "--no-fps", dest="fps", action="store_false", default=True,
        help="Ocultar FPS en pantalla"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ No se encontró el checkpoint: {args.checkpoint}")
        print("  Entrena el modelo primero con: python scripts/train.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    config = load_config(args.config)

    # Sobreescribir desde CLI
    if args.camera is not None:
        config["camera_index"] = args.camera
    if not args.tts:
        config["enable_tts"] = False
    if not args.fps:
        config["show_fps"] = False

    run_realtime_recognition(
        model_path=args.checkpoint,
        config=config,
        device=device,
    )


if __name__ == "__main__":
    main()
