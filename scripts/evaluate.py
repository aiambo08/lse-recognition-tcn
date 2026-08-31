#!/usr/bin/env python3
"""
scripts/evaluate.py — CLI de evaluación
=========================================

Evalúa un checkpoint guardado sobre el conjunto de test.

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint models/best_model.pth
    python scripts/evaluate.py --config configs/default.yaml --no-plots

Opciones:
    --checkpoint PATH   Ruta al checkpoint .pth (default: models/best_model.pth)
    --config PATH       Ruta al YAML de configuración (default: configs/default.yaml)
    --hands-only        Usar solo landmarks de manos (default: True)
    --no-plots          No mostrar gráficos
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
from lse_recognition.data.dataset import LandmarksNormalizer, SignLanguageDataset, create_dataloaders
from lse_recognition.evaluation.metrics import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa un checkpoint del modelo TCN en el conjunto de test",
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
        "--hands-only", dest="hands_only", action="store_true", default=True,
        help="Usar solo landmarks de manos"
    )
    parser.add_argument(
        "--no-hands-only", dest="hands_only", action="store_false",
    )
    parser.add_argument(
        "--no-plots", dest="show_plots", action="store_false", default=True,
        help="No mostrar gráficos de confusión"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ No se encontró el checkpoint: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    config = load_config(args.config)

    # Necesitamos el test_loader y idx_to_word
    # Usamos create_dataloaders para detectar CSVs automáticamente
    _, _, test_loader, word_to_idx, idx_to_word = create_dataloaders(
        config,
        use_hands_only=args.hands_only,
        normalize=True,
    )

    acc, f1 = evaluate_model(
        model_path=args.checkpoint,
        test_loader=test_loader,
        idx_to_word=idx_to_word,
        device=device,
        show_plots=args.show_plots,
    )

    print(f"\n✅ Evaluación completada: Accuracy={acc:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
