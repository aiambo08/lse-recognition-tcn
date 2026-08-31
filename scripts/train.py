#!/usr/bin/env python3
"""
scripts/train.py — CLI de entrenamiento
========================================

Uso:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --hands-only --seed 123
    python scripts/train.py --no-normalize

Opciones:
    --config PATH       Ruta al YAML de configuración (default: configs/default.yaml)
    --hands-only        Usar solo landmarks de manos (por defecto True)
    --no-hands-only     Usar landmarks completos (manos + pose)
    --no-normalize      Deshabilitar normalización geométrica
    --seed INT          Semilla aleatoria (por defecto la del config, 42)
    --epochs INT        Sobreescribe el número de epochs del config
    --lr FLOAT          Sobreescribe el learning rate del config
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

from lse_recognition.config import load_config
from lse_recognition.training.trainer import run_training
from lse_recognition.evaluation.metrics import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo TCN para reconocimiento de LSE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Ruta al archivo YAML de configuración"
    )
    parser.add_argument(
        "--hands-only", dest="hands_only", action="store_true", default=True,
        help="Usar solo landmarks de manos (126 features)"
    )
    parser.add_argument(
        "--no-hands-only", dest="hands_only", action="store_false",
        help="Usar landmarks completos (manos + pose)"
    )
    parser.add_argument(
        "--no-normalize", dest="normalize", action="store_false", default=True,
        help="Deshabilitar normalización geométrica"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Semilla aleatoria (por defecto, usa la del config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Sobreescribe el número máximo de epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Sobreescribe el learning rate"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluar el mejor modelo en test set al terminar"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Cargar configuración
    config = load_config(args.config)

    # Sobreescribir valores desde CLI
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["learning_rate"] = args.lr

    print("⚙️  Configuración:")
    print(f"   Config file:   {args.config}")
    print(f"   Hands only:    {args.hands_only}")
    print(f"   Normalize:     {args.normalize}")
    print(f"   Seed:          {args.seed or config['seed']}")
    print(f"   Epochs:        {config['epochs']}")
    print(f"   LR:            {config['learning_rate']}")
    print()

    trainer, history, test_loader, idx_to_word = run_training(
        config=config,
        use_hands_only=args.hands_only,
        normalize=args.normalize,
        seed=args.seed,
    )

    # Mostrar mejor F1 alcanzado
    best_f1 = max(history["val_f1"])
    best_epoch = history["val_f1"].index(best_f1) + 1
    print(f"\n🏆 Mejor F1 validación: {best_f1:.4f} (epoch {best_epoch})")
    print(f"   Checkpoint guardado en: {config.get('model_dir', 'models')}/best_model.pth")

    # Evaluación opcional en test set
    if args.evaluate:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = f"{config.get('model_dir', 'models')}/best_model.pth"
        print("\n🔍 Evaluando en test set...")
        evaluate_model(model_path, test_loader, idx_to_word, device=device)


if __name__ == "__main__":
    main()
