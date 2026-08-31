#!/usr/bin/env python3
"""
scripts/export_model.py — CLI para Exportación de Modelos a ONNX y TorchScript
=============================================================================

Uso:
    python scripts/export_model.py --model mstcn --output models/lse_mstcn.onnx --format onnx
    python scripts/export_model.py --model tcn --output models/lse_tcn.pt --format torchscript
"""

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lse_recognition.config import load_config
from lse_recognition.deployment import export_to_onnx, export_to_torchscript
from lse_recognition.models import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Exporta modelos de reconocimiento de LSE a ONNX y TorchScript",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mstcn",
        choices=["tcn", "mstcn", "attention_lstm", "stgcn", "lstm"],
        help="Tipo de modelo a exportar",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/lse_model.onnx",
        help="Ruta de destino del archivo exportado",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "both"],
        help="Formato de exportación",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta a pesos pre-entrenados (.pth) si existen",
    )

    args = parser.parse_args()
    config = load_config()

    logger.info(f"Instanciando arquitectura: [{args.model.upper()}]")
    model = create_model(config, model_type=args.model, device="cpu")

    if args.checkpoint and Path(args.checkpoint).exists():
        import torch
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Pesos cargados desde: {args.checkpoint}")

    out_path = Path(args.output)
    seq_len = config.get("seq_length", 40)
    feat_dim = config.get("input_features", 126)

    if args.format in ("onnx", "both"):
        onnx_file = out_path if args.format == "onnx" else out_path.with_suffix(".onnx")
        export_to_onnx(model, onnx_file, seq_length=seq_len, input_features=feat_dim)
        logger.info(f"✅ ONNX exportado exitosamente: {onnx_file}")

    if args.format in ("torchscript", "both"):
        ts_file = out_path if args.format == "torchscript" else out_path.with_suffix(".pt")
        export_to_torchscript(model, ts_file, seq_length=seq_len, input_features=feat_dim)
        logger.info(f"✅ TorchScript exportado exitosamente: {ts_file}")


if __name__ == "__main__":
    main()
