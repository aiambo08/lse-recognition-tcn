#!/usr/bin/env python3
"""
scripts/download_dilse.py — Descarga Automatizada de Vídeos y Extracción DILSE (Fundación CNSE)
=============================================================================================

Descarga los vídeos de alta calidad de signos oficiales de la LSE desde el portal DILSE
de la Fundación CNSE y extrae sus landmarks automáticamente para construir un dataset
profesional multi-clase.

Uso:
    python scripts/download_dilse.py --num-words 30
    python scripts/download_dilse.py --words HOLA GRACIAS COMER CASA AYUDA
    python scripts/download_dilse.py --all
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import urllib3
import requests
from tqdm import tqdm

urllib3.disable_warnings()

# Añadir src/ al path
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lse_recognition.data.extraction import BatchLandmarkExtractor
from lse_recognition.data.ingestion import DatasetManifestBuilder

DEFAULT_VOCABULARY = [
    "HOLA", "AYUDA", "COMER", "CASA", "AGUA", "TRABAJAR", "AMIGO",
    "MADRE", "PADRE", "FAMILIA", "ESCUELA", "PROFESOR", "APRENDER",
    "ESCRIBIR", "LEER", "TELEFONO", "CIUDAD", "PERRO", "GATO",
    "QUERER", "PODER", "TENER", "HACER", "SABER", "ENTENDER",
    "HOMBRE", "MUJER", "NINO", "MEDICO", "HOSPITAL",
    "GRACIAS", "POR_FAVOR", "SI", "NO", "YO", "TU", "BANIO"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def find_dilse_video_urls(word: str) -> List[str]:
    """Busca y extrae URLs de vídeo en DILSE para un término dado."""
    clean_term = word.lower().replace("_", " ")
    url = f"https://fundacioncnse-dilse.org/?buscar={clean_term}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        if r.status_code != 200:
            return []
        movs = re.findall(r'https?://[^\s"\'>]+\.(?:mov|mp4)', r.text, re.IGNORECASE)
        valid_movs = [m for m in set(movs) if "bddilse" in m or "stories" in m]
        # Ordenar priorizando coincidencias exactas con el nombre
        valid_movs.sort(key=lambda x: 0 if f"/{clean_term}." in x.lower() or f"/{clean_term}-" in x.lower() else 1)
        return valid_movs
    except Exception as e:
        print(f"⚠️ Error buscando {word}: {e}")
        return []


def download_video(url: str, dest_path: Path) -> bool:
    """Descarga un archivo de vídeo desde una URL."""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, headers=HEADERS, stream=True, timeout=25, verify=False)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"⚠️ Error descargando {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Descarga de vocabulario oficial LSE desde DILSE (Fundación CNSE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-words", type=int, default=30,
        help="Número de palabras a descargar del vocabulario por defecto"
    )
    parser.add_argument(
        "--words", nargs="+", default=None,
        help="Lista explícita de palabras a descargar"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Descargar el vocabulario completo por defecto (37 palabras)"
    )
    parser.add_argument(
        "--output-dir", default="data/raw_videos",
        help="Directorio destino para vídeos descargados"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Solo descargar vídeos sin extraer landmarks con MediaPipe"
    )
    args = parser.parse_args()

    if args.words:
        vocab = [w.upper() for w in args.words]
    elif args.all:
        vocab = DEFAULT_VOCABULARY
    else:
        vocab = DEFAULT_VOCABULARY[:args.num_words]

    base_out = Path(args.output_dir)
    print("=" * 60)
    print("DESCARGA DE VOCABULARIO OFICIAL LSE — DILSE (Fundación CNSE)")
    print("=" * 60)
    print(f"Palabras a procesar: {len(vocab)}")
    print(f"Destino de vídeos:   {base_out.resolve()}")
    print("=" * 60 + "\n")

    downloaded_records = []

    for word in tqdm(vocab, desc="Buscando y descargando vídeos"):
        urls = find_dilse_video_urls(word)
        if not urls:
            print(f"❌ No se encontró vídeo para: {word}")
            continue

        for i, video_url in enumerate(urls[:2]):  # Descargar hasta 2 variantes por seña
            suffix = Path(video_url).suffix or ".mov"
            signer_id = "signer_dilse_native"
            dest_file = base_out / signer_id / word / f"{word}_var{i+1:02d}{suffix}"

            if not dest_file.exists():
                success = download_video(video_url, dest_file)
                if success:
                    print(f"✅ Descargado: {word} (variante {i+1}) -> {dest_file.name}")
                else:
                    continue
            else:
                print(f"⏩ Ya existe: {dest_file.name}")

            downloaded_records.append({
                "sample_id": f"{word}_{signer_id}_var{i+1:02d}",
                "word": word,
                "signer_id": signer_id,
                "video_path": str(dest_file),
            })
        time.sleep(0.3)  # Pausa respetuosa

    if not downloaded_records:
        print("\n❌ No se descargaron vídeos.")
        return

    import pandas as pd
    manifest_df = pd.DataFrame(downloaded_records)
    manifest_csv = Path("data/metadata/dilse_manifest.csv")
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\n📄 Manifiesto generado en: {manifest_csv} ({len(manifest_df)} vídeos)")

    # Extracción de landmarks
    if not args.skip_extraction:
        print("\n" + "=" * 60)
        print("EXTRACCIÓN DE LANDMARKS (MediaPipe Hands)")
        print("=" * 60)
        try:
            extractor = BatchLandmarkExtractor()
            processed_df = extractor.process_manifest(
                manifest_df=manifest_df,
                output_dir="data/landmarks_hands_only",
                overwrite=False,
            )
            processed_df.to_csv(manifest_csv, index=False)
            extractor.close()
            print("✅ Extracción completada.")
        except Exception as e:
            print(f"⚠️ Error o MediaPipe no disponible: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 PROCESO DILSE COMPLETADO EXITOSAMENTE ({len(manifest_df)} muestras)")
    print("=" * 60)


if __name__ == "__main__":
    main()
