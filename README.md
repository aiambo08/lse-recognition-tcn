# Sistema Profesional de Reconocimiento de Lengua de Signos Española (LSE) con TCN

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-24%2F24%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Descripción General

Este proyecto implementa una solución integral y modular para el **reconocimiento en tiempo real de Lengua de Signos Española (LSE)** utilizando **Temporal Convolutional Networks (TCN)** sobre secuencias temporales de *landmarks* articulares extraídos con **MediaPipe Hands**.

El sistema está estructurado como un **paquete Python estándar (`lse_recognition`)** acompañado de scripts CLI ejecutables, configuración centralizada por YAML, pipeline de entrenamiento con early stopping, evaluación con matrices de confusión y un motor de inferencia en tiempo real con sistema de votación, cooldown adaptativo y síntesis de voz (Text-to-Speech).

---

## 🚀 Características Principales

- **Arquitectura TCN Causal & Dilatada**: Receptive field de 61 frames (cubre la secuencia completa de 60 frames) con 677K parámetros y ~38.5 MFLOPs por inferencia.
- **Normalización Geométrica Invariante**: Normalización por frame centrada en la muñeca (`WRIST`) y escalada por la distancia articular `WRIST` $\rightarrow$ `MIDDLE_MCP`, asegurando robustez ante traslación y escala del sujeto.
- **Paquete Modular `lse_recognition`**: Separación limpia entre capas de datos (`data`), arquitecturas (`models`), entrenamiento (`training`), evaluación (`evaluation`) e inferencia (`inference`).
- **Configuración Externa YAML**: Todos los hiperparámetros (modelo, entrenamiento, filtros, inferencia, TTS) centralizados en `configs/default.yaml`.
- **Scripts CLI Robustos**: Comandos para entrenar (`scripts/train.py`), evaluar (`scripts/evaluate.py`) y ejecutar en tiempo real (`scripts/realtime.py`).
- **Inferencia en Tiempo Real**: Buffer circular temporal, filtro de estabilización por votación consecutiva (3 votos), cooldown selectivo por palabra y síntesis de voz asíncrona (pyttsx3 offline / gTTS online).
- **Suite de Pruebas Automatizadas**: 24 tests unitarios y smoke tests en `pytest` cubriendo normalización, datasets, arquitecturas TCN/LSTM y el ciclo del Trainer.

---

## 📊 Métricas de Rendimiento

Evaluado en el conjunto de test (60 secuencias independientes, 10 clases balanceadas):

| Métrica | Valor |
|---------|-------|
| **Test Accuracy** | **100.0%** (1.0000) |
| **F1 Macro** | **1.0000** |
| **F1 Weighted** | **1.0000** |
| **Latencia de Inferencia** | **~2.8 ms** por secuencia (GPU RTX) / **~12 ms** (CPU) |
| **FPS en Tiempo Real** | **~30 FPS** estables (con extracción MediaPipe) |
| **Campo Receptivo TCN** | **61 frames** |
| **Parámetros del Modelo** | **677,450** (todos entrenables) |

---

## 🗂️ Estructura del Proyecto

```
lse-recognition-tcn/
├── src/
│   └── lse_recognition/                 # Paquete Python principal
│       ├── __init__.py                  # Versionado, exports y soporte UTF-8
│       ├── config.py                    # Carga y validación de configs YAML
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py               # LandmarksNormalizer, SignLanguageDataset, DataLoader factory
│       │   └── utils.py                 # Utilidades de re-muestreo y procesamiento
│       ├── models/
│       │   ├── __init__.py
│       │   ├── tcn.py                   # TCNResidualBlock, TCNSignClassifier, create_model
│       │   └── lstm.py                  # LSTMSignClassifier (modelo comparativo)
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py               # Trainer con Adam, ReduceLROnPlateau y EarlyStopping
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py               # evaluate_model, matrices de confusión y gráficos
│       └── inference/
│           ├── __init__.py
│           ├── predictor.py             # RealtimeSignPredictor, LandmarkExtractor, LSERealtimeSystem
│           └── tts.py                   # TextToSpeech asíncrono multi-backend (pyttsx3/gTTS)
├── configs/
│   └── default.yaml                     # Hiperparámetros del sistema
├── scripts/
│   ├── train.py                         # CLI: entrenamiento del modelo
│   ├── evaluate.py                      # CLI: evaluación en test set
│   └── realtime.py                      # CLI: inferencia interactiva con webcam y TTS
├── notebooks/
│   ├── 01_modular_pipeline_demo.ipynb   # Demo interactiva del paquete modular
│   ├── fase1_extraction.ipynb           # Extracción y control de calidad (MediaPipe)
│   └── fase2_training.ipynb             # Notebook original de experimentación
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py                  # Tests de normalización y Dataset
│   ├── test_model.py                    # Tests de arquitecturas TCN y LSTM
│   └── test_trainer.py                  # Tests del ciclo de entrenamiento y checkpoints
├── data/                                # Datasets y metadatos (CSV y .npy)
│   ├── metadata/
│   │   ├── train_split_augmented.csv
│   │   ├── val_split.csv
│   │   └── test_split.csv
│   └── landmarks_hands_only/            # Coordenadas articulares (42 pts × 3)
├── models/                              # Modelos entrenados y mappings
│   ├── best_model.pth                   # Checkpoint del modelo óptimo
│   ├── label_mapping.json               # Mapeo palabra <-> ID
│   └── training_history.json            # Curvas históricas de loss y métricas
├── docs/                                # Documentación arquitectónica
│   └── project_context_dossier.md       # Dossier técnico completo
├── pyproject.toml                       # Especificación estándar PEP 517/518
├── requirements.txt                     # Dependencias base (CPU)
├── requirements-gpu.txt                 # Dependencias optimizadas con CUDA 11.8+
├── requirements-dev.txt                 # Dependencias para testing y desarrollo
├── .gitignore                           # Exclusiones git
├── ANEXO_TCN.md                         # Fundamentación teórica y matemática de TCN
└── README.md                            # Esta documentación
```

---

## 📦 Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/lse-recognition-tcn.git
cd lse-recognition-tcn
```

### 2. Crear y Activar Entorno Virtual

```bash
# Con venv estándar
python -m venv .venv

# Activar en Windows
.venv\Scripts\activate

# Activar en Linux/macOS
source .venv/bin/activate
```

### 3. Instalar el Paquete

#### Opción A: Modo Desarrollo (Recomendado)
```bash
pip install -e ".[dev,tts]"
```

#### Opción B: Con Soporte GPU CUDA
```bash
pip install -r requirements-gpu.txt
pip install -e .
```

---

## 💻 Uso mediante Scripts CLI

### 1. Entrenar el Modelo
```bash
# Entrenamiento estándar usando configs/default.yaml
python scripts/train.py

# Con opciones personalizadas
python scripts/train.py --epochs 80 --lr 0.0005 --seed 42 --evaluate
```

Opciones disponibles en `train.py`:
- `--config PATH`: Ruta al archivo de configuración YAML (default: `configs/default.yaml`).
- `--hands-only`: Usar únicamente landmarks de manos (126 features).
- `--no-normalize`: Desactivar la normalización geométrica.
- `--seed INT`: Fijar semilla aleatoria para reproducibilidad.
- `--epochs INT`: Sobreescribir el número máximo de epochs.
- `--lr FLOAT`: Sobreescribir el learning rate inicial.
- `--evaluate`: Ejecutar automáticamente la evaluación en el test set al finalizar el entrenamiento.

### 2. Evaluar un Checkpoint
```bash
# Evalúa el mejor modelo guardado en el conjunto de test
python scripts/evaluate.py --checkpoint models/best_model.pth

# Sin abrir ventanas gráficas
python scripts/evaluate.py --checkpoint models/best_model.pth --no-plots
```

### 3. Inferencia en Tiempo Real (Webcam + Voz)
```bash
# Ejecutar sistema completo con cámara por defecto (0) y TTS
python scripts/realtime.py --checkpoint models/best_model.pth

# Usar cámara externa (1) y desactivar síntesis de voz
python scripts/realtime.py --camera 1 --no-tts
```

**Controles en la ventana de vídeo:**
- `[ESPACIO]`: Pausar / Reanudar el reconocimiento.
- `[R]`: Reiniciar el buffer temporal de predicción.
- `[Q]` o `[ESC]`: Salir del programa y mostrar resumen de sesión.

---

## 🐍 Uso desde Python (API)

```python
import torch
from lse_recognition import load_config, run_training, evaluate_model, TCNSignClassifier

# 1. Cargar configuración
config = load_config("configs/default.yaml")

# 2. Entrenar modelo
trainer, history, test_loader, idx_to_word = run_training(
    config=config,
    use_hands_only=True,
    normalize=True,
    seed=42
)

# 3. Evaluar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acc, f1 = evaluate_model(
    model_path="models/best_model.pth",
    test_loader=test_loader,
    idx_to_word=idx_to_word,
    device=device
)
print(f"Accuracy obtenido: {acc:.4f}, F1: {f1:.4f}")
```

---

## ⚙️ Configuración (`configs/default.yaml`)

El comportamiento del sistema se controla a través de `configs/default.yaml`:

```yaml
model:
  seq_length: 60           # Longitud fija de la secuencia temporal
  input_features: 126      # 42 landmarks × 3 coordenadas (x, y, z)
  num_classes: 10          # Vocabulario de 10 palabras LSE
  projection_dim: 128      # Dimensión de proyección lineal inicial
  tcn_channels: [128, 128, 128, 256]  # Canales por bloque residual
  kernel_size: 3           # Tamaño de kernel de convolución
  tcn_dropout: 0.3         # Dropout en bloques TCN
  fc_hidden_dim: 128       # Capa densa del clasificador
  fc_dropout: 0.4

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 120
  early_stopping_patience: 15
  lr_scheduler_patience: 6
  lr_scheduler_factor: 0.5
  seed: 42
  use_hands_only: true
  normalize: true

inference:
  camera_index: 0
  frame_width: 640
  frame_height: 480
  buffer_size: 90          # Buffer circular de 3 segundos a 30 FPS
  min_frames_to_predict: 30
  confidence_threshold_high: 0.75
  stabilization_votes: 3   # 3 votos consecutivos para confirmar el signo
  cooldown_duration: 3.0   # Segundos de cooldown para la misma palabra
  prediction_interval: 5   # Intervalo de cálculo para optimizar CPU
  enable_tts: true         # Síntesis de voz
  tts_language: es
  tts_rate: 150
```

---

## 🧪 Ejecución de Tests Automatizados

El proyecto incluye 24 pruebas automatizadas con `pytest`:

```bash
# Ejecutar todas las pruebas
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ --cov=lse_recognition -v
```

**Áreas evaluadas:**
- `tests/test_dataset.py`: Invarianza de normalización geométrica, centrado en muñeca, supresión de mano inactiva, re-muestreo temporal e integridad del `Dataset`.
- `tests/test_model.py`: Shapes de salida, probabilidades normalizadas (Softmax), cálculo exacto del receptive field ($RF=61$), ausencia de NaNs y modelos LSTM auxiliares.
- `tests/test_trainer.py`: Ciclo de optimización, convergencia, métricas finitas y serialización de checkpoints válidos.

---

## 📚 Vocabulario Soportado

El modelo clasifica 10 signos cotidianos de la Lengua de Signos Española:

1. **HOLA** (1 mano)
2. **GRACIAS** (2 manos)
3. **POR_FAVOR** (1 mano)
4. **SI** (1 mano)
5. **NO** (1 mano)
6. **YO** (1 mano)
7. **TU** (1 mano)
8. **AYUDA** (2 manos)
9. **BANIO** / BAÑO (1 mano)
10. **COMER** (1 mano)

---

## 📄 Licencia

Este proyecto se distribuye bajo los términos de la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
