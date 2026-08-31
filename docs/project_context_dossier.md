# lse-recognition-tcn — Technical Context Dossier

## 1. Executive Summary & Stack

- **Domain & Purpose:** Reconocimiento de Lengua de Signos Española (LSE) — sistema completo de clasificación de gestos signados mediante landmarks de manos, que cubre pipeline ML (Fase 1: extracción + Fase 2: entrenamiento/inferencia) y tiempo real con síntesis de voz.
- **Core Stack:** Python 3.9-3.12 (recomendado 3.11) · PyTorch (CUDA 11.8 opcional) · MediaPipe (extracción de landmarks) · OpenCV (captura de vídeo) · NumPy / Pandas / scikit-learn · Matplotlib / Seaborn · pyttsx3 / gTTS (TTS)
- **Storage & Infrastructure:** Sistema de ficheros local únicamente. Sin base de datos, sin nube, sin colas de mensajes.
  - Modelos: `models/` (`.pth`, `.json`)
  - Datos: `data/landmarks_hands_only/` (`.npy`) + `data/metadata/` (`.csv`)
- **Entry Points:**
  - `fase1.ipynb` → pipeline de extracción de datos y augmentación
  - `fase2.ipynb` → entrenamiento, evaluación y sistema de tiempo real (producción)

---

## 2. Architecture & File Structure

```
lse-recognition-tcn/
├── fase1.ipynb                         # Fase 1: captura vídeo → extracción MediaPipe → .npy + splits CSV
├── fase2.ipynb                         # Fase 2: dataset, TCN model, trainer, evaluación, tiempo real
├── README.md                           # Guía completa de uso
├── ANEXO_TCN.md                        # Fundamentos matemáticos del TCN (notación, FLOPs, backprop)
├── vocabulario_10_palabras.txt         # Las 10 palabras del vocabulario (una por línea)
├── signos.html                         # [Stub HTML, uso desconocido – posiblemente referencia visual]
│
├── data/
│   ├── metadata/
│   │   ├── dataset_clean_10words.csv   # Dataset completo sin splits (referencia)
│   │   ├── landmarks_metadata.csv      # Metadata de todos los landmarks extraídos
│   │   ├── train_split.csv             # Split entrenamiento sin augmentación
│   │   ├── train_split_augmented.csv   # Split entrenamiento CON augmentación (usado en training)
│   │   ├── val_split.csv               # Split validación (originales únicamente)
│   │   └── test_split.csv              # Split test (originales únicamente)
│   ├── landmarks_hands_only/           # Arrays .npy shape (60, 42, 3) por muestra
│   │   ├── AYUDA/ · BANIO/ · COMER/ · GRACIAS/ · HOLA/
│   │   ├── NO/ · POR_FAVOR/ · SI/ · TU/ · YO/
│   │   └── [Convención nombre]: {WORD}_rep_{repID}_{HHMMSS}[_aug_N].npy
│   ├── landmarks/                      # Landmarks completos (body+hands) – versión alternativa no usada en training
│   └── raw_videos/                     # Vídeos fuente originales (opcionales, para re-extracción)
│
└── models/
    ├── best_model.pth                  # Checkpoint PyTorch del mejor modelo (8MB, ~677K params)
    ├── label_mapping.json              # word→index: {AYUDA:0, BANIO:1, ..., YO:9}
    └── training_history.json           # Historial epoch-level: train/val loss, acc, f1, lr
```

**Paradigma arquitectónico:** Pipeline monolítico por notebooks. No existe separación en módulos Python independientes: toda la lógica de dataset, modelo, entrenamiento e inferencia está definida in-cell dentro de `fase2.ipynb`.

**Cross-Cutting Concerns:**
- Logging: `print()` con emojis Unicode (✅❌📄) — sin framework de logging estructurado.
- Reproducibilidad: No se fija semilla (`torch.manual_seed`) de forma explícita en el README [UNVERIFIED: puede estar definida en una celda de `fase2.ipynb`].
- Portabilidad: El código detecta automáticamente `cuda` vs `cpu`.

---

## 3. Core Data Flow & Implementations

| Feature / Subsystem | Archivos primarios | Persistencia | Lógica clave |
|---|---|---|---|
| **Fase 1: Extracción de landmarks** | `fase1.ipynb` | `data/landmarks_hands_only/**/*.npy`, `data/metadata/*.csv` | Webcam → OpenCV → MediaPipe Holistic → hands-only 21 pts × 2 manos × 3 coords = 126 features/frame → padding/truncado a 60 frames → `.npy` shape `(60, 42, 3)` |
| **Augmentación de datos** | `fase1.ipynb` | `train_split_augmented.csv` + `*_aug_N.npy` | Temporal warping + escalado espacial ±10% + ruido gaussiano σ=0.01. Solo en train. Ratio: 536 train / 62 val / 62 test |
| **Dataset & DataLoader** | `fase2.ipynb` (class `LSEDataset`) | RAM | Lee `.npy` desde CSV, aplica `LandmarksNormalizer` (centrado en muñeca landmark 0, escalado por dist. wrist→middle_mcp landmark 9), reshape `(T, 42, 3)→(T, 126)` |
| **Normalización geométrica** | `fase2.ipynb` (class `LandmarksNormalizer`) | — | `x_norm = (x - wrist) / ‖middle_mcp - wrist‖` frame-by-frame. Invariante a escala y traslación. |
| **Arquitectura TCN** | `fase2.ipynb` (class `SignLanguageTCN`) | `models/best_model.pth` | Proj(126→128) + 4 × ResidualBlock(Conv1D dilatadas [1,2,4,8]) + GAP + FC(256→128→10). RF=61 frames. 677,130 params. |
| **Entrenamiento** | `fase2.ipynb` (`Trainer` class / `run_training()`) | `models/best_model.pth`, `training_history.json` | Adam (lr=1e-3, wd=1e-4) + ReduceLROnPlateau (factor=0.5, patience=6) + EarlyStopping (patience=15, métrica: F1 val) + CrossEntropyLoss |
| **Evaluación** | `fase2.ipynb` (`evaluate_model()`) | — | Accuracy, F1 macro/weighted, reporte por clase, matrices de confusión (absoluta y normalizada) |
| **Inferencia tiempo real** | `fase2.ipynb` (`run_realtime_recognition()`) | — | OpenCV webcam → MediaPipe frame-a-frame → normalización → TCN → voting system + cooldown selectivo → pyttsx3/gTTS |
| **Estabilización predicciones** | `fase2.ipynb` | — | Buffer circular 90 frames, umbral confianza 75%, 3 votos consecutivos consistentes, cooldown 3s para la misma palabra |

---

## 4. Environment & Runtime Context

**Variables de entorno requeridas:** Ninguna. La configuración completa está en `MODEL_CONFIG` y `INFERENCE_CONFIG` dentro del notebook.

**Workflow de desarrollo local:**
```bash
# 1. Crear entorno virtual (Python 3.11 recomendado)
python -m venv venv
venv\Scripts\activate               # Windows
source venv/bin/activate            # Linux/Mac

# 2. Instalar dependencias (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
pip install opencv-python mediapipe
pip install pyttsx3                  # TTS offline (recomendado)
# alternativa TTS online:
# pip install gtts playsound==1.2.2

# 3. Ejecutar pipeline
jupyter notebook fase1.ipynb        # Solo si se necesita re-extraer datos
jupyter notebook fase2.ipynb        # Entrenamiento + evaluación + tiempo real
```

**Hiperparámetros clave (MODEL_CONFIG):**
```python
MODEL_CONFIG = {
    'seq_length': 60,           # frames por muestra
    'projection_dim': 128,
    'tcn_channels': [128, 128, 128, 256],
    'kernel_size': 3,
    'tcn_dropout': 0.3,
    'fc_dropout': 0.4,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 120,
    'early_stopping_patience': 15,
    'lr_scheduler_patience': 6,
    'confidence_threshold_high': 0.75,
    'cooldown_duration': 3.0,
    'stabilization_votes': 3
}
```

**Estrategia de testing:** No existen tests unitarios automatizados. La validación es empírica vía celdas del notebook. Las métricas de evaluación final sirven como test de regresión implícito.

---

## 5. Implementation Status & Technical Hotspots

### ✅ Estable / Implementado
- Pipeline completo Fase 1 → Fase 2
- Arquitectura TCN con bloques residuales dilatados
- Normalización geométrica `LandmarksNormalizer`
- Sistema de entrenamiento con early stopping y LR scheduling
- Evaluación: accuracy, F1, matrices de confusión
- Sistema de tiempo real con webcam + voting + TTS
- Modelo pre-entrenado incluido (`models/best_model.pth`, ~8MB)
- Dataset completo con splits y augmentaciones ya generadas

### ⚠️ En progreso / Experimental
- `signos.html` — fichero de 290 bytes sin propósito documentado [UNVERIFIED]
- `data/landmarks/` — directorio paralelo con landmarks completos (cuerpo entero); no se usa en el training actual pero está extraído (posible extensión futura)
- La LR nunca se llegó a reducir en el entrenamiento registrado (todos los epochs con lr=0.001), lo que puede indicar convergencia rápida o que el LR scheduler no se activó antes del early stopping en época 21

### 🔴 Deuda técnica y footguns conocidos
1. **Sin requirements.txt / pyproject.toml**: Las dependencias deben instalarse manualmente siguiendo el README. Riesgo de incompatibilidades de versión.
2. **Monolithic notebooks**: Todo el código en celdas Jupyter sin estructura modular. Dificulta reutilización, testing unitario e integración en producción.
3. **Ratio params/muestras extremo**: 677K parámetros vs 536 muestras (ratio ~1263). Mitigado con regularización pero frágil ante expansión del vocabulario.
4. **Dataset pequeño y homogéneo**: 62 muestras de test (6 por clase). El 100% accuracy en test es casi con certeza sobreoptimista — el modelo no ha sido validado con usuarios externos ni en condiciones de iluminación variadas.
5. **Nombre de clase inconsistente**: `BAÑO` aparece como `BANIO` en archivos (sanitizado para compatibilidad ASCII), pero el vocabulario usa `BAÑO`. Requiere atención si se procesa la carpeta directamente.
6. **TTS bloqueante**: `pyttsx3` en algunos entornos puede bloquear el hilo de captura de vídeo. `playsound==1.2.2` es un pin de versión frágil.
7. **Sin semilla aleatoria fijada**: El entrenamiento no es reproducible bit-a-bit entre ejecuciones [UNVERIFIED: verificar celdas del notebook].
8. **macOS artifacts**: Presencia de `__MACOSX/` y múltiples `.DS_Store` en el repositorio (deben añadirse al `.gitignore`).

### 📐 Convenciones a seguir
- **Nombres de ficheros landmark**: `{WORD}_rep_{repID}_{HHMMSS}[_aug_N].npy`
- **Etiquetas de clase**: Usar el mapping de `models/label_mapping.json` para consistencia (BANIO, no BAÑO)
- **Normalización siempre activa** (`normalize=True`) — los pesos del modelo pre-entrenado asumen datos normalizados
- **Sólo `landmarks_hands_only`** para el modelo actual — `landmarks/` (cuerpo completo) no es compatible sin reentrenar
- **Configuración centralizada**: Modificar únicamente `MODEL_CONFIG` e `INFERENCE_CONFIG` dentro del notebook; no hardcodear valores en otras celdas

---

## Diagrama de Flujo de Datos

```
[Webcam / Vídeos]
       │
       ▼
  MediaPipe Holistic                     (fase1.ipynb)
  → 21 puntos/mano × 2 manos × (x,y,z)
  → shape (T_orig, 42, 3)
       │
  Padding / Truncado a T=60 frames
  → Augmentación (warping, escala, ruido)
       │
       ▼
  .npy files (60, 42, 3) + splits CSV    (data/)
       │
       ▼
  LSEDataset.__getitem__()               (fase2.ipynb)
  → LandmarksNormalizer → (60, 126)
  → torch.Tensor
       │
       ▼
  SignLanguageTCN.forward()
  ├── Linear Projection: (B,60,126) → (B,60,128)
  ├── TCNBlock d=1: (B,60,128) → (B,60,128)
  ├── TCNBlock d=2: (B,60,128) → (B,60,128)
  ├── TCNBlock d=4: (B,60,128) → (B,60,128)
  ├── TCNBlock d=8: (B,60,128) → (B,60,256)  [residual 1×1 conv]
  ├── Global Average Pool: (B,256,60) → (B,256)
  └── FC: (B,256) → (B,128) → (B,10) → Softmax
       │
       ▼
  Predicción + Softmax Confidence         (tiempo real)
  → VotingSystem (buffer 90f, N=3, thr=0.75)
  → TTS (pyttsx3 / gTTS)
```
