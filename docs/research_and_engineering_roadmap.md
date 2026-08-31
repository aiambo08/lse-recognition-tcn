# Hoja de Ruta Estratégica: LSE Recognition — Hacia un Proyecto de Investigación y Producción Publicable

Este documento define la estrategia técnica, científica y de ingeniería para evolucionar el sistema desde el prototipo MVP actual hasta una **solución de vanguardia publicable en conferencias/revistas de visión artificial e interacción persona-ordenador (HCI)** (ej. *IEEE Access, Pattern Recognition Letters, CVPR Workshops, Interacción*).

---

## 1. Análisis Exhaustivo de Datasets de LSE (Lengua de Signos Española)

Para que los resultados sean científicamente válidos y representativos del mundo real, el sistema debe evaluarse bajo un protocolo **Signer-Independent (Cross-Signer)** sobre corpus con múltiples personas sordas nativas.

### Comparativa de Corpus y Datasets

| Dataset / Corpus | Institución / Origen | Nº Clases | Nº Signantes | Modalidad | Formato / Tipo | Acceso / Licencia | Adecuación al Proyecto |
|---|---|---|---|---|---|---|---|
| **Corpus LSE** | **CNLSE** (Centro de Normalización Lingüística de la LSE / Real Patronato) | Vocabulario abierto (+1.000 señas) | **>50 signantes** nativos de toda España | Aislado y Continuo | Vídeos HD + Anotaciones ELAN (`.eaf`) | Solicitud académica formal (CNLSE) | ⭐⭐⭐⭐⭐ **Máximo estándar de referencia lingüística** |
| **DILSE (Diccionario LSE)** | **Fundación CNSE** | ~4.500 palabras estándar | Múltiples especialistas nativos | Signos Aislados | Vídeos MP4 web / vocabulario normativo | Público / Scraping académico regulado | ⭐⭐⭐⭐⭐ **Ideal para Isolated SLR a gran escala (50-200 clases)** |
| **LSE_Lex40 / LSE_Lex** | Grupos Universitarios (UAH / UPM / UPV) | 40 – 100 señas cotidianas | 10 – 20 signantes | Signos Aislados | Vídeos RGB / Landmarks pre-extraídos | Repositorios académicos (Zenodo / UAH) | ⭐⭐⭐⭐ **Excelente para benchmarking directo y rápido** |
| **UAH-DactiloLSE** | Universidad de Alcalá | 27 letras (Alfabeto dactilológico) | 12 signantes | Estático / Dinámico | Imágenes y secuencias de vídeo | Acceso abierto (Repositorio UAH) | ⭐⭐⭐ **Complemento ideal para deletreo manual** |
| **WLASL / AUTSL** *(Referencia Internacional)* | CVPR / ChaLearn Benchmark | 100 – 2.000 clases | 100+ signantes | Signos Aislados | Vídeos + Landmarks | Abierto (Kaggle / GitHub) | ⭐⭐⭐⭐ **Benchmark comparativo cruzado para la sección de experimentos** |

---

## 2. Protocolo Científico de Validación (Cross-Signer / LOSO)

En un paper o proyecto profesional, **nunca se divide el dataset de forma aleatoria por vídeos**, sino **por signantes**:

```
Signantes Totales: [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10]
├── Train Set (80%):  [S1, S2, S3, S4, S5, S6, S7, S8]
├── Val Set (10%):    [S9]
└── Test Set (10%):   [S10]  ← Nunca visto durante el entrenamiento (Signer-Independent)
```

O mediante **Leave-One-Signer-Out Cross-Validation (LOSO-CV)**: se realizan $K$ iteraciones evaluando cada vez sobre un signante diferente y promediando los resultados.

---

## 3. Desglose de Fases de Implementación (Roadmap)

### Fase A: Pipeline de Datos e Ingesta Automatizada
- [x] **Módulo de Ingesta (`src/lse_recognition/data/ingestion.py`)**:
  - Descargador / Parser automático para DILSE (`scripts/download_dilse.py`).
  - Manifiesto unificado multi-signante y escáner de directorios.
- [x] **Extractor Multimodal Avanzado (`src/lse_recognition/data/extraction.py`)**:
  - Extracción robusta con soporte dual (MediaPipe Legacy + Tasks API).
  - Cálculo de features cinemáticas derivadas (`src/lse_recognition/data/kinematics.py`): velocidades 3D $(\Delta x, \Delta y, \Delta z)$, aceleraciones y distancias euclidianas inter-digitales.
- [x] **Splitter Cross-Signer (`src/lse_recognition/data/splits.py`)**:
  - Particionado Signer-Independent y generador de folds LOSO-CV / Stratified Group Splits.

### Fase B: Arquitecturas Avanzadas y Benchmarking Comparativo
- [x] **Multi-Scale TCN (MS-TCN)** (`src/lse_recognition/models/mstcn.py`):
  - Ramas paralelas con múltiples kernels ($k=3, 5, 7$) + Squeeze-and-Excitation (Channel Attention) + Residual Shortcuts.
- [x] **Spatial-Temporal Graph Convolutional Network (ST-GCN)** (`src/lse_recognition/models/stgcn.py`):
  - Modelado biomecánico del grafo esquelético articular de manos (42 nodos, 21 conexiones anatómicas por mano).
- [x] **BiLSTM con Temporal Attention** (`src/lse_recognition/models/attention_lstm.py`):
  - Mecanismo de auto-atención temporal aditiva para ponderación explicable de frames clave.
- [x] **Suite de Benchmarking Comparativo Automatizado** (`scripts/benchmark.py`):
  - TCN vs MS-TCN vs BiLSTM vs Attention-BiLSTM vs ST-GCN con exportación en Markdown y LaTeX IEEE/ACM.

### Fase C: Rigor Experimental y Ablation Studies
- [x] **Estudios de Ablación (Ablation Studies)** (`scripts/ablation.py`, `docs/ablation_study.md`):
  - Impacto de la normalización geométrica (con vs sin normalización).
  - Impacto del campo receptivo (dilaciones $[1,2]$, $[1,2,4]$, $[1,2,4,8]$, $[1,2,4,8,16]$).
  - Impacto de multi-escala temporal ($k=3$ vs $k \in \{3,5\}$ vs $k \in \{3,5,7\}$).
- [x] **Calibración de Confianza (Confidence Calibration)** (`src/lse_recognition/evaluation/calibration.py`):
  - Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score y Temperature Scaling ($T^*$).
- [x] **Análisis de Eficiencia Computacional**:
  - Medición formal de número de parámetros, latencia (ms) en CPU y GPU, throughput (FPS) y consumo de memoria.

### Fase D: Despliegue de Producción y APIs
- [x] **Exportación Optimizada** (`src/lse_recognition/deployment/export.py`, `scripts/export_model.py`):
  - Conversión a **ONNX Runtime** con ejes dinámicos y validación de paridad numérica, y exportación a **TorchScript** (`.pt`).
- [x] **Servidor de Inferencia / API** (`src/lse_recognition/server/app.py`):
  - Endpoint **FastAPI** REST (`/predict/sequence`, `/calibrate`, `/classes`, `/health`) y WebSocket (`/ws/stream`) para streaming en tiempo real a 60 FPS con ventana deslizante.
- [x] **Interfaz Web de Demostración** (`src/lse_recognition/demo/app.py`):
  - Dashboard interactivo multi-pestaña (Streamlit) con selección de modelo/backend, gráfico de barras de probabilidades calibradas en directo, métricas de latencia/FPS y síntesis de voz (TTS).

### Fase E: Redacción del Artículo Científico / Dossier Técnico
- Estructura estándar IEEE / ACM:
  1. *Abstract & Introduction*: Contexto social, accesibilidad y estado del arte en LSE.
  2. *Related Work*: Análisis crítico de TCNs, GCNs y Vision Transformers en SLR.
  3. *Methodology*: Normalización geométrica invariante + Arquitectura TCN causal dilatada.
  4. *Experimental Setup*: Descripción del corpus multi-signante, protocolo LOSO-CV.
  5. *Results & Discussion*: Tablas comparativas, matrices de confusión, estudio de ablación.
  6. *Real-Time Deployment*: Análisis de latencia y sistema de votación/cooldown.
  7. *Conclusions & Future Work*: Extensión a LSE continuo y modelos multimodales.
