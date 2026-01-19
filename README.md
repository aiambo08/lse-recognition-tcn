# Sistema de Reconocimiento de Lengua de Signos Española (LSE) con TCN

## Descripción General

Este proyecto implementa un **sistema completo de reconocimiento de Lengua de Signos Española (LSE)** utilizando **Temporal Convolutional Networks (TCN)** para clasificación de secuencias de landmarks de manos. El sistema incluye entrenamiento, evaluación exhaustiva e inferencia en tiempo real con síntesis de voz.

### Características Principales

- ✅ **Modelo TCN de alta precisión** (~98% accuracy en test)
- ✅ **Normalización geométrica** invariante a escala y traslación
- ✅ **Sistema de tiempo real** con webcam (30 FPS)
- ✅ **Text-to-Speech integrado** (pyttsx3/gTTS)
- ✅ **Interfaz visual interactiva** con métricas en tiempo real
- ✅ **Early stopping y learning rate scheduling**
- ✅ **Evaluación detallada** con matrices de confusión

---

## Resultados Esperados

Con la configuración predeterminada (TCN, 60 frames, hands-only landmarks):

| Métrica | Valor |
|---------|-------|
| **Test Accuracy** | ~98.3% |
| **F1 Macro** | ~0.983 |
| **Tiempo de inferencia** | ~28ms (GPU) |
| **Campo receptivo** | 61 frames |
| **Parámetros del modelo** | ~677K |

---

## Requisitos del Sistema

### Hardware

- **Webcam** (para inferencia en tiempo real)
- **GPU CUDA** (opcional, acelera entrenamiento ~3x)
- **RAM**: Es necesario una cantidad decente de RAM

### Software

- **Python**: entre python 3.9 y python 3.12 para que sea compatible con MediaPipe (nostros hemos trabajado con Python 3.11)
- **Sistema Operativo**: Windows, Linux o macOS

---

## Instalación

### 1. Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Instalar Dependencias

#### Dependencias Core (Entrenamiento y Evaluación)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

**Nota**: Para CPU-only, usa:
```bash
pip install torch torchvision torchaudio
```

#### Dependencias para Tiempo Real

```bash
pip install opencv-python mediapipe
```

#### Text-to-Speech (Elige una opción)

**Opción 1: pyttsx3 (Recomendado - Offline)**
```bash
pip install pyttsx3
```

**Opción 2: gTTS (Requiere internet)**
```bash
pip install gtts playsound==1.2.2
```

### 3. Verificar Instalación

```python
import torch
import cv2
import mediapipe as mp

print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"OpenCV: {cv2.__version__}")
print(f"MediaPipe: {mp.__version__}")
```

---

## Estructura del Proyecto

```
practica3_aln/
├── fase2.ipynb                    # Notebook principal 
├── README.md                      # Este documento
├── fase1.ipynb                    # Notebook extracción de datos
├── data/                          # Datos generados en Fase 1
│   ├── metadata/
│   │   ├── train_split_augmented.csv
│   │   ├── val_split.csv
│   │   └── test_split.csv
│   ├── landmarks_hands_only/      # Archivos .npy (42 puntos × 3 coords)
│   │   ├── AYUDA/
│   │   ├── BAÑO/
│   │   ├── COMER/
│   │   └── ... (10 palabras)
│   └── raw_videos/                # Videos originales (opcional)
├── models/                        # Modelos entrenados (generado automáticamente)
    ├── best_model.pth             # Checkpoint del mejor modelo
    ├── label_mapping.json         # Mapeo palabra → índice
    └── training_history.json      # Historial de métricas

```

---

## Guía de Uso

### Paso 1: Preparar los Datos

**Requisito previo**: Debes haber completado la **Fase 1** para generar:
- `data/metadata/*.csv` (splits de train/val/test)
- `data/landmarks_hands_only/*.npy` (secuencias de landmarks)

Verifica que existan estos archivos:
```python
from pathlib import Path

assert Path('data/metadata/train_split_augmented.csv').exists()
assert Path('data/metadata/val_split.csv').exists()
assert Path('data/metadata/test_split.csv').exists()
print("✅ Datos listos")
```

### Paso 2: Entrenar el Modelo

Abre `fase2.ipynb` y ejecuta las celdas en orden:

#### 2.1 Configuración Inicial (Secciones 1-2)

```python
# Ejecutar celdas de imports y configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ Using device: {device}')
```

#### 2.2 Ejecutar Entrenamiento (Sección 7)

```python
trainer, history, test_loader, idx_to_word = run_training(
    MODEL_CONFIG, 
    use_hands_only=True,   # Usar solo landmarks de manos
    normalize=True          # Aplicar normalización geométrica
)
```

**Salida esperada**:
```
📄 Usando splits: train: data/metadata/train_split_augmented.csv ...
✅ Dataset cargado: ... | muestras: 536 | clases: 10
✅ Modelo TCN creado: 677,130 parámetros totales
   Campo receptivo: 61 frames (de 60 totales)
============================================================
Iniciando entrenamiento
============================================================
Epoch 1: Train loss 1.2345, acc 0.4567 | Val loss 0.9876, acc 0.6543, F1 0.6321
...
  ✅ Mejor modelo actualizado (F1=1.0000)
============================================================
✅ Entrenamiento completado
============================================================
```

**Duración**: Si se usa GPU es bastante rápido.

### Paso 3: Evaluar el Modelo

```python
evaluate_model('models/best_model.pth', test_loader, idx_to_word, device)
```

**Salida esperada**:
```
============================================================
RESULTADOS
============================================================
✅ Test Accuracy:     1.0000 (100%)
✅ F1 Macro:          1.0000
✅ F1 Weighted:       1.0000
============================================================

REPORTE POR CLASE
============================================================

              precision    recall  f1-score   support

       AYUDA     1.0000    1.0000    1.0000         6
       BANIO     1.0000    1.0000    1.0000         6
       COMER     1.0000    1.0000    1.0000         6
     GRACIAS     1.0000    1.0000    1.0000         6
        HOLA     1.0000    1.0000    1.0000         6
          NO     1.0000    1.0000    1.0000         6
   POR_FAVOR     1.0000    1.0000    1.0000         6
          SI     1.0000    1.0000    1.0000         6
          TU     1.0000    1.0000    1.0000         6
          YO     1.0000    1.0000    1.0000         6

    accuracy                         1.0000        60
   macro avg     1.0000    1.0000    1.0000        60
weighted avg     1.0000    1.0000    1.0000        60
```

Se generarán automáticamente:
- **Matriz de confusión absoluta**
- **Matriz de confusión normalizada**
- **Análisis de errores por clase**

### Paso 4: Ejecutar Sistema en Tiempo Real

```python
run_realtime_recognition('models/best_model.pth')
```

**Controles durante la ejecución**:
- `[ESPACIO]` - Pausar/Reanudar detección
- `[R]` - Limpiar buffer de frames
- `[Q]` o `[ESC]` - Salir del sistema

**Interfaz visual**:
- **Panel superior**: FPS, palabra detectada, confianza
- **Panel lateral derecho**: Top-3 predicciones con probabilidades
- **Panel inferior**: Estado del buffer, confianza promedio, historial

**Audio**: Síntesis de voz automática al detectar cada signo (si TTS está habilitado)

---

## Conceptos Clave Implementados

### 1. Normalización Geométrica

**Clase**: `LandmarksNormalizer`

**Objetivo**: Hacer el modelo invariante a:
- ✅ **Escala** (tamaño de la mano del usuario)
- ✅ **Traslación** (distancia a la cámara, posición en el frame)

**Estrategia**:
1. **Centrado**: Restar coordenadas de la muñeca (landmark 0)
2. **Escalado**: Dividir por distancia wrist → middle_mcp (landmark 9)
3. **Frame-by-frame**: Aplicar independientemente en cada fotograma

**Fórmula**:

```
landmark_normalizado = (landmark - wrist) / ||middle_mcp - wrist||
```

**Ventajas**:
- Robustez ante variaciones de distancia
- Funciona con diferentes tamaños de manos
- No requiere calibración previa

---

### 2. Temporal Convolutional Network (TCN)

(En el archivo [ANEXO_TCN.md] se explica matemáticamente de forma detallada el modelo TCN)

**Arquitectura**: 4 bloques residuales con dilated convolutions

**Dilaciones exponenciales**: [1, 2, 4, 8]
- Permite campo receptivo de **61 frames** con solo 4 capas
- Captura dependencias temporales a corto y largo plazo

**Ventajas sobre LSTM**:
| Característica | TCN | LSTM |
|----------------|-----|------|
| Paralelización | ✅ Sí | ❌ Secuencial |
| Memoria GPU | Menor | Mayor |
| Vanishing gradients | ✅ No sufre | ⚠️ Problema conocido |
| Velocidad (inferencia) | **28ms** | 42ms |
| Precisión (dataset pequeño) | **98.3%** | 96.1% |

**Componentes**:
1. **Proyección inicial**: 126 features → 128 dims
2. **Bloques TCN**: 4 × [Conv1D → BatchNorm → ReLU → Dropout] + Residual
3. **Global Average Pooling**: Agregación temporal
4. **Clasificador**: 2 capas densas → 10 clases

**Campo receptivo**:
```
RF = 1 + 2 × (dilation_1 × 2 + dilation_2 × 2 + dilation_3 × 2 + dilation_4 × 2)
   = 1 + 2 × (1×2 + 2×2 + 4×2 + 8×2)
   = 61 frames
```

---

### 3. Augmentación de Datos

**Aplicada en Fase 1**, presente en `train_split_augmented.csv`:

1. **Temporal warping**: Deformaciones no-lineales en el eje temporal
2. **Escalado espacial**: Variaciones aleatorias de escala (±10%)
3. **Ruido gaussiano**: Perturbaciones en coordenadas (σ=0.01)

**Resultado**: 
- Train: **536 muestras** (originales + augmentadas)
- Val: **62 muestras** (solo originales)
- Test: **62 muestras** (solo originales)

---

### 4. Sistema de Estabilización en Tiempo Real

**Problema**: Predicciones fluctuantes frame-a-frame

**Solución**: Voting system + cooldown

**Componentes**:
1. **Buffer circular**: 90 frames (3 segundos a 30 FPS)
2. **Stabilization votes**: Requiere N=3 predicciones consecutivas consistentes
3. **Confidence threshold**: Solo acepta predicciones con confianza > 75%
4. **Cooldown selectivo**: Bloquea detecciones de la **misma palabra** durante 3 segundos (permite detectar palabras diferentes)

**Algoritmo**:
```
# Verificar si estamos en cooldown para la misma palabra
time_since_last = time.time() - last_prediction_time
in_cooldown = time_since_last < cooldown_duration

if in_cooldown and predicted_word == last_prediction:
    return None  # Bloquear solo si es la misma palabra

if confidence > 0.75:
    prediction_history.append(predicted_word)
    
    if len(prediction_history) >= 3:
        most_common_word, count = Counter(prediction_history).most_common(1)[0]
        
        if count >= 3:
            ✅ CONFIRMAR PREDICCIÓN
            Activar TTS
            Actualizar last_prediction y last_prediction_time
            Limpiar historial
```

---

### 5. Early Stopping y Learning Rate Scheduling

**Early Stopping**:
- **Métrica**: F1 en validación
- **Paciencia**: 15 epochs sin mejora
- **Objetivo**: Prevenir overfitting

**ReduceLROnPlateau**:
- **Factor**: 0.5 (reduce LR a la mitad)
- **Paciencia**: 6 epochs sin mejora en val_loss
- **Objetivo**: Escapar de plateaus

**Ejemplo de evolución del LR**:
```
Epoch 1-20:  lr = 1e-3
Epoch 21-30: lr = 5e-4  (reducido)
Epoch 31-40: lr = 2.5e-4 (reducido nuevamente)
```

---

## Configuración Avanzada

### Modificar Hiperparámetros del Modelo

Edita `MODEL_CONFIG` en la sección 2 del notebook:

```python
MODEL_CONFIG = {
    'seq_length': 60,           # Longitud de secuencia (frames)
    'projection_dim': 128,      # Dimensión de proyección inicial
    'tcn_channels': [128, 128, 128, 256],  # Canales por bloque TCN
    'kernel_size': 3,           # Tamaño de kernel de convolución
    'tcn_dropout': 0.3,         # Dropout en bloques TCN (0-1)
    'fc_dropout': 0.4,          # Dropout en clasificador (0-1)
    
    # Entrenamiento
    'batch_size': 32,           # Tamaño de batch
    'learning_rate': 1e-3,      # Learning rate inicial
    'weight_decay': 1e-4,       # Regularización L2
    'epochs': 120,              # Máximo de epochs
    'early_stopping_patience': 15,  # Paciencia para early stopping
    'lr_scheduler_patience': 6,     # Paciencia para LR scheduler
    
    # Parámetros tiempo real (referencia)
    'confidence_threshold_high': 0.75,
    'cooldown_duration': 3.0,
    'stabilization_votes': 3
}
```

### Ajustar Parámetros de Tiempo Real

Edita `INFERENCE_CONFIG` en la sección 8.2:

```python
INFERENCE_CONFIG = {
    'camera_index': 0,          # 0=webcam principal, 1=externa
    'frame_width': 640,
    'frame_height': 480,
    'buffer_size': 90,          # Frames en buffer (↑ = más memoria)
    'min_frames_to_predict': 30,  # Mínimo para predecir (↓ = más rápido)
    'confidence_threshold_high': 0.75,  # Umbral de aceptación (↑ = más estricto)
    'stabilization_votes': 3,   # Votos requeridos (↑ = más estable)
    'cooldown_duration': 3.0,   # Segundos entre detecciones de la misma palabra
    'prediction_interval': 5,   # Predecir cada N frames (optimización)
    'enable_tts': True,         # Activar/desactivar TTS
}
```

---

## Troubleshooting

### Problemas Comunes

#### 1. Error: "No se encontraron los CSV necesarios"

**Causa**: Faltan archivos de Fase 1

**Solución**:
```bash
# Verificar existencia
ls data/metadata/train_split_augmented.csv
ls data/metadata/val_split.csv
ls data/metadata/test_split.csv

# Si no existen, ejecutar Fase 1 primero
```

#### 2. Error: "CUDA out of memory"

**Causa**: Batch size demasiado grande para GPU

**Solución**:
```python
# Reducir batch_size en MODEL_CONFIG
MODEL_CONFIG['batch_size'] = 16  # o 8
```

#### 3. Webcam no se abre

**Causa**: Índice de cámara incorrecto

**Solución**:
```python
# Probar diferentes índices
INFERENCE_CONFIG['camera_index'] = 1  # o 2, 3...
```

#### 4. No detecta manos

**Posibles causas y soluciones**:
- ✅ **Iluminación insuficiente** → Añadir luz frontal
- ✅ **Manos fuera de encuadre** → Ajustar distancia (50-80cm)
- ✅ **Confidence muy bajo** → Reducir `mediapipe_min_detection_confidence` a 0.3
- ✅ **Fondo complejo** → Usar fondo uniforme

#### 5. Predicciones inestables

**Solución**: Aumentar estabilización
```python
INFERENCE_CONFIG['stabilization_votes'] = 5  # en vez de 3
INFERENCE_CONFIG['confidence_threshold_high'] = 0.85  # más estricto
```

#### 6. TTS no funciona

**Solución**:
```bash
# Reinstalar pyttsx3
pip uninstall pyttsx3
pip install pyttsx3

# O usar gTTS como alternativa
pip install gtts playsound==1.2.2
```

#### 7. Error: "ImportError: DLL load failed" (Windows)

**Causa**: Falta Visual C++ Redistributable

**Solución**:
- Descargar e instalar [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

## Interpretación de Resultados

### Métricas de Evaluación

**Accuracy**: Porcentaje de predicciones correctas
```
Accuracy = (TP + TN) / Total
```
- **Bueno**: > 95%
- **Excelente**: > 98%

**F1 Macro**: Promedio armónico de precision y recall (balanceado por clase)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Bueno**: > 0.90
- **Excelente**: > 0.95

**Matriz de Confusión**:
- **Diagonal**: Predicciones correctas
- **Fuera de diagonal**: Confusiones entre clases
- **Normalizada**: Muestra tasas de error por clase

### Análisis de Curvas de Aprendizaje

**Curvas saludables**:
```
Train Loss: Decreciente suave
Val Loss:   Decreciente, convergencia con train
Val F1:     Creciente hasta plateau
```

**Señales de problemas**:
- ❌ **Overfitting**: `val_loss` aumenta mientras `train_loss` disminuye
- ❌ **Underfitting**: Ambas losses altas y no convergen
- ❌ **Learning rate alto**: Oscilaciones violentas

