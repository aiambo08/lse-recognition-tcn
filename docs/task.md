# Task List — Refactorización Profesional (Completada ✅)

## Fase 1: Scaffolding del proyecto
- `[x]` Crear `.gitignore`
- `[x]` Crear `pyproject.toml`
- `[x]` Crear `requirements.txt` / `requirements-gpu.txt` / `requirements-dev.txt`
- `[x]` Crear `configs/default.yaml`

## Fase 2: Paquete src/lse_recognition
- `[x]` `src/lse_recognition/__init__.py` (con fix de UTF-8 en consola Windows)
- `[x]` `src/lse_recognition/config.py` (carga y aplanado de YAML)
- `[x]` `src/lse_recognition/data/__init__.py`
- `[x]` `src/lse_recognition/data/dataset.py` (LandmarksNormalizer, SignLanguageDataset, create_dataloaders)
- `[x]` `src/lse_recognition/data/utils.py`
- `[x]` `src/lse_recognition/models/__init__.py`
- `[x]` `src/lse_recognition/models/tcn.py` (TCNResidualBlock, TCNSignClassifier, create_model)
- `[x]` `src/lse_recognition/models/lstm.py` (LSTMSignClassifier)
- `[x]` `src/lse_recognition/training/__init__.py`
- `[x]` `src/lse_recognition/training/trainer.py` (Trainer con Adam/ReduceLROnPlateau/EarlyStopping, run_training con semillas)
- `[x]` `src/lse_recognition/evaluation/__init__.py`
- `[x]` `src/lse_recognition/evaluation/metrics.py` (evaluate_model, matrices de confusión, reporte por clase)
- `[x]` `src/lse_recognition/inference/__init__.py`
- `[x]` `src/lse_recognition/inference/predictor.py` (RealtimeLandmarkExtractor, RealtimeSignPredictor, LSERealtimeSystem)
- `[x]` `src/lse_recognition/inference/tts.py` (TextToSpeech multi-backend asíncrono)

## Fase 3: Scripts CLI
- `[x]` `scripts/train.py` (CLI para entrenamiento con overrides de config)
- `[x]` `scripts/evaluate.py` (CLI para evaluación de checkpoints en test set)
- `[x]` `scripts/realtime.py` (CLI para inferencia interactiva con webcam y TTS)

## Fase 4: Tests Automatizados
- `[x]` `tests/__init__.py`
- `[x]` `tests/test_dataset.py` (Normalizer geométrica, shapes, rescale, dataset)
- `[x]` `tests/test_model.py` (Bloques residuales, receptive field = 61, TCN & LSTM forward, no NaN)
- `[x]` `tests/test_trainer.py` (Ciclo de optimización sintético, métricas finitas, checkpointing)
- `[x]` Ejecución: **24/24 tests pasados con éxito (100%)**

## Fase 5: Docs y notebooks
- `[x]` Estructurar directorio `notebooks/` (`01_modular_pipeline_demo.ipynb`, `fase1_extraction.ipynb`, `fase2_training.ipynb`)
- `[x]` Rediseñar y enriquecer `README.md` completo con arquitectura, instalación, CLI y API Python.
- `[x]` Actualizar `docs/project_context_dossier.md` y `docs/task.md`.

## Fase 6: Verificación y Validación
- `[x]` `pytest tests/`: 24 passed.
- `[x]` `scripts/evaluate.py --checkpoint models/best_model.pth --no-plots`: Test Accuracy 100%, F1 1.0000.
