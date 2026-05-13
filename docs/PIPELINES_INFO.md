================================================================================
INFORMACIÓN DE PIPELINES KEDRO - PROYECTO DE PREDICCIÓN DE DESEMPEÑO
================================================================================

DESCRIPCIÓN GENERAL:
El proyecto implementa un flujo completo de Machine Learning usando Kedro,
que gestiona la carga, limpieza, transformación, validación y modelado de
datos de empleados (recursos humanos).

================================================================================
PIPELINES IMPLEMENTADOS:
================================================================================

1. DATA INGESTION (data_ingestion)
   Propósito: Carga inicial de datos crudos y diagnóstico
   Nodo: explorar_y_diagnosticar
   Entrada: empleados_raw, evaluaciones_raw, capacitaciones_raw, ausencias_raw
   Salida: reporte_diagnostico_inicial.json
   Archivos: 4 CSVs del directorio data/01_raw/

2. DATA CLEANING (data_cleaning)
   Propósito: Limpieza y normalización de cada tabla
   Nodo: clean_hr_data (iterativo para cada dataset)
   Entrada: {dataset}_raw + params:cleaning_params
   Salida: empleados_int, evaluaciones_int, capacitaciones_int, ausencias_int
   Ubicación: data/02_intermediate/

3. DATA TRANSFORMATION (data_transformation)
   Propósito: Agregación de datos y creación de features
   Nodos:
     - aggregate_evaluations: Agrupa evaluaciones por empleado
     - aggregate_ausencias: Agrupa ausencias por empleado
     - aggregate_capacitaciones: Agrupa capacitaciones por empleado
     - create_master_table: Une todas las tablas
     - scale_and_encode: Normalización final
   Salida: primary_dataset.csv (309 empleados × 14 features)
   Ubicación: data/03_primary/

4. DATA VALIDATION (data_validation)
   Propósito: Validación de calidad de datos finales
   Nodos:
     - validate_primary_data: Validación de completitud y tipos
     - compare_before_after: Comparación pre/post transformación
     - save_final_report: Genera reporte
   Salida: validation_report.txt
   Ubicación: data/08_reporting/

5. MODEL TRAINING (model_training) ★ NUEVO
   Propósito: División de datos, escalado y entrenamiento de modelos
   Nodos:
     - split_data: Divide en train/test (80/20)
     - scale_features: Normaliza features con StandardScaler
     - train_regression_models: Entrena 7 modelos diferentes
     - get_predictions: Obtiene predicciones de test
     - save_models: Guarda modelos en pickle
   Modelos entrenados:
     1. Linear Regression
     2. Ridge (alpha=1.0)
     3. Lasso (alpha=0.1)
     4. Random Forest (100 estimadores)
     5. Gradient Boosting (100 estimadores)
     6. SVR (kernel=rbf)
     7. K-Nearest Neighbors (k=5)
   Salida: 7 modelos pickle + scaler.pkl
   Ubicación: results/models/

6. MODEL EVALUATION (model_evaluation) ★ NUEVO
   Propósito: Evaluación completa de modelos
   Nodos:
     - calculate_metrics: Calcula RMSE, MAE, R² para todos los modelos
     - cross_validation_scores: CV scores 5-fold
     - get_residuals_analysis: Análisis de residuos del mejor modelo
     - generate_evaluation_report: Reporte detallado
   Métricas calculadas:
     - R² (R-squared)
     - RMSE (Root Mean Squared Error)
     - MAE (Mean Absolute Error)
     - CV Scores (5-fold cross-validation)
   Salida: evaluation_report.txt
   Ubicación: data/08_reporting/

7. HYPERPARAMETER OPTIMIZATION (hyperparameter_optimization) ★ NUEVO
   Propósito: Optimización de hiperparámetros
   Nodos:
     - optimize_random_forest: GridSearchCV para Random Forest
     - optimize_gradient_boosting: RandomizedSearchCV para GB
     - compare_optimized_models: Comparación de versiones optimizadas
     - generate_optimization_report: Reporte de optimización
   
   Random Forest (GridSearchCV):
     - n_estimators: [50, 100, 200]
     - max_depth: [5, 10, 15, None]
     - min_samples_split: [2, 5, 10]
     - min_samples_leaf: [1, 2, 4]
     - Total combinaciones: 72 (CV 5-fold)
   
   Gradient Boosting (RandomizedSearchCV):
     - n_estimators: [50, 100, 150, 200]
     - learning_rate: [0.01, 0.05, 0.1, 0.15]
     - max_depth: [3, 5, 7, 9]
     - subsample: [0.6, 0.8, 1.0]
     - 20 iteraciones con CV 5-fold
   
   Salida: optimization_report.txt
   Ubicación: data/08_reporting/

================================================================================
CÓMO EJECUTAR LOS PIPELINES:
================================================================================

Ejecutar pipeline por defecto (todos):
  $ kedro run

Ejecutar un pipeline específico:
  $ kedro run --pipeline=data_ingestion
  $ kedro run --pipeline=data_cleaning
  $ kedro run --pipeline=transformation
  $ kedro run --pipeline=validation
  $ kedro run --pipeline=model_training
  $ kedro run --pipeline=model_evaluation
  $ kedro run --pipeline=hyperparameter_optimization

Ejecutar múltiples pipelines en secuencia:
  $ kedro run --pipeline=model_training,model_evaluation,hyperparameter_optimization

Visualizar pipelines:
  $ kedro viz

Listar nodos disponibles:
  $ kedro list

================================================================================
ESTRUCTURA DE DIRECTORIOS:
================================================================================

data/
├── 01_raw/                 # Datos crudos originales
├── 02_intermediate/        # Datos después de limpieza
├── 03_primary/            # Dataset principal después de transformación
├── 04_feature/            # Features intermedias (agregaciones)
├── 05_model_input/        # Datos preparados para modelado
├── 06_models/             # Modelos entrenados y resultados
├── 07_model_output/       # Salida de validación
└── 08_reporting/          # Reportes finales

src/prueba/pipelines/
├── data_ingestion/        # Carga e ingesta
├── data_cleaning/         # Limpieza de datos
├── data_transformation/   # Transformación y agregación
├── data_validation/       # Validación de calidad
├── model_training/        # Entrenamiento de modelos ★ NUEVO
├── model_evaluation/      # Evaluación de modelos ★ NUEVO
└── hyperparameter_optimization/  # Optimización ★ NUEVO

conf/base/
├── catalog.yml            # Definición de datasets
├── parameters.yml         # Parámetros de pipelines
└── logging.yml            # Configuración de logs

================================================================================
PARÁMETROS DE CONFIGURACIÓN:
================================================================================

Se encuentran en: conf/base/parameters.yml

model_training:
  split:
    test_size: 0.2              # Tamaño del conjunto test
    random_state: 42            # Seed para reproducibilidad
    target_column: "puntaje_desempeno"  # Variable objetivo

model_evaluation:
  cv:
    cv_folds: 5                 # Número de folds para validación cruzada

hyperparameter_optimization:
  random_forest:
    cv_folds: 5
  gradient_boosting:
    cv_folds: 5

================================================================================
OUTPUTS GENERADOS:
================================================================================

Reportes de texto:
  - data/08_reporting/diagnostico_inicial.json
  - data/08_reporting/reporte_validacion.txt
  - data/08_reporting/evaluation_report.txt
  - data/08_reporting/optimization_report.txt

Modelos entrenados:
  - results/models/linear_regression_model.pkl
  - results/models/ridge_model.pkl
  - results/models/lasso_model.pkl
  - results/models/random_forest_model.pkl
  - results/models/gradient_boosting_model.pkl
  - results/models/svr_model.pkl
  - results/models/knn_model.pkl
  - results/models/scaler.pkl

Datos procesados:
  - data/03_primary/primary_dataset.csv (309 rows × 14 columns)
  - data/02_intermediate/ (4 archivos limpios)

================================================================================
FLUJO DE EJECUCIÓN RECOMENDADO:
================================================================================

1. Ejecutar ingesta inicial (diagnóstico):
   $ kedro run --pipeline=data_ingestion

2. Ejecutar limpieza de datos:
   $ kedro run --pipeline=data_cleaning

3. Ejecutar transformación y generación de features:
   $ kedro run --pipeline=transformation

4. Ejecutar validación de calidad:
   $ kedro run --pipeline=validation

5. Ejecutar entrenamiento de modelos:
   $ kedro run --pipeline=model_training

6. Ejecutar evaluación de modelos:
   $ kedro run --pipeline=model_evaluation

7. Ejecutar optimización de hiperparámetros:
   $ kedro run --pipeline=hyperparameter_optimization

O alternativamente, ejecutar TODO de una vez:
   $ kedro run

================================================================================
INFORMACIÓN DE CONTACTO / DOCUMENTACIÓN:
================================================================================

Para más información sobre Kedro, visita:
  - https://kedro.readthedocs.io/
  - https://github.com/kedro-org/kedro

Documentación del proyecto:
  - Notebooks de análisis: notebooks/01-05_*.ipynb
  - Módulos de utilidad: src/prueba/

================================================================================
