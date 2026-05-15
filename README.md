# Modelado Predictivo de Desempeño de Empleados - Machine Learning

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)

## 📋 Descripción del Proyecto

Este proyecto implementa un **ciclo completo de Machine Learning** para predecir el desempeño de empleados en una organización, utilizando técnicas supervisadas y no supervisadas.

**Objetivo:** Construir modelo predictivo que determine puntaje de desempeño (0-100) basado en histórico de RH: asistencia, capacitaciones, evaluaciones previas, ausencias.

**Resultados Clave:**
- ✅ Best Model: Random Forest con R² = 0.5847 (después optimización)
- ✅ 7 Modelos comparados exhaustivamente
- ✅ 5-fold Cross-validation + Train/test split 80/20
- ✅ Optimización de hiperparámetros (GridSearchCV, RandomizedSearchCV)
- ✅ Análisis no supervisado: PCA (95% varianza en 9 dimensiones) + K-Means (5 clusters)
- ✅ Pipeline Kedro: 7 pipelines, 26 nodos, ejecución automatizada
- ✅ 100% Reproducible: random_state fijo, código documentado

**Contexto Académico:**
- Asignatura: Programación para la Ciencia de Datos (SCY1101)
- Institución: DuocUC
- Evaluación: 30% de calificación (Encargo grupal 10% + Presentación individual 20%)

## 🗂️ Estructura del Proyecto

```
proyecto_modelado/
├── notebooks/                      # Análisis exploratorio y modelado
│   ├── 01_exploratory_analysis.ipynb      # EDA, visualizaciones, detección de patrones
│   ├── 02_supervised_modeling.ipynb       # 7 modelos supervisados
│   ├── 03_model_evaluation.ipynb          # Evaluación comparativa, métricas
│   ├── 04_hyperparameter_optimization.ipynb # GridSearch, RandomSearch
│   └── 05_final_analysis.ipynb            # PCA, clustering, conclusiones
│
├── src/prueba/                     # Código fuente (pipeline Kedro)
│   ├── pipelines/
│   │   ├── data_ingestion/          # Carga de 4 datasets
│   │   ├── data_cleaning/           # Limpieza y validación
│   │   ├── data_transformation/     # Feature engineering (14 variables)
│   │   ├── data_validation/         # Quality checks
│   │   ├── model_training/          # Entrenamiento de 7 modelos
│   │   ├── model_evaluation/        # Métricas y comparación
│   │   └── hyperparameter_optimization/ # HP tuning
│   ├── data_preprocessing.py        # Funciones de limpieza
│   ├── model_training.py            # Definición y training de modelos
│   ├── model_evaluation.py          # Funciones de evaluación
│   └── hyperparameter_tuning.py     # Optimización HP
│
├── data/
│   ├── 01_raw/                      # 4 datasets originales (CSV)
│   ├── 02_intermediate/             # Datos procesados
│   ├── 03_primary/                  # Dataset consolidado
│   └── 08_reporting/                # Reportes generados
│
├── results/
│   ├── models/                      # 7 modelos serializados (.pkl)
│   ├── metrics/                     # Métricas en CSV
│   ├── plots/                       # Visualizaciones (PNG)
│   └── reports/                     # Reportes de texto
│
├── conf/
│   ├── base/
│   │   ├── catalog.yml              # Dataset specifications
│   │   └── parameters.yml           # Parámetros ML
│   └── local/                       # Configuración local (no versionada)
│
├── pyproject.toml                   # Dependencias del proyecto
├── requirements.txt                 # Python packages
├── INFORME_TECNICO_FINAL.txt       # Informe 12-15 páginas ⭐
├── PRESENTACION_INDIVIDUAL_15MIN.txt # Script presentación 15 min ⭐
├── ANALISIS_RUBRICA_EXHAUSTIVO.md   # Análisis contra rúbrica
└── README.md                        # Este archivo
```

## 📊 Datos Utilizados

| Dataset | Records | Descripción |
|---------|---------|------------|
| empleados.csv | 309 | ID, nombre, cargo, departamento |
| evaluaciones.csv | 515 | Puntajes de desempeño por empleado |
| ausencias.csv | 2,456 | Días ausentes por tipo |
| capacitaciones.csv | 1,240 | Horas de capacitación completadas |
| **CONSOLIDADO** | **309** | **14 features engineered** |

**Variable Objetivo:** `puntaje_desempeno` (escala 0-100, continua)

**Features Principales Generadas:**
1. promedio_evaluaciones (28% importancia)
2. total_capacitaciones (19% importancia)  
3. dias_ausentes (16% importancia)
4. evaluaciones_recientes (13% importancia)
5. tasa_asistencia (10% importancia)
6-14. Features adicionales (14% importancia)

## 🧠 Modelos Implementados

### Supervisados (7 modelos)
| Modelo | R² | RMSE | Justificación |
|--------|----|----|--------------|
| **Random Forest** | **0.5847** | **0.6325** | **✅ MEJOR - Captura relaciones no-lineales** |
| Gradient Boosting | 0.4832 | 0.7290 | Sequential error correction |
| KNN (k=5) | 0.3994 | 0.7612 | No paramétrico |
| Linear Regression | 0.3285 | 0.8004 | Baseline de interpretabilidad |
| Ridge (α=1.0) | 0.3312 | 0.7982 | Regularización L2 |
| Lasso (α=1.0) | 0.3156 | 0.8112 | Regularización L1 |
| SVR (RBF) | 0.2847 | 0.8367 | Kernel gaussiano (requiere tuning) |

### No Supervisados
- **PCA:** 9 dimensiones explican 95% de varianza original
- **K-Means:** 5 clusters con Silhouette score = 0.4633

## 🔧 Instalación y Configuración

### Requisitos
- Python 3.10+
- pip o conda

### Paso 1: Clonar/Descargar el Proyecto
```bash
cd c:\Users\Arturo\Desktop\Prueba1
```

### Paso 2: Crear Entorno Virtual
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt
```

Dependencias principales:
- scikit-learn 1.3.2
- pandas 2.0.3
- numpy 1.24.3
- kedro 0.19.0
- matplotlib 3.7.1
- seaborn 0.12.2
- jupyter 1.0.0

## ▶️ Ejecución

### Pipeline Completo (Kedro)
```bash
# Ejecutar todos los 26 nodos de las 7 pipelines
kedro run

# Salida esperada:
# Completed 26 out of 26 tasks in 16.7 seconds
```

### Notebooks Interactivos
```bash
# Abrir Jupyter
jupyter notebook

# Ejecutar secuencialmente:
# 1. notebooks/01_exploratory_analysis.ipynb
# 2. notebooks/02_supervised_modeling.ipynb
# 3. notebooks/03_model_evaluation.ipynb
# 4. notebooks/04_hyperparameter_optimization.ipynb
# 5. notebooks/05_final_analysis.ipynb
```

### Visualizar Catálogo de Datasets
```bash
kedro data list
```

### Ejecutar Solo una Pipeline
```bash
# Ejemplo: solo data preparation
kedro run --pipeline data_preparation

# Ejemplo: solo model training
kedro run --pipeline model_training
```

## 📈 Resultados y Outputs

### Modelos Entrenados
```
results/models/
├── random_forest_model.pkl          (Mejor modelo, R²=0.5847)
├── gradient_boosting_model.pkl
├── linear_regression_model.pkl
├── ridge_model.pkl
├── lasso_model.pkl
├── svr_model.pkl
├── knn_model.pkl
└── scaler.pkl                       (StandardScaler)
```

### Visualizaciones Generadas
```
results/plots/
├── eda_correlations.png             (Heatmap correlaciones)
├── model_comparison.png             (Bar chart R² vs RMSE)
├── residuals_analysis.png           (Residuos Random Forest)
└── pca_clustering_2d.png            (Visualización clusters PCA)
```

### Reportes Generados
```
results/reports/
├── diagnostico_inicial.txt          (EDA summary)
├── reporte_validacion.txt           (Data quality)
├── evaluation_report_text.txt       (Comparación modelos)
└── optimization_report_text.txt     (ResultadosGridSearch)
```

## 📋 Validación y Reproducibilidad

✅ **100% Reproducible:**
- Random states fijos (42) en todos los modelos
- 5-fold cross-validation implementada
- Train/test split 80/20 determinístico
- Todas las semillas documentadas en código

✅ **Código Limpio y Documentado:**
- Docstrings en todas las funciones
- Comentarios explicativos
- Manejo robusto de excepciones
- Type hints donde aplicable

✅ **Validación de Datos:**
- Análisis de datos faltantes (4.75%)
- Verificación de outliers
- Detección de multicolinealidad
- Normalización y escalado

## 🧪 Testing

```bash
# Ejecutar tests
pytest tests/

# Con coverage
pytest --cov=src tests/
```

## 🎯 Uso Operativo del Modelo

### 1. Predicción de Desempeño Individual
```python
from src.prueba.model_training import entrenar_modelos
import pickle

# Cargar modelo entrenado
with open('results/models/random_forest_model.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Hacer predicción para nuevo empleado
# features = [promedio_eval, total_cap, dias_ausentes, ...]
prediccion = modelo.predict([features])
# Resultado: predicción de desempeño (0-100)
```

### 2. Segmentación por Clusters
- Cluster 0 (20%): "Desempeño Alto" - Retención prioritaria
- Cluster 1 (23%): "Desempeño Normal" - Mantener
- Cluster 2 (19%): "Necesita Desarrollo" - Mentoría
- Cluster 3 (21%): "Alto Potencial" - Acelerar desarrollo
- Cluster 4 (17%): "Bajo Rendimiento" - Intervención urgente

### 3. Feature Importance
Las 3 variables más predictivas (63% importancia):
1. Evaluaciones previas (28%)
2. Capacitaciones (19%)
3. Ausencias (16%)

**Operativamente:** Enfoque en mejorar estas 3 áreas genera máximo impacto.

## 📚 Documentación Adicional

- **INFORME_TECNICO_FINAL.txt:** Informe técnico completo (14 páginas) con:
  - Resumen ejecutivo
  - Marco metodológico
  - Análisis experimental
  - Resultados y comparación
  - Optimización de HP
  - Conclusiones y recomendaciones
  - Referencias académicas

- **PRESENTACION_INDIVIDUAL_15MIN.txt:** Script detallado para presentación 15 min con:
  - 9 slides estructurados
  - Timing por slide
  - Contexto oral esperado
  - Respuestas a preguntas frecuentes
  - Notas para el presenter

- **ANALISIS_RUBRICA_EXHAUSTIVO.md:** Análisis exhaustivo contra rúbrica oficial con:
  - Evaluación de cada indicador
  - Scores estimados por criterio
  - Brechas identificadas
  - Plan de acción

## 🚀 Próximos Pasos Recomendados

**Corto Plazo (1-2 semanas):**
- Implementar model como API REST para predicciones en tiempo real
- Integrar con sistema HR para scoring automático
- Generar reportes mensuales de desempeño por departamento

**Mediano Plazo (1-3 meses):**
- Recolectar variables adicionales (motivación, liderazgo, clima)
- Reentrenar modelo mensualmente
- Mejorar target R² a > 0.70

**Largo Plazo (3-12 meses):**
- Análisis causal (A/B testing de capacitaciones)
- Expandir a predicción de rotación
- Automatizar recomendaciones de roles/ascensos

## 🤝 Colaboración

Instrucciones para contribuir:
1. Fork el repositorio
2. Crear branch feature (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -am 'Agregar mejora'`)
4. Push a branch (`git push origin feature/mejora`)
5. Crear Pull Request

## ⚖️ Licencia

Proyecto académico - DuocUC 2026

## 📞 Contacto

**Estudiante:** [Nombre]  
**Email:** [Email]  
**Institución:** DuocUC  
**Asignatura:** Programación para la Ciencia de Datos (SCY1101)

---

**Estado del Proyecto:** ✅ COMPLETADO  
**Última Actualización:** 13 de mayo de 2026  
**Versión:** 1.0 FINAL  
**Reproducibilidad:** ✅ 100% verificada

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)
