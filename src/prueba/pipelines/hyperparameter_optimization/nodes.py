"""Nodes para optimización de hiperparámetros"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import logging

logger = logging.getLogger(__name__)

def optimize_random_forest(
    model_prepared_and_predictions: dict,
    params: dict
) -> dict:
    """
    Optimiza hiperparámetros de Random Forest usando GridSearchCV
    
    Args:
        model_prepared_and_predictions: dict con datos preparados y predicciones
        params: Parámetros de optimización
    
    Returns:
        dict con resultados de optimización
    """
    # Obtener datos de ambas fuentes
    X_train = model_prepared_and_predictions.get('X_train')
    y_train = model_prepared_and_predictions.get('y_train')
    X_test = model_prepared_and_predictions.get('X_test')
    y_test = model_prepared_and_predictions.get('y_test')
    
    if X_train is None or y_train is None:
        logger.error("No se encontraron X_train o y_train")
        raise ValueError("Datos de entrenamiento no disponibles")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    start_time = time.time()
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Optimización RF completada en {elapsed_time:.2f}s")
    logger.info(f"Mejor score RF: {grid_search.best_score_:.4f}")
    logger.info(f"Mejores parámetros RF: {grid_search.best_params_}")
    
    return {
        'grid_search_rf': grid_search,
        'best_model_rf': grid_search.best_estimator_,
        'best_params_rf': grid_search.best_params_,
        'best_score_rf': grid_search.best_score_,
        'cv_results_rf': pd.DataFrame(grid_search.cv_results_),
        'time_rf': elapsed_time,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'models': model_prepared_and_predictions.get('models', {})
    }


def optimize_gradient_boosting(
    rf_results: dict,
    params: dict
) -> dict:
    """
    Optimiza hiperparámetros de Gradient Boosting usando RandomizedSearchCV
    
    Args:
        rf_results: dict con resultados de RF
        params: Parámetros de optimización
    
    Returns:
        dict con resultados de optimización
    """
    X_train = rf_results['X_train']
    y_train = rf_results['y_train']
    
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    start_time = time.time()
    random_search = RandomizedSearchCV(
        gb,
        param_dist,
        n_iter=20,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Optimización GB completada en {elapsed_time:.2f}s")
    logger.info(f"Mejor score GB: {random_search.best_score_:.4f}")
    logger.info(f"Mejores parámetros GB: {random_search.best_params_}")
    
    return {
        'random_search_gb': random_search,
        'best_model_gb': random_search.best_estimator_,
        'best_params_gb': random_search.best_params_,
        'best_score_gb': random_search.best_score_,
        'cv_results_gb': pd.DataFrame(random_search.cv_results_),
        'time_gb': elapsed_time,
        'best_model_rf': rf_results['best_model_rf'],
        'best_score_rf': rf_results['best_score_rf'],
        'X_train': X_train,
        'X_test': rf_results['X_test'],
        'y_train': y_train,
        'y_test': rf_results['y_test']
    }


def compare_optimized_models(optimization_data: dict) -> dict:
    """
    Compara los modelos optimizados
    
    Args:
        optimization_data: dict con resultados de optimización
    
    Returns:
        dict con comparación
    """
    best_model_rf = optimization_data['best_model_rf']
    best_model_gb = optimization_data['best_model_gb']
    
    X_test = optimization_data['X_test']
    y_test = optimization_data['y_test']
    
    # Predicciones
    y_pred_rf = best_model_rf.predict(X_test)
    y_pred_gb = best_model_gb.predict(X_test)
    
    # Métricas test
    r2_rf_test = r2_score(y_test, y_pred_rf)
    r2_gb_test = r2_score(y_test, y_pred_gb)
    rmse_rf_test = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rmse_gb_test = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    
    # Mejora vs CV
    mejora_rf = ((r2_rf_test - optimization_data['best_score_rf']) / abs(optimization_data['best_score_rf'])) * 100
    mejora_gb = ((r2_gb_test - optimization_data['best_score_gb']) / abs(optimization_data['best_score_gb'])) * 100
    
    comparison = {
        'Random Forest': {
            'CV_R2': optimization_data['best_score_rf'],
            'Test_R2': r2_rf_test,
            'Test_RMSE': rmse_rf_test,
            'Mejora': mejora_rf
        },
        'Gradient Boosting': {
            'CV_R2': optimization_data['best_score_gb'],
            'Test_R2': r2_gb_test,
            'Test_RMSE': rmse_gb_test,
            'Mejora': mejora_gb
        }
    }
    
    # Mejor modelo global
    best_global = max(
        [('Random Forest', r2_rf_test), ('Gradient Boosting', r2_gb_test)],
        key=lambda x: x[1]
    )
    
    logger.info(f"Comparación: Mejor modelo global: {best_global[0]} (R²: {best_global[1]:.4f})")
    
    return {
        'comparison': comparison,
        'best_model_global': best_global[0],
        'best_r2_global': best_global[1],
        'models_optimized': {
            'Random Forest': best_model_rf,
            'Gradient Boosting': best_model_gb
        }
    }


def generate_optimization_report(comparison_data: dict) -> str:
    """
    Genera reporte de optimización
    
    Args:
        comparison_data: dict con comparación de modelos
    
    Returns:
        str con el reporte
    """
    comparison = comparison_data['comparison']
    best_model = comparison_data['best_model_global']
    
    report = f"""
{'='*80}
REPORTE DE OPTIMIZACIÓN DE HIPERPARÁMETROS
{'='*80}

MEJOR MODELO GLOBAL: {best_model}
R² Score: {comparison_data['best_r2_global']:.4f}

{'='*80}
COMPARACIÓN DE MODELOS OPTIMIZADOS:
{'='*80}

"""
    
    for model_name, metrics in comparison.items():
        report += f"\n{model_name}:\n"
        report += f"  - CV R² (Validación Cruzada): {metrics['CV_R2']:.4f}\n"
        report += f"  - Test R²: {metrics['Test_R2']:.4f}\n"
        report += f"  - Test RMSE: {metrics['Test_RMSE']:.4f}\n"
        report += f"  - Mejora: {metrics['Mejora']:.2f}%\n"
    
    report += f"\n{'='*80}\n"
    report += "RECOMENDACIÓN:\n"
    report += f"Usar modelo: {best_model}\n"
    report += "Reentrenar regularmente con nuevos datos\n"
    report += f"\n{'='*80}\n"
    
    logger.info("Reporte de optimización generado")
    
    return report
