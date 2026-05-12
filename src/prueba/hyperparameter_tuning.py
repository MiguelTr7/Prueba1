"""
Modulo de optimizacion de hiperparametros.
Incluye funciones para GridSearchCV y RandomizedSearchCV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time


def optimizar_random_forest(X_train, y_train, cv=5, random_state=42):
    """
    Optimiza hiperparametros de Random Forest usando GridSearchCV.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        cv (int): Numero de folds
        random_state (int): Semilla
        
    Returns:
        dict: Diccionario con mejor modelo y resultados
    """
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Iniciando optimizacion de Random Forest...")
    inicio = time.time()
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    tiempo = time.time() - inicio
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': pd.DataFrame(grid_search.cv_results_),
        'tiempo': tiempo
    }


def optimizar_gradient_boosting(X_train, y_train, cv=5, random_state=42):
    """
    Optimiza hiperparametros de Gradient Boosting usando RandomizedSearchCV.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        cv (int): Numero de folds
        random_state (int): Semilla
        
    Returns:
        dict: Diccionario con mejor modelo y resultados
    """
    
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.6, 0.8, 1.0]
    }
    
    print("Iniciando optimizacion de Gradient Boosting...")
    inicio = time.time()
    
    random_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=random_state),
        param_dist,
        n_iter=20,
        cv=cv,
        scoring='r2',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    tiempo = time.time() - inicio
    
    return {
        'best_model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': pd.DataFrame(random_search.cv_results_),
        'tiempo': tiempo
    }


def comparar_optimizaciones(resultados_rf, resultados_gb):
    """
    Compara resultados de optimizacion entre modelos.
    
    Args:
        resultados_rf (dict): Resultados de Random Forest
        resultados_gb (dict): Resultados de Gradient Boosting
        
    Returns:
        pd.DataFrame: Dataframe comparativo
    """
    
    comparacion = pd.DataFrame({
        'Modelo': ['Random Forest', 'Gradient Boosting'],
        'Mejor Score (R2 CV)': [resultados_rf['best_score'], resultados_gb['best_score']],
        'Tiempo (segundos)': [resultados_rf['tiempo'], resultados_gb['tiempo']]
    })
    
    comparacion = comparacion.sort_values('Mejor Score (R2 CV)', ascending=False)
    
    return comparacion


def crear_reporte_optimizacion(resultados_rf, resultados_gb, ruta_salida='results/reports/optimizacion.txt'):
    """
    Crea reporte en texto con resultados de optimizacion.
    
    Args:
        resultados_rf (dict): Resultados de Random Forest
        resultados_gb (dict): Resultados de Gradient Boosting
        ruta_salida (str): Ruta para guardar el reporte
    """
    
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE DE OPTIMIZACION DE HIPERPARAMETROS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RANDOM FOREST\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mejor Score (R2 CV): {resultados_rf['best_score']:.4f}\n")
        f.write(f"Mejores Parametros:\n")
        for param, valor in resultados_rf['best_params'].items():
            f.write(f"  - {param}: {valor}\n")
        f.write(f"Tiempo de ejecucion: {resultados_rf['tiempo']:.2f} segundos\n\n")
        
        f.write("GRADIENT BOOSTING\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mejor Score (R2 CV): {resultados_gb['best_score']:.4f}\n")
        f.write(f"Mejores Parametros:\n")
        for param, valor in resultados_gb['best_params'].items():
            f.write(f"  - {param}: {valor}\n")
        f.write(f"Tiempo de ejecucion: {resultados_gb['tiempo']:.2f} segundos\n")
    
    print(f"Reporte guardado en: {ruta_salida}")
