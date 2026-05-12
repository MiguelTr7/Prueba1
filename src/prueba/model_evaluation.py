"""
Modulo de evaluacion y comparacion de modelos.
Incluye funciones para calcular metricas y comparar rendimiento.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error, cross_val_score
)
from sklearn.model_selection import cross_validate


def evaluar_modelos_regresion(modelos, X_train, X_test, y_train, y_test, cv=5):
    """
    Evalua multiples modelos de regresion con metricas estándar.
    
    Args:
        modelos (dict): Diccionario con modelos entrenados
        X_train, X_test: Features de train y test
        y_train, y_test: Targets de train y test
        cv (int): Numero de folds para validacion cruzada
        
    Returns:
        pd.DataFrame: Dataframe con metricas de todos los modelos
    """
    
    resultados = []
    
    for nombre_modelo, modelo in modelos.items():
        
        # Predicciones
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # Metricas en train
        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        
        # Metricas en test
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
        
        # Validacion cruzada
        scoring = {
            'r2': 'r2',
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'neg_mean_absolute_error': 'neg_mean_absolute_error'
        }
        cv_results = cross_validate(modelo, X_train, y_train, cv=cv, scoring=scoring)
        
        r2_cv = cv_results['test_r2'].mean()
        rmse_cv = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
        mae_cv = -cv_results['test_neg_mean_absolute_error'].mean()
        
        resultados.append({
            'Modelo': nombre_modelo,
            'RMSE_Train': rmse_train,
            'RMSE_Test': rmse_test,
            'RMSE_CV': rmse_cv,
            'MAE_Train': mae_train,
            'MAE_Test': mae_test,
            'MAE_CV': mae_cv,
            'R2_Train': r2_train,
            'R2_Test': r2_test,
            'R2_CV': r2_cv,
            'MAPE_Test': mape_test,
            'Overfitting': r2_train - r2_test
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    # Ordenar por R2_Test descendente
    df_resultados = df_resultados.sort_values('R2_Test', ascending=False).reset_index(drop=True)
    
    return df_resultados


def obtener_residuos(modelos, X_test, y_test):
    """
    Calcula residuos para cada modelo.
    
    Args:
        modelos (dict): Diccionario con modelos entrenados
        X_test: Features de test
        y_test: Target de test
        
    Returns:
        dict: Diccionario con residuos de cada modelo
    """
    
    residuos = {}
    
    for nombre_modelo, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        residuos[nombre_modelo] = y_test.values - y_pred
    
    return residuos


def comparar_predicciones(modelos, X_test, y_test, num_muestras=10):
    """
    Compara predicciones de multiples modelos para las primeras n muestras.
    
    Args:
        modelos (dict): Diccionario con modelos entrenados
        X_test: Features de test
        y_test: Target de test
        num_muestras (int): Numero de muestras a mostrar
        
    Returns:
        pd.DataFrame: Dataframe con comparacion de predicciones
    """
    
    df_comparacion = pd.DataFrame({'y_real': y_test.iloc[:num_muestras].values})
    
    for nombre_modelo, modelo in modelos.items():
        y_pred = modelo.predict(X_test.iloc[:num_muestras])
        df_comparacion[f'{nombre_modelo}_pred'] = y_pred
        df_comparacion[f'{nombre_modelo}_error'] = np.abs(y_pred - df_comparacion['y_real'].values)
    
    return df_comparacion


def crear_reporte_metricas(metricas_df, ruta_salida='results/metrics/metricas_modelos.csv'):
    """
    Guarda reporte de metricas en archivo CSV.
    
    Args:
        metricas_df (pd.DataFrame): Dataframe con metricas
        ruta_salida (str): Ruta para guardar el archivo
    """
    
    metricas_df.to_csv(ruta_salida, index=False)
    print(f"Reporte guardado en: {ruta_salida}")
