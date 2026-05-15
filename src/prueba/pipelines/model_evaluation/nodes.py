"""Nodes para evaluación de modelos de machine learning"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import os
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(prediction_data: dict, params: dict) -> dict:
    """
    Calcula métricas de evaluación (RMSE, MAE, R²) para todos los modelos
    
    Args:
        prediction_data: dict con predicciones
        params: Parámetros de evaluación
    
    Returns:
        dict con métricas de todos los modelos
    """
    predictions = prediction_data['predictions']
    y_test = prediction_data['y_test']
    
    metrics = {}
    
    for model_name, y_pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        logger.info(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'y_test': y_test,
        'models': prediction_data['models']
    }


def cross_validation_scores(
    prediction_data: dict,
    params: dict
) -> dict:
    """
    Calcula scores de validación cruzada para cada modelo
    
    Args:
        prediction_data: dict con datos y modelos
        params: Parámetros de CV
    
    Returns:
        dict con scores de CV
    """
    models = prediction_data['models'] if 'models' in prediction_data else {}
    X_train = prediction_data.get('X_train')
    y_train = prediction_data.get('y_train')
    
    cv_scores = {}
    
    if X_train is not None and y_train is not None and models:
        cv_folds = params.get('cv_folds', 5)
        
        for model_name, model in models.items():
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                cv_scores[model_name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
                logger.info(f"{model_name} - CV Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                logger.warning(f"No se pudo calcular CV para {model_name}: {e}")
    else:
        logger.warning("No hay datos de training disponibles para validación cruzada")
    
    return {
        'cv_scores': cv_scores,
        'metrics': prediction_data.get('metrics', {}),
        'predictions': prediction_data['predictions'],
        'y_test': prediction_data['y_test'],
        'models': prediction_data.get('models', {})
    }


def get_residuals_analysis(evaluation_data: dict) -> dict:
    """
    Analiza los residuos del mejor modelo
    
    Args:
        evaluation_data: dict con evaluación completa
    
    Returns:
        dict con análisis de residuos
    """
    metrics = evaluation_data['metrics']
    predictions = evaluation_data['predictions']
    y_test = evaluation_data['y_test']
    
    # Encontrar mejor modelo por R²
    best_model_name = max(metrics, key=lambda x: metrics[x]['R2'])
    y_pred_best = predictions[best_model_name]
    
    residuals = y_test.values - y_pred_best
    
    residuals_analysis = {
        'best_model': best_model_name,
        'residuals_mean': residuals.mean(),
        'residuals_std': residuals.std(),
        'residuals_min': residuals.min(),
        'residuals_max': residuals.max(),
        'residuals': residuals
    }
    
    logger.info(f"Mejor modelo: {best_model_name}")
    logger.info(f"Residuos - Media: {residuals.mean():.4f}, Std: {residuals.std():.4f}")
    
    return {
        'residuals_analysis': residuals_analysis,
        'best_model_name': best_model_name,
        'models': evaluation_data['models'],
        'cv_scores': evaluation_data.get('cv_scores', {}),
        'metrics': metrics
    }


def generate_evaluation_report(analysis_data: dict, params: dict) -> str:
    """
    Genera un reporte de evaluación completo
    
    Args:
        analysis_data: dict con análisis completo
        params: Parámetros de reporte
    
    Returns:
        str con el reporte
    """
    best_model = analysis_data['best_model_name']
    residuals = analysis_data['residuals_analysis']
    metrics = analysis_data['metrics']
    cv_scores = analysis_data.get('cv_scores', {})
    
    report = f"""
{'='*80}
REPORTE DE EVALUACIÓN DE MODELOS
{'='*80}

MEJOR MODELO: {best_model}

MÉTRICAS DE TEST:
{'-'*80}
"""
    
    for model_name, model_metrics in metrics.items():
        report += f"\n{model_name}:\n"
        report += f"  - R²: {model_metrics['R2']:.4f}\n"
        report += f"  - RMSE: {model_metrics['RMSE']:.4f}\n"
        report += f"  - MAE: {model_metrics['MAE']:.4f}\n"
    
    if cv_scores:
        report += f"\n\nVALIDACIÓN CRUZADA (5-FOLD):\n"
        report += f"{'-'*80}\n"
        
        for model_name, cv in cv_scores.items():
            report += f"\n{model_name}:\n"
            report += f"  - Mean R²: {cv['mean']:.4f}\n"
            report += f"  - Std: {cv['std']:.4f}\n"
    
    report += f"\n\nANÁLISIS DE RESIDUOS ({best_model}):\n"
    report += f"{'-'*80}\n"
    report += f"  - Media: {residuals['residuals_mean']:.4f}\n"
    report += f"  - Desv. Estándar: {residuals['residuals_std']:.4f}\n"
    report += f"  - Mínimo: {residuals['residuals_min']:.4f}\n"
    report += f"  - Máximo: {residuals['residuals_max']:.4f}\n"
    
    report += f"\n{'='*80}\n"
    
    logger.info("Reporte de evaluación generado")
    
    return report
