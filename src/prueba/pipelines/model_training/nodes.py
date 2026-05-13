"""Nodes para entrenamiento de modelos de machine learning"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle
import os
import logging

logger = logging.getLogger(__name__)

def split_data(
    dataset: pd.DataFrame,
    params: dict
) -> dict:
    """
    Divide los datos en train y test
    
    Args:
        dataset: DataFrame con los datos preparados
        params: Parámetros de split (test_size, random_state)
    
    Returns:
        dict con X_train, X_test, y_train, y_test, feature_names
    """
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)
    target_column = params.get('target_column', 'puntaje_desempeno')
    
    # Separar features y target
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    logger.info(f"Datos divididos: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X.columns)
    }


def scale_features(data_split: dict, params: dict) -> dict:
    """
    Normaliza las características usando StandardScaler
    
    Args:
        data_split: dict con datos divididos
        params: Parámetros de escalado
    
    Returns:
        dict con datos escalados
    """
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(data_split['X_train'])
    X_test_scaled = scaler.transform(data_split['X_test'])
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=data_split['feature_names'])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=data_split['feature_names'])
    
    logger.info("Características escaladas exitosamente")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': data_split['y_train'],
        'y_test': data_split['y_test'],
        'feature_names': data_split['feature_names'],
        'scaler': scaler
    }


def train_regression_models(
    data_prepared: dict,
    params: dict
) -> dict:
    """
    Entrena 7 modelos de regresión diferentes
    
    Args:
        data_prepared: dict con datos preparados y escalados
        params: Parámetros de modelos
    
    Returns:
        dict con modelos entrenados
    """
    X_train = data_prepared['X_train']
    y_train = data_prepared['y_train']
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, epsilon=0.1),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"Modelo entrenado: {name}")
    
    logger.info(f"Se entrenaron {len(trained_models)} modelos exitosamente")
    
    return {
        'models': trained_models,
        'X_train': X_train,
        'X_test': data_prepared['X_test'],
        'y_train': y_train,
        'y_test': data_prepared['y_test'],
        'feature_names': data_prepared['feature_names'],
        'scaler': data_prepared['scaler']
    }


def get_predictions(trained_data: dict) -> dict:
    """
    Obtiene predicciones de todos los modelos
    
    Args:
        trained_data: dict con modelos entrenados
    
    Returns:
        dict con predicciones
    """
    models = trained_data['models']
    X_test = trained_data['X_test']
    
    predictions = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
    
    logger.info(f"Predicciones generadas para {len(predictions)} modelos")
    
    return {
        'predictions': predictions,
        'y_test': trained_data['y_test'],
        'models': models,
        'feature_names': trained_data['feature_names'],
        'scaler': trained_data['scaler']
    }


def save_models(prediction_data: dict, params: dict) -> str:
    """
    Guarda los modelos entrenados
    
    Args:
        prediction_data: dict con modelos y predicciones
        params: Parámetros de guardado
    
    Returns:
        str con ruta de guardado
    """
    models_dir = params.get('models_dir', 'results/models')
    os.makedirs(models_dir, exist_ok=True)
    
    models = prediction_data['models']
    
    for name, model in models.items():
        safe_name = name.replace(' ', '_').lower()
        filepath = os.path.join(models_dir, f"{safe_name}_model.pkl")
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Modelo guardado: {filepath}")
    
    # Guardar scaler también
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(prediction_data['scaler'], f)
    
    logger.info(f"Scaler guardado: {scaler_path}")
    
    return models_dir
