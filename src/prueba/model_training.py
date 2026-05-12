"""
Modulo de entrenamiento de modelos supervisados.
Incluye modelos de regresion y clasificacion con Scikit-learn.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


def entrenar_modelos_regresion(X_train, y_train, random_state=42):
    """
    Entrena multiples modelos de regresion.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Diccionario con modelos entrenados
    """
    
    modelos = {}
    
    # 1. Regresion Lineal
    modelos['Linear Regression'] = LinearRegression()
    modelos['Linear Regression'].fit(X_train, y_train)
    
    # 2. Ridge Regression
    modelos['Ridge'] = Ridge(alpha=1.0, random_state=random_state)
    modelos['Ridge'].fit(X_train, y_train)
    
    # 3. Lasso Regression
    modelos['Lasso'] = Lasso(alpha=0.1, random_state=random_state)
    modelos['Lasso'].fit(X_train, y_train)
    
    # 4. Random Forest Regressor
    modelos['Random Forest'] = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    modelos['Random Forest'].fit(X_train, y_train)
    
    # 5. Gradient Boosting Regressor
    modelos['Gradient Boosting'] = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state
    )
    modelos['Gradient Boosting'].fit(X_train, y_train)
    
    # 6. Support Vector Regressor
    modelos['SVR'] = SVR(kernel='rbf', C=100)
    modelos['SVR'].fit(X_train, y_train)
    
    # 7. K-Neighbors Regressor
    modelos['KNN'] = KNeighborsRegressor(n_neighbors=5)
    modelos['KNN'].fit(X_train, y_train)
    
    return modelos


def obtener_predicciones(modelos, X_data):
    """
    Obtiene predicciones de todos los modelos para un conjunto de datos.
    
    Args:
        modelos (dict): Diccionario con modelos entrenados
        X_data (pd.DataFrame): Features para predecir
        
    Returns:
        dict: Diccionario con predicciones de cada modelo
    """
    
    predicciones = {}
    
    for nombre_modelo, modelo in modelos.items():
        predicciones[nombre_modelo] = modelo.predict(X_data)
    
    return predicciones


def obtener_features_importance(modelos):
    """
    Obtiene la importancia de features para modelos basados en arboles.
    
    Args:
        modelos (dict): Diccionario con modelos entrenados
        
    Returns:
        dict: Diccionario con importancia de features por modelo
    """
    
    importance_dict = {}
    
    modelos_con_importancia = ['Random Forest', 'Gradient Boosting']
    
    for modelo_name in modelos_con_importancia:
        if modelo_name in modelos:
            modelo = modelos[modelo_name]
            if hasattr(modelo, 'feature_importances_'):
                importance_dict[modelo_name] = modelo.feature_importances_
    
    return importance_dict
