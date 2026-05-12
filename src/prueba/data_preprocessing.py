"""
Modulo de preprocesamiento de datos para el proyecto de prediccion de desempeno.
Incluye funciones para limpieza, transformacion y preparacion de datos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def cargar_datos_crudos(ruta_datos='data/01_raw/'):
    """
    Carga todos los archivos CSV del directorio de datos crudos.
    
    Args:
        ruta_datos (str): Ruta al directorio con archivos CSV
        
    Returns:
        dict: Diccionario con dataframes cargados
    """
    import os
    datos = {}
    
    for archivo in os.listdir(ruta_datos):
        if archivo.endswith('.csv'):
            nombre = archivo.replace('.csv', '')
            ruta_completa = os.path.join(ruta_datos, archivo)
            datos[nombre] = pd.read_csv(ruta_completa)
            print(f"  Cargado: {nombre} ({datos[nombre].shape[0]} filas, {datos[nombre].shape[1]} columnas)")
            
    return datos


def limpiar_datos_basico(df, umbral_nulos=0.3):
    """
    Realiza limpieza basica de datos: elimina columnas con muchos nulos.
    
    Args:
        df (pd.DataFrame): Dataframe a limpiar
        umbral_nulos (float): Proporcion de nulos permitida (default 0.3)
        
    Returns:
        pd.DataFrame: Dataframe limpio
    """
    # Eliminar columnas con mas del umbral de nulos
    df_limpio = df.dropna(thresh=len(df) * (1 - umbral_nulos), axis=1)
    
    # Para columnas numericas: rellenar con la mediana
    cols_numericas = df_limpio.select_dtypes(include=[np.number]).columns
    for col in cols_numericas:
        if df_limpio[col].isnull().sum() > 0:
            df_limpio[col].fillna(df_limpio[col].median(), inplace=True)
    
    # Para columnas categoricas: rellenar con la moda
    cols_categoricas = df_limpio.select_dtypes(include=['object']).columns
    for col in cols_categoricas:
        if df_limpio[col].isnull().sum() > 0:
            df_limpio[col].fillna(df_limpio[col].mode()[0], inplace=True)
    
    return df_limpio


def crear_features_empleado(datos):
    """
    Crea features agregadas a partir de los datos de empleados, ausencias y capacitaciones.
    
    Args:
        datos (dict): Diccionario con dataframes (empleados, ausencias, capacitaciones, evaluaciones)
        
    Returns:
        pd.DataFrame: Dataframe con features de empleado y target (puntaje_desempeno)
    """
    
    # Limpiar datos
    empleados = limpiar_datos_basico(datos['empleados'].copy())
    ausencias = limpiar_datos_basico(datos['ausencias'].copy())
    capacitaciones = limpiar_datos_basico(datos['capacitaciones'].copy())
    evaluaciones = limpiar_datos_basico(datos['evaluaciones'].copy())
    
    # Seleccionar solo evaluaciones con puntaje_desempeno valido
    evaluaciones = evaluaciones[evaluaciones['puntaje_desempeno'].notna()].copy()
    
    # Agregar ausencias por empleado
    ausencias_agg = ausencias.groupby('id_empleado').agg({
        'dias': ['sum', 'mean', 'max'],
        'justificada': lambda x: (x == 'S').sum()
    }).reset_index()
    ausencias_agg.columns = ['id_empleado', 'total_dias_ausencia', 'promedio_dias_ausencia', 
                             'max_dias_ausencia', 'ausencias_justificadas']
    
    # Agregar capacitaciones por empleado
    capacitaciones_agg = capacitaciones.groupby('id_empleado').agg({
        'id_capacitacion': 'count',
        'horas': 'sum',
        'nota_final': ['mean', 'max']
    }).reset_index()
    capacitaciones_agg.columns = ['id_empleado', 'num_capacitaciones', 'total_horas_capacitacion',
                                  'promedio_nota_capacitacion', 'max_nota_capacitacion']
    
    # Agregar evaluaciones (tomar la mas reciente)
    evaluaciones_reciente = evaluaciones.sort_values('periodo', na_position='last').groupby('id_empleado').first().reset_index()
    
    # Merge de todos los datos
    df_final = empleados[['id_empleado', 'departamento', 'cargo', 'tipo_contrato', 'jornada']].copy()
    
    df_final = df_final.merge(ausencias_agg, on='id_empleado', how='left')
    df_final = df_final.merge(capacitaciones_agg, on='id_empleado', how='left')
    df_final = df_final.merge(evaluaciones_reciente[['id_empleado', 'puntaje_desempeno', 
                                                       'competencias_tecnicas', 'competencias_blandas']], 
                             on='id_empleado', how='left')
    
    # Rellenar nulos en features nuevas
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_final[col].isnull().sum() > 0:
            df_final[col].fillna(0, inplace=True)
    
    return df_final


def preparar_datos_para_ml(df_features, test_size=0.2, random_state=42):
    """
    Prepara datos para modelos ML: divide en train/test y normaliza.
    
    Args:
        df_features (pd.DataFrame): Dataframe con features
        test_size (float): Proporcion de test
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Diccionario con X_train, X_test, y_train, y_test, scaler
    """
    
    # Separar target
    y = df_features['puntaje_desempeno'].copy()
    X = df_features.drop('puntaje_desempeno', axis=1).copy()
    
    # Eliminar id_empleado
    if 'id_empleado' in X.columns:
        X = X.drop('id_empleado', axis=1)
    
    # Codificar variables categoricas
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir a DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.reset_index(drop=True),
        'y_test': y_test.reset_index(drop=True),
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns)
    }
