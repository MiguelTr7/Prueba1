import pandas as pd
import numpy as np
import logging

def clean_hr_data(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Nodo maestro de limpieza para los datasets de Recursos Humanos.
    """
    logger = logging.getLogger(__name__)
    df_cleaned = df.copy()

    # 1. Eliminación de Duplicados
    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    logger.info(f"Removidos {initial_count - len(df_cleaned)} duplicados.")

    # 2. Normalización de Strings y Corrección de Tipos
    # Aplicar strip y lower a todas las columnas de texto
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.upper()

    # 3. Estandarización de Fechas
    # Buscamos columnas que contengan 'fecha' en su nombre
    for col in [c for c in df_cleaned.columns if 'fecha' in c]:
        df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')

    # 4. Tratamiento de Valores Nulos (Imputación por Mediana/Moda)
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().any():
            if df_cleaned[col].dtype in [np.float64, np.int64]:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

    # 5. Tratamiento de Outliers (Método IQR)
    # Solo en columnas numéricas específicas como 'salario' o 'puntaje'
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Capamos los valores (Capping) para no perder filas
        df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
        df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])

    return df_cleaned