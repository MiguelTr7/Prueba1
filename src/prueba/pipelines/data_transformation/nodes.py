import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def aggregate_evaluations(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el promedio de desempeño por empleado."""
    return df.groupby("id_empleado").agg({
        "puntaje_desempeno": "mean",
        "competencias_tecnicas": "mean",
        "competencias_blandas": "mean"
    }).reset_index()

def aggregate_ausencias(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula total de días y cantidad de ausencias por empleado."""
    res = df.groupby("id_empleado")["dias"].agg(["sum", "count"]).reset_index()
    res.columns = ["id_empleado", "total_dias_ausencia", "n_ausencias"]
    return res

def aggregate_capacitaciones(df: pd.DataFrame) -> pd.DataFrame:
    """Feature Engineering: Horas totales y promedio de notas."""
    return df.groupby("id_empleado").agg({
        "horas": "sum",
        "nota_final": "mean"
    }).rename(columns={
        "horas": "total_horas_capacitacion", 
        "nota_final": "promedio_notas_cursos"
    }).reset_index()

def create_master_table(empleados: pd.DataFrame, ev: pd.DataFrame, aus: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    """Une todas las tablas (Joins)."""
    master = empleados.merge(ev, on="id_empleado", how="left")
    master = master.merge(aus, on="id_empleado", how="left")
    master = master.merge(cap, on="id_empleado", how="left")
    
    # Solo llenamos con 0 las columnas numéricas que creamos en los pasos anteriores
    metric_cols = [
        "puntaje_desempeno", "competencias_tecnicas", "competencias_blandas",
        "total_dias_ausencia", "n_ausencias", 
        "total_horas_capacitacion", "promedio_notas_cursos"
    ]
    cols_to_fill = [c for c in metric_cols if c in master.columns]
    master[cols_to_fill] = master[cols_to_fill].fillna(0)
    
    return master

def scale_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Normalización y Codificación con limpieza de caracteres especiales."""
    
    # --- PASO CRÍTICO: Limpieza de datos sucios (como el '~') ---
    num_cols = ["salario", "puntaje_desempeno", "total_dias_ausencia", "total_horas_capacitacion"]
    
    for col in num_cols:
        if col in df.columns:
            # Si la columna es texto, quitamos el '~' y cualquier espacio
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('~', '', regex=False).str.strip()
            
            # Convertimos a número. Si hay algo que no se puede convertir, se vuelve NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Llenamos los posibles NaN resultantes con la mediana de la columna
            df[col] = df[col].fillna(df[col].median())

    # 1. Codificación (Variables Categóricas)
    cat_cols = ["departamento", "cargo", "tipo_contrato"]
    # Aseguramos que existan en el DF antes de procesar
    cat_cols_present = [c for c in cat_cols if c in df.columns]
    df[cat_cols_present] = df[cat_cols_present].astype(str)
    df = pd.get_dummies(df, columns=cat_cols_present, drop_first=True)
    
    # 2. Normalización (Escalado)
    scaler = StandardScaler()
    # Solo escalamos las columnas que realmente existan y sean numéricas
    cols_to_scale = [c for c in num_cols if c in df.columns]
    
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df