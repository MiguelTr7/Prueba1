import pandas as pd

def validate_primary_data(df: pd.DataFrame) -> dict:
    """Verifica integridad y esquemas del dataset final."""
    report = {}
    null_counts = df.isnull().sum().sum()
    report["total_nulls"] = int(null_counts)
    report["is_clean"] = null_counts == 0
    
    expected_cols = ["id_empleado", "puntaje_desempeno", "total_horas_capacitacion"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    report["missing_columns"] = missing_cols
    
    if "salario" in df.columns:
        report["avg_scaled_salary"] = float(df["salario"].mean())
        
    return report

def compare_before_after(df_raw: pd.DataFrame, df_primary: pd.DataFrame) -> str:
    """Compara cantidad de registros antes y después del procesamiento."""
    rows_initial = len(df_raw)
    rows_final = len(df_primary)
    diff = rows_initial - rows_final
    
    report_str = (
        "=== REPORTE DE COMPARACIÓN ===\n"
        f"Registros iniciales (empleados): {rows_initial}\n"
        f"Registros finales (master table): {rows_final}\n"
        f"Diferencia: {diff} registros\n"
    )
    
    if diff == 0:
        report_str += "ESTADO: Integridad referencial mantenida perfectamente.\n"
    else:
        report_str += f"ALERTA: Se perdieron {diff} registros en los Joins.\n"
        
    return report_str

def save_final_report(summary_dict: dict, comparison_str: str) -> str:
    """Combina todo en un solo reporte para el output final."""
    full_report = comparison_str + "\n"
    full_report += "=== INTEGRIDAD DE DATOS ===\n"
    for key, value in summary_dict.items():
        full_report += f"{key}: {value}\n"
    
    return full_report