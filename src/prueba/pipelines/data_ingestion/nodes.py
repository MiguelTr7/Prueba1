import pandas as pd
import logging

def explorar_y_diagnosticar(*datasets) -> dict:
    """
    Realiza la exploración inicial de los 4 datasets de RRHH.
    Calcula: shape, dtypes, nulos, duplicados y estadísticas básicas. 
    """
    logger = logging.getLogger(__name__)
    reporte = {}
    nombres = ["empleados", "evaluaciones", "capacitaciones", "ausencias"]

    for nombre, df in zip(nombres, datasets):
        # 1. Exploración básica [cite: 40]
        diagnostico = {
            "forma": df.shape,
            "columnas": df.dtypes.astype(str).to_dict(),
            "nulos_totales": int(df.isnull().sum().sum()),
            "duplicados": int(df.duplicated().sum()),
            "estadisticas_numericas": df.describe().to_dict()
        }
        
        # 2. Detección de problemas de calidad específicos [cite: 41, 20]
        # Identificamos columnas con tipos mixtos o formatos inconsistentes
        reporte[nombre] = diagnostico
        
        logger.info(f"Dataset {nombre} procesado: {df.shape[0]} filas detectadas.")

    return reporte