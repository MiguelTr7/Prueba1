from kedro.pipeline import Pipeline, node, pipeline
from .nodes import explorar_y_diagnosticar

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=explorar_y_diagnosticar,
            inputs=[
                "empleados_raw", 
                "evaluaciones_raw", 
                "capacitaciones_raw", 
                "ausencias_raw"
            ],
            outputs="reporte_diagnostico_inicial",
            name="nodo_exploracion_inicial_rrhh",
        ),
    ])