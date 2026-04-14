from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_hr_data

# src/prueba/pipelines/data_cleaning/pipeline.py

def create_pipeline(**kwargs) -> Pipeline:
    datasets = ["empleados", "evaluaciones", "capacitaciones", "ausencias"]
    nodes = []
    
    for ds in datasets:
        nodes.append(
            node(
                func=clean_hr_data,
                inputs=[f"{ds}_raw", "params:cleaning_params"],
                # CAMBIAMOS de f"int_{ds}" a f"{ds}_int"
                # Ahora sí coincide con: empleados_int, evaluaciones_int, etc.
                outputs=f"{ds}_int", 
                name=f"node_clean_{ds}"
            )
        )
    
    return pipeline(nodes)