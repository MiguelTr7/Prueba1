from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_hr_data

def create_pipeline(**kwargs) -> Pipeline:
    datasets = ["empleados", "evaluaciones", "capacitaciones", "ausencias"]
    nodes = []
    
    for ds in datasets:
        nodes.append(
            node(
                func=clean_hr_data,
                inputs=[f"{ds}_raw", "params:cleaning_params"],
                # CAMBIAMOS "{ds}_cleaned" por "int_{ds}" para que coincida con el resto
                outputs=f"int_{ds}", 
                name=f"node_clean_{ds}"
            )
        )
    
    return pipeline(nodes)