from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    aggregate_evaluations, 
    aggregate_ausencias, 
    aggregate_capacitaciones, 
    create_master_table,
    scale_and_encode
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # 1. Agregamos las evaluaciones por empleado
        node(
            func=aggregate_evaluations,
            inputs="int_evaluaciones",
            outputs="prm_evaluaciones_agg",
            name="agg_evaluaciones_node",
        ),
        # 2. Agregamos las ausencias por empleado
        node(
            func=aggregate_ausencias,
            inputs="int_ausencias",
            outputs="prm_ausencias_agg",
            name="agg_ausencias_node",
        ),
        # 3. Agregamos las capacitaciones por empleado
        node(
            func=aggregate_capacitaciones,
            inputs="int_capacitaciones",
            outputs="prm_capacitaciones_agg",
            name="agg_capacitaciones_node",
        ),
        # 4. Unimos todo con la tabla de empleados limpia
        node(
            func=create_master_table,
            inputs=[
                "int_empleados", 
                "prm_evaluaciones_agg", 
                "prm_ausencias_agg", 
                "prm_capacitaciones_agg"
            ],
            outputs="prm_master_table_raw",
            name="join_tables_node",
        ),
        # 5. Escalado y codificación final
        node(
            func=scale_and_encode,
            inputs="prm_master_table_raw",
            outputs="primary_dataset",
            name="final_preprocessing_node",
        ),
    ])