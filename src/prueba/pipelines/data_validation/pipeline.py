from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_primary_data, compare_before_after, save_final_report

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=validate_primary_data,
            inputs="primary_dataset",
            outputs="val_summary_dict",
            name="validate_primary_node",
        ),
        node(
            func=compare_before_after,
            # CAMBIAMOS "int_empleados" por "empleados_int"
            inputs=["empleados_int", "primary_dataset"], 
            outputs="val_comparison_text",
            name="compare_before_after_node",
        ),
        node(
            func=save_final_report,
            inputs=["val_summary_dict", "val_comparison_text"],
            outputs="validation_report",
            name="generate_final_report_node",
        ),
    ])