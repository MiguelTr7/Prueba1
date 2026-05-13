"""Pipeline para optimización de hiperparámetros"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    optimize_random_forest,
    optimize_gradient_boosting,
    compare_optimized_models,
    generate_optimization_report
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=optimize_random_forest,
            inputs=["model_data_prepared", "params:hyperparameter_optimization.random_forest"],
            outputs="rf_optimization_results",
            name="optimize_rf_node",
        ),
        node(
            func=optimize_gradient_boosting,
            inputs=["rf_optimization_results", "params:hyperparameter_optimization.gradient_boosting"],
            outputs="gb_optimization_results",
            name="optimize_gb_node",
        ),
        node(
            func=compare_optimized_models,
            inputs="gb_optimization_results",
            outputs="optimization_comparison",
            name="compare_models_node",
        ),
        node(
            func=generate_optimization_report,
            inputs="optimization_comparison",
            outputs="optimization_report_text",
            name="generate_optimization_report_node",
        ),
    ])
