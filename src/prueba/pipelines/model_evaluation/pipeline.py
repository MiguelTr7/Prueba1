"""Pipeline para evaluación de modelos"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    calculate_metrics,
    cross_validation_scores,
    get_residuals_analysis,
    generate_evaluation_report
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=calculate_metrics,
            inputs=["model_predictions", "params:model_evaluation.metrics"],
            outputs="model_metrics_calculated",
            name="calculate_metrics_node",
        ),
        node(
            func=cross_validation_scores,
            inputs=["model_metrics_calculated", "params:model_evaluation.cv"],
            outputs="model_cv_scores",
            name="cross_validation_node",
        ),
        node(
            func=get_residuals_analysis,
            inputs="model_cv_scores",
            outputs="model_residuals_analysis",
            name="residuals_analysis_node",
        ),
        node(
            func=generate_evaluation_report,
            inputs=["model_residuals_analysis", "params:model_evaluation.report"],
            outputs="evaluation_report_text",
            name="generate_report_node",
        ),
    ])
