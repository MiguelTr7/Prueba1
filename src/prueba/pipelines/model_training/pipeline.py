"""Pipeline para entrenamiento de modelos"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_data,
    scale_features,
    train_regression_models,
    get_predictions,
    save_models
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["primary_dataset", "params:model_training.split"],
            outputs="model_data_split",
            name="split_data_node",
        ),
        node(
            func=scale_features,
            inputs=["model_data_split", "params:model_training.scaling"],
            outputs="model_data_prepared",
            name="scale_features_node",
        ),
        node(
            func=train_regression_models,
            inputs=["model_data_prepared", "params:model_training.models"],
            outputs="model_training_results",
            name="train_models_node",
        ),
        node(
            func=get_predictions,
            inputs="model_training_results",
            outputs="model_predictions",
            name="get_predictions_node",
        ),
        node(
            func=save_models,
            inputs=["model_predictions", "params:model_training.save"],
            outputs="models_directory",
            name="save_models_node",
        ),
    ])
