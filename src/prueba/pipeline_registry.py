from __future__ import annotations
from kedro.pipeline import Pipeline
from prueba.pipelines import data_ingestion as di
from prueba.pipelines import data_cleaning as dc
from prueba.pipelines import data_transformation as dt
from prueba.pipelines import data_validation as dv
from prueba.pipelines import model_training as mt
from prueba.pipelines import model_evaluation as me
from prueba.pipelines import hyperparameter_optimization as ho

def register_pipelines() -> dict[str, Pipeline]:
    ingestion_p = di.create_pipeline()
    cleaning_p = dc.create_pipeline()
    transformation_p = dt.create_pipeline()
    validation_p = dv.create_pipeline()
    model_training_p = mt.create_pipeline()
    model_evaluation_p = me.create_pipeline()
    hyperopt_p = ho.create_pipeline()
    
    return {
        "ingestion": ingestion_p,
        "data_cleaning": cleaning_p,
        "transformation": transformation_p,
        "validation": validation_p,
        "model_training": model_training_p,
        "model_evaluation": model_evaluation_p,
        "hyperparameter_optimization": hyperopt_p,
        "__default__": (
            ingestion_p + cleaning_p + transformation_p + validation_p + 
            model_training_p + model_evaluation_p + hyperopt_p
        ),
    }