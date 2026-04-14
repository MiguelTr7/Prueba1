from __future__ import annotations
from kedro.pipeline import Pipeline
from prueba.pipelines import data_ingestion as di
from prueba.pipelines import data_cleaning as dc
from prueba.pipelines import data_transformation as dt # <--- Nuevo
from prueba.pipelines import data_validation as dv # <--- Nuevo

def register_pipelines() -> dict[str, Pipeline]:
    ingestion_p = di.create_pipeline()
    cleaning_p = dc.create_pipeline()
    transformation_p = dt.create_pipeline() # <--- Nuevo
    validation_p = dv.create_pipeline() # <--- Nuevo
    return {
        "ingestion": ingestion_p,
        "cleaning": cleaning_p,
        "transformation": transformation_p, # <--- Nuevo
        "validation": validation_p, # <--- Nuevo
        "__default__": ingestion_p + cleaning_p + transformation_p+ validation_p, # <--- Nuevo
    }