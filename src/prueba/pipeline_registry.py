from __future__ import annotations
from kedro.pipeline import Pipeline
from prueba.pipelines import data_ingestion as di
from prueba.pipelines import data_cleaning as dc
from prueba.pipelines import data_transformation as dt # <--- Nuevo
from prueba.pipelines import data_validation as dv # <--- Nuevo

def register_pipelines() -> dict[str, Pipeline]:
    ingestion_p = di.create_pipeline()
    cleaning_p = dc.create_pipeline()
    transformation_p = dt.create_pipeline()
    validation_p = dv.create_pipeline()
    
    return {
        "ingestion": ingestion_p,
        "data_cleaning": cleaning_p,      # <--- Cambiado de "cleaning" a "data_cleaning"
        "transformation": transformation_p,
        "validation": validation_p,
        "__default__": ingestion_p + cleaning_p + transformation_p + validation_p,
    }