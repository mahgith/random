from typing import Dict, Any
from kfp import dsl
from components.data.ingest.csv_to_bigquery import ingest_csv_to_bigquery
from components.data.preprocess.clean_and_merge_tables import clean_and_merge_tables
from components.data.featurize.build_training_dataset import build_training_dataset
from components.ops.monitor.data_drift import data_drift_evaluation
# from components.ml.train.model import train_xgboost_model

@dsl.pipeline(
    name="Preprocessing volume forecast",
    description="Pipeline to preprocess volume raw data"
)
def my_first_pipeline(
        infra: Dict[str, Any],
        params_ingestion_shipments: Dict[str, Any],
        params_ingestion_events: Dict[str, Any],
        params_clean_and_merge_tables: Dict[str, Any],
        params_build_training_dataset: Dict[str, Any],
        # params_train_xgboost_model: Dict[str, Any]
):
    
    # Task for Shipments Ingestion
    ingest_shipments_task = ingest_csv_to_bigquery(
        infra=infra,
        job_params=params_ingestion_shipments
    )
    ingest_shipments_task.set_display_name("Ingest of shipments") # type: ignore

    # Task for Events Ingestion
    ingest_events_task = ingest_csv_to_bigquery(
        infra=infra,
        job_params=params_ingestion_events
    ) # type: ignore
    ingest_events_task.set_display_name("Ingest of events") # type: ignore

    # Task for merging and cleaning data
    clean_task = clean_and_merge_tables(
        infra=infra,
        job_params=params_clean_and_merge_tables
    ) # type: ignore
    clean_task.set_display_name("Clean and Merge Data")
    clean_task.after(ingest_shipments_task, ingest_events_task)       

    # Task for feature engineering
    build_dataset_task = build_training_dataset(
        processed_data=clean_task.outputs["output_dataset"],
        infra=infra,
        job_params=params_build_training_dataset
    ) # type: ignore
    build_dataset_task.set_display_name("Feature engineering")
    build_dataset_task.after(clean_task)  

    # Task for feature engineering
    data_drift_evaluation_task = data_drift_evaluation(
        input_data=build_dataset_task.outputs["training_dataset"],
    ) # type: ignore
    data_drift_evaluation_task.set_display_name("Data Drift Evaluation")
    data_drift_evaluation_task.after(build_dataset_task)    

    # train_task = train_xgboost_model(
    #     training_data=build_dataset_task.outputs["training_dataset"],
    #     infra=infra,
    #     job_params=params_train_xgboost_model
    # ) # type: ignore
    # train_task.set_display_name("Train XGBoost Model")

