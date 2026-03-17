from typing import Dict, Any
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# Docker image configuration
ML_IMAGE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/ml-training:latest"

@dsl.component(base_image=ML_IMAGE)
def train_xgboost_model(
    training_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    infra: Dict[str, Any],
    job_params: Dict[str, Any],
) -> None:
    """
    Trains an XGBoost binary classifier using preprocessed training data stored in BigQuery 
    and registers the resulting model in Vertex AI Model Registry when performance thresholds 
    are met.

    This component loads the training dataset referenced by the KFP `Dataset` artifact, 
    performs preprocessing (column filtering, one‑hot encoding, and leakage‑prevention cleanup), 
    splits the data into training and test subsets, and trains an XGBoost model with class 
    imbalance adjustment (`scale_pos_weight`). After training, it evaluates the model using 
    accuracy, F1 Score, recall, and precision, and logs all metrics to the KFP `Metrics` artifact 
    along with a confusion matrix stored as metadata.

    If the model’s F1 score and recall meet the defined thresholds, the component serializes the 
    trained model using Joblib, saves the feature list for inference, and uploads the model to 
    Vertex AI Model Registry using the standard XGBoost serving container.

    Args:
        training_data (Input[Dataset]): 
            KFP dataset artifact containing:
            - table_id (str): Fully‑qualified BigQuery table used as training input.
        infra (Dict[str, Any]): 
            Global infrastructure configuration.
            - project_id (str): GCP Project where BigQuery and Vertex AI operations run.
            - location (str): Region for BigQuery operations and model registry.
        job_params (Dict[str, Any]): 
            Training hyperparameters and configuration/Business parameters.
            - n_estimators (int): Number of boosting rounds.
            - learning_rate (float): Step size shrinkage.
            - max_depth (int): Maximum tree depth for base learners.

        Outputs:
            model_artifact (Output[Model]): 
                Artifact storing:
                - the serialized Joblib model inside the artifact directory.
                - metadata indicating model quality and registration status.
                - optional resource name if the model is registered in Vertex AI.
            metrics_artifact (Output[Metrics]): 
                Artifact capturing:
                - evaluation metrics (accuracy, F1, precision, recall).
                - confusion_matrix (list[list[int]]): Serialized confusion matrix.

    Returns:
        None.
    """
    import os
    import structlog
    import pandas as pd
    import xgboost as xgb
    import joblib 
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix
    )
    from google.cloud import bigquery
    from google.cloud import aiplatform
    from common.core.logger import get_logger

    # Initialize the application structured logger
    logger = get_logger("training-task")

    try:

        # Unpack infrastructure and job configuration parameters into local variables
        project_id = infra["project_id"]
        location = infra["location"]
        n_estimators = job_params["n_estimators"]
        learning_rate = job_params["learning_rate"]
        max_depth = job_params["max_depth"]

        # Bind context variables. All subsequent logs will include these fields.
        # This is for filtering logs in GCP Cloud Logging.
        structlog.contextvars.bind_contextvars(
            project_id=project_id,
            training_data=training_data.metadata["table_id"],
        )           

        TARGET_COLUMN = "late_delivery"
        F1_THRESHOLD = 0.7
        RECALL_THRESHOLD = 0.6     

        # Data Loading and preprocessing steps
        source_table = training_data.metadata["table_id"]
        client = bigquery.Client(project=project_id, location=location)
        df = client.query(f"SELECT * FROM `{source_table}`").to_dataframe()

        # Drop columns to prevent Data Leakage
        drop_cols = [
           'shipment_id', 'pickup_timestamp',
            'last_event_status', 'last_event_timestamp',
            'pickup_datetime', 'actual_duration_hrs',
            'processed_at'
        ]
        df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # One-Hot Encoding
        categorical_cols = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()
        df_processed = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
        df_processed.columns = df_processed.columns.str.replace(r'[<\[\]]', '', regex=True)

        # Train/Test Split
        X = df_processed.drop(columns=[TARGET_COLUMN])
        y = df_processed[TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Model training
        logger.info("Starting XGBoost training...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "precision_score": precision_score(y_test, y_pred, zero_division=0)

        }
        logger.info("Model Metrics",
                    accuracy=metrics["accuracy"],
                    f1_score=metrics["f1_score"],
                    recall=metrics["recall"],
                    precision_score=metrics["precision_score"])

        # Saving metrics in metadata artifact
        for k, v in metrics.items():
            metrics_artifact.log_metric(k, v)

        # Saving confusion matrix as extra metadata
        cm = confusion_matrix(y_test, y_pred).tolist()
        metrics_artifact.metadata["confusion_matrix"] = cm


        # Check if metrics are good enough
        is_model_good = metrics["f1_score"] >= F1_THRESHOLD and metrics["recall"] >= RECALL_THRESHOLD
        model_artifact.metadata["is_champion"] = is_model_good


        # Saving model
        # KFP expects the file to be saved at model_artifact.path
        # Vertex AI requires the filename 'model.joblib' for automatic deployment
        os.makedirs(model_artifact.path, exist_ok=True)
        model_filename = os.path.join(model_artifact.path, "model.joblib")
        logger.info("Saving model with joblib",
                    model_filename=model_filename,
                    artifact_destination=model_artifact.uri)
        joblib.dump(model, model_filename)

        # Save feature list for future predictions (inference)
        features_path = os.path.join(model_artifact.path, "features.json")
        pd.Series(X.columns).to_json(features_path, orient='values')

        if is_model_good:
            logger.info("Model performance passed thresholds. Registering in Model Registry...",
                        f1=metrics["f1_score"],
                        threshold=F1_THRESHOLD)

            aiplatform.init(project=project_id, location=location)


            SERVING_IMAGE_URI = "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"

            registered_model = aiplatform.Model.upload(
                display_name="forecast-delivery-model",
                artifact_uri=model_artifact.uri,
                serving_container_image_uri=SERVING_IMAGE_URI,
                description="Production-ready logistics delay model",
                labels={
                    "f1_score": str(round(metrics["f1_score"], 2)).replace(".", "_"),
                    "triggered_by": "samir-hinojosa",
                }
            )
            model_artifact.metadata["resource_name"] = registered_model.resource_name
            logger.info("Model registered successfully", resource=registered_model.resource_name)
        else:
            model_artifact.metadata["registry_status"] = "skipped_low_performance"
            logger.warn("Model performance below thresholds. Skipping registry.",
                        f1=metrics["f1_score"],
                        f1_threshold=F1_THRESHOLD,
                        recall=metrics["recall"],
                        recall_threshold=RECALL_THRESHOLD)

    except Exception as e:
        logger.error("Training failed", error_detail=str(e), exc_info=True)
        raise e