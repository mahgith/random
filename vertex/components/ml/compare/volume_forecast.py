from typing import Dict, Any
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# Docker image configuration
PROPHET = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/prophet:latest"

@dsl.component(base_image=PROPHET)
def gate_champion_vs_challenger(
    candidate_model: Input[Model],
    eval_data: Input[Dataset],
    metrics: Output[Metrics],
    infra: Dict[str, Any],
    job_params: Dict[str, Any],
) -> bool:
    """
    Executes a Champion vs. Challenger evaluation against the current production model.

    This component compares a newly trained candidate model (Challenger) against the 
    best performing model currently registered in the Vertex AI Model Registry (Champion). 
    It downloads the Champion's artifact from GCS, generates predictions on a shared 
    holdout dataset, and computes performance metrics using WAPE as the primary indicator. 
    The Challenger wins only if it exceeds the Champion's performance by a strictly 
    defined hybrid margin (maximum of an absolute or relative delta) without degrading 
    secondary metrics (RMSE, Bias) beyond allowable tolerances. If no Champion exists 
    (Day 0), the Challenger wins automatically.

    Args:
        candidate_model (Input[Model]): 
            The serialized candidate Prophet model to be evaluated.
        eval_data (Input[Dataset]): 
            The holdout validation dataset in CSV format.
        metrics (Output[Metrics]): 
            The KFP artifact where comparative performance metrics are logged.
        infra (Dict[str, Any]): 
            Global infrastructure configuration (project_id, location).
        job_params (Dict[str, Any]): 
            Evaluation parameters:
            - model_display_name (str): The display name to query in Vertex Model Registry.
            - champion_label_env (str): Environment label (e.g., 'prod').
            - champion_label_role (str): Role label (e.g., 'champion').
            - delta_rel (float): Minimum required relative WAPE improvement (e.g., 0.03 for 3%).
            - delta_abs (float): Minimum required absolute WAPE improvement.
            - rmse_tol (float, optional): Maximum allowed relative degradation in RMSE.
            - bias_margin (float, optional): Maximum allowed absolute degradation in Bias.

    Returns:
        bool: True if the Challenger defeats the Champion (or if Day 0), False otherwise.
    """

    import os
    import pandas as pd
    import structlog
    from google.cloud import aiplatform
    from google.cloud import storage
    from prophet.serialize import model_from_json
    from common.core.logger import get_logger
    from common.prophet.helper import parse_gcs_uri, download_text_from_gcs, pick_latest_model
    
    # Initialize the application structured logger
    from common.prophet.helper import compute_forecast_metrics

    logger = get_logger("champion-challenger-gate")

    try:
        # Unpack infrastructure and job configuration parameters into local variables
        project_id = infra["project_id"]
        location = infra["location"]
        model_display_name = job_params["model_display_name"]
        champion_env = job_params["champion_label_env"]
        champion_role = job_params["champion_label_role"]
        delta_rel = float(job_params["delta_rel"])  
        delta_abs = float(job_params["delta_abs"])  
        rmse_tol = job_params.get("rmse_tol")            
        bias_margin = job_params.get("bias_margin")      

        # Bind context variables. All subsequent logs will include these fields.
        structlog.contextvars.bind_contextvars(
            project_id=project_id,
        )  

        logger.info(
            "Starting Champion vs Challenger gate.",
            model_display_name=model_display_name,
            champion_label_env=champion_env,
            champion_label_role=champion_role,
            delta_rel=delta_rel,
            delta_abs=delta_abs,
            rmse_tol=rmse_tol,
            bias_margin=bias_margin
        )

        # Load challenger model 
        candidate_model_path = os.path.join(candidate_model.path, "model.json")
        with open(candidate_model_path, "r") as fin:
            challenger = model_from_json(fin.read())

        # Load evaluation data with Fail-Fast column renaming 
        df_eval = pd.read_csv(eval_data.path)
        
        if "ds" not in df_eval.columns or "y" not in df_eval.columns:
            logger.error("Expected columns 'ds' and 'y' not found.", found_columns=list(df_eval.columns))
            raise ValueError(f"Expected columns 'ds' and 'y' not found. Found columns: {list(df_eval.columns)}")
            
        df_eval = df_eval.sort_values(by="ds").reset_index(drop=True)
        y_true = df_eval["y"].astype(float)

        # Find champion in Vertex Model Registry 
        aiplatform.init(project=project_id, location=location)
        filter_expr = (
            f'display_name="{model_display_name}" '
            f'AND labels.env="{champion_env}" '
            f'AND labels.role="{champion_role}"'
        )
        champion_candidates = aiplatform.Model.list(filter=filter_expr)

        # Day 0 automatic approval
        if not champion_candidates:
            logger.info("No champion model found in registry. Day 0 bootstrap -> Candidate WINS.")
            metrics.log_metric("day0_no_champion_found", 1.0)
            return True

        champion_model = pick_latest_model(champion_candidates)
        logger.info("Champion model found.", champion_resource_name=champion_model.resource_name)

        # Load champion model from GCS 
        champ_artifact_uri = champion_model.uri
        champ_model_json = download_text_from_gcs(champ_artifact_uri, "model.json")
        champion = model_from_json(champ_model_json)

        # Predict with both models 
        challenger_fcst = challenger.predict(df_eval[["ds"]])
        champion_fcst = champion.predict(df_eval[["ds"]])

        yhat_challenger = challenger_fcst["yhat"].astype(float)
        yhat_champion = champion_fcst["yhat"].astype(float)

        # Compute metrics via Unified Helper 
        cand_metrics = compute_forecast_metrics(y_true=y_true, y_pred=yhat_challenger)
        champ_metrics = compute_forecast_metrics(y_true=y_true, y_pred=yhat_champion)

        wape_challenger = cand_metrics["wape"]
        wape_champion = champ_metrics["wape"]
        rmse_challenger = cand_metrics["rmse"]
        rmse_champion = champ_metrics["rmse"]
        bias_challenger = cand_metrics["me_bias"]
        bias_champion = champ_metrics["me_bias"]

        # Log metrics to KFP
        metrics.log_metric("champion_wape", wape_champion)
        metrics.log_metric("candidate_wape", wape_challenger)
        metrics.log_metric("champion_rmse", rmse_champion)
        metrics.log_metric("candidate_rmse", rmse_challenger)
        metrics.log_metric("champion_bias", bias_champion)
        metrics.log_metric("candidate_bias", bias_challenger)

        # Decide winner using hybrid improvement rule 
        required_improvement = max(delta_abs, delta_rel * wape_champion)
        wape_target = wape_champion - required_improvement

        # Candidate wins if it improves WAPE by at least the required amount
        wins = (wape_challenger <= wape_target)

        # Optional guardrail: RMSE should not get worse beyond tolerance
        if rmse_tol is not None:
            rmse_tol = float(rmse_tol)
            wins = wins and (rmse_challenger <= rmse_champion * (1.0 + rmse_tol))

        # Optional guardrail: abs(Bias) should not increase too much
        if bias_margin is not None:
            bias_margin = float(bias_margin)
            wins = wins and (abs(bias_challenger) <= abs(bias_champion) + bias_margin)

        metrics.log_metric("required_improvement_wape", required_improvement)
        metrics.log_metric("wape_target", wape_target)
        metrics.log_metric("wins_against_champion", 1.0 if wins else 0.0)

        logger.info(
            "Champion vs Challenger decision completed.",
            wape_champion=wape_champion,
            wape_challenger=wape_challenger,
            required_improvement=required_improvement,
            wape_target=wape_target,
            wins=wins
        )

        return bool(wins)

    except Exception as e:
        logger.error("Failure during Champion vs Challenger gate.", error=str(e))
        raise RuntimeError(f"Failure during Champion vs Challenger gate: {e}") from e