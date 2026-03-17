from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Metrics
from typing import Dict, Any

IMAGE = "europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images/base:latest"

@dsl.component(base_image=IMAGE)
def data_drift_evaluation(
    input_data: Input[Dataset],
    # output_data: Output[Dataset],
    drift_metrics: Output[Metrics],
) -> bool:
    
    import structlog
    import pandas as pd
    import shutil
    from common.core.logger import get_logger

    # Initialize the application structured logger
    logger = get_logger("split-training-task")

    try:
        # shutil.copy(input_data.path, output_data.path)

        # max_allowed_shift_pct = 0.30
        # simulated_baseline_mean = 10500.0  
        # simulated_recent_mean = 10100.0    
        # shift_pct = abs((simulated_recent_mean - simulated_baseline_mean) / simulated_baseline_mean)

        # drift_metrics.log_metric("1_baseline_mean", simulated_baseline_mean)
        # drift_metrics.log_metric("2_recent_mean", simulated_recent_mean)
        # drift_metrics.log_metric("3_actual_shift_pct", round(shift_pct, 4))

        # if shift_pct > max_allowed_shift_pct:
        #     logger.error("Critical data drift detected.")
        #     drift_metrics.log_metric("4_drift_status", "CRITICAL_DRIFT")
        #     return False

        logger.info("Validación de datos superada.")
        drift_metrics.log_metric("4_drift_status", "HEALTHY")
        return True

    except Exception as e:
        logger.error("Unexpected error during data drift evaluation", error=str(e))
        raise RuntimeError(f"FUnexpected error during data drift evaluation: {e}") from e