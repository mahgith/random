"""
PIPELINE RUNNER
===============
Compiles and submits the pipeline to Vertex AI.

Usage:
    python scripts/run_pipeline.py                    # uses configs/pipeline_config.yaml
    python scripts/run_pipeline.py --compile-only     # just compile, don't submit
    python scripts/run_pipeline.py --dry-run          # print config, don't run

Prerequisites:
    1. Fill in configs/pipeline_config.yaml with your GCP values
    2. Authenticate: gcloud auth application-default login
    3. Install requirements: pip install -r requirements.txt
    4. Enable APIs:
         gcloud services enable aiplatform.googleapis.com
         gcloud services enable storage.googleapis.com

What this script does:
    1. Loads settings from pipeline_config.yaml
    2. Compiles the pipeline to JSON
    3. Submits the JSON to Vertex AI Pipelines
    4. Prints a URL to monitor the run in the GCP console
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Make sure imports work when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import settings


def compile_pipeline(output_path: str) -> None:
    """Compile the pipeline definition to JSON."""
    from kfp.v2.compiler import Compiler
    from pipelines.pipeline.forecasting_pipeline import forecasting_pipeline

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Compiler().compile(
        pipeline_func=forecasting_pipeline,
        package_path=output_path,
    )
    print(f"[✓] Pipeline compiled to: {output_path}")


def submit_pipeline(compiled_path: str) -> None:
    """Submit the compiled pipeline to Vertex AI."""
    from google.cloud import aiplatform

    # Initialise the Vertex AI SDK
    aiplatform.init(
        project=settings.PROJECT_ID,
        location=settings.REGION,
        staging_bucket=settings.ARTIFACT_BUCKET,
        experiment=settings.EXPERIMENT_NAME,
    )

    # Generate a unique run ID so you can tell runs apart in the UI
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job_display_name = f"{settings.PIPELINE_NAME}-{timestamp}"

    print(f"[→] Submitting pipeline run: {job_display_name}")
    print(f"    Project:  {settings.PROJECT_ID}")
    print(f"    Region:   {settings.REGION}")
    print(f"    Bucket:   {settings.ARTIFACT_BUCKET}")

    # Build the parameter dict from settings
    # These map to the @pipeline function parameters in forecasting_pipeline.py
    pipeline_parameters = {
        "project_id": settings.PROJECT_ID,
        "region": settings.REGION,
        "artifact_bucket": settings.ARTIFACT_BUCKET,
        "model_display_name": settings.MODEL_DISPLAY_NAME,
        "serving_container_image_uri": settings.SERVING_CONTAINER,
        "experiment_name": settings.EXPERIMENT_NAME,
        "lookback_days": settings.LOOKBACK_DAYS,
        "forecast_horizon": settings.FORECAST_HORIZON,
        **settings.HYPERPARAMETERS,  # n_estimators, max_depth, learning_rate
    }

    job = aiplatform.PipelineJob(
        display_name=job_display_name,
        template_path=compiled_path,
        job_id=job_display_name,
        parameter_values=pipeline_parameters,
        enable_caching=True,   # reuse cached step results if inputs haven't changed
    )

    # submit() starts the job and returns immediately.
    # Use .run() if you want to block until the job finishes.
    job.submit(
        service_account=None,   # uses the default Compute Engine service account
        # If you have a custom service account: "my-sa@my-project.iam.gserviceaccount.com"
    )

    console_url = (
        f"https://console.cloud.google.com/vertex-ai/pipelines/runs"
        f"?project={settings.PROJECT_ID}"
    )
    print(f"\n[✓] Pipeline submitted!")
    print(f"    Monitor at: {console_url}")
    print(f"    Job name:   {job_display_name}")


def main():
    parser = argparse.ArgumentParser(description="Compile and submit the forecasting pipeline")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile the pipeline, don't submit to Vertex AI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration and exit without compiling or submitting")
    parser.add_argument("--output-path", default=settings.COMPILED_PIPELINE_PATH,
                        help="Path for compiled pipeline JSON")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — configuration ===")
        print(f"  PROJECT_ID:          {settings.PROJECT_ID}")
        print(f"  REGION:              {settings.REGION}")
        print(f"  ARTIFACT_BUCKET:     {settings.ARTIFACT_BUCKET}")
        print(f"  MODEL_DISPLAY_NAME:  {settings.MODEL_DISPLAY_NAME}")
        print(f"  EXPERIMENT_NAME:     {settings.EXPERIMENT_NAME}")
        print(f"  HYPERPARAMETERS:     {settings.HYPERPARAMETERS}")
        print()
        print("Edit configs/pipeline_config.yaml to change these values.")
        return

    compile_pipeline(args.output_path)

    if not args.compile_only:
        submit_pipeline(args.output_path)


if __name__ == "__main__":
    main()
