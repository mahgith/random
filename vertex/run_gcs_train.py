"""
GCS TRAINING PIPELINE RUNNER
=============================
Compiles and submits the gcs_train_pipeline to Vertex AI.

Usage
-----
    # Print resolved parameters, do nothing
    python vertex/run_gcs_train.py --dry-run

    # Compile the pipeline JSON only (no GCP call)
    python vertex/run_gcs_train.py --compile-only

    # Compile + submit to Vertex AI
    python vertex/run_gcs_train.py

    # Use a different params file
    python vertex/run_gcs_train.py --params vertex/parameters/gcs_train/params_v2.yaml

Before running
--------------
    1. Edit vertex/parameters/gcs_train/params_v1.yaml:
         - Set data.date_column to the actual column name in your CSV
    2. Build and push the ml-training Docker image:
         cd vertex/docker/base   && make release   # rebuild base if needed
         cd vertex/docker/ml-training && make release
    3. Authenticate:
         gcloud auth application-default login
"""

import argparse
import sys
from pathlib import Path

import yaml

DEFAULT_PARAMS_FILE = Path(__file__).parent / "parameters" / "gcs_train" / "params_v1.yaml"
COMPILED = Path(__file__).parent / "pipelines" / "gcs_train_pipeline.json"

# Make vertex/ importable so the pipeline can import its components
sys.path.insert(0, str(Path(__file__).parent))


def load_params(params_file: Path) -> dict:
    with open(params_file) as f:
        cfg = yaml.safe_load(f)

    infra   = cfg["infra"]
    data    = cfg["data"]
    train   = cfg["training"]
    evalu   = cfg["evaluation"]

    date_column = data["date_column"]
    if date_column == "PLACEHOLDER_date_column":
        raise ValueError(
            "Please set data.date_column in your params file to the actual "
            "date column name in your CSV before running the pipeline."
        )

    return {
        # infra
        "project_id":      infra["project_id"],
        "location":        infra["location"],
        "bucket_name":     infra["bucket_name"],
        "service_account": infra.get("service_account"),
        # data
        "gcs_uri":         data["gcs_uri"],
        "date_column":     date_column,
        "target_column":   data["target_column"],
        "warehouse_id":    data.get("warehouse_id", "GENNEVILLIERS"),
        # training
        "lookback_days":                   int(train["lookback_days"]),
        "half_life_days":                  int(train["half_life_days"]),
        "prophet_changepoint_prior_scale": float(train["prophet_changepoint_prior_scale"]),
        "clip_min_ratio":                  float(train["clip_min_ratio"]),
        "clip_max_ratio":                  float(train["clip_max_ratio"]),
        "lgbm_n_estimators":               int(train["lgbm_n_estimators"]),
        "lgbm_learning_rate":              float(train["lgbm_learning_rate"]),
        "lgbm_num_leaves":                 int(train["lgbm_num_leaves"]),
        # evaluation
        "forecast_horizon":      int(evalu["forecast_horizon"]),
        "backtest_step_days":    int(evalu["backtest_step_days"]),
        "evaluation_start_date": evalu["evaluation_start_date"],
    }


def compile_pipeline() -> None:
    from kfp.compiler import Compiler
    from pipelines.gcs_train_pipeline import gcs_train_pipeline

    COMPILED.parent.mkdir(parents=True, exist_ok=True)
    Compiler().compile(pipeline_func=gcs_train_pipeline, package_path=str(COMPILED))
    print(f"[OK] Compiled → {COMPILED}")


def submit(params: dict) -> None:
    from google.cloud import aiplatform

    aiplatform.init(
        project=params["project_id"],
        location=params["location"],
    )

    # Build the parameter_values dict — only pipeline params, no infra keys
    pipeline_params = {k: v for k, v in params.items()
                       if k not in ("project_id", "location", "bucket_name", "service_account")}
    # project_id is also a pipeline param
    pipeline_params["project_id"] = params["project_id"]

    job = aiplatform.PipelineJob(
        display_name="gcs-train-pipeline",
        template_path=str(COMPILED),
        pipeline_root=f"gs://{params['bucket_name']}/pipeline_root/gcs_train",
        parameter_values=pipeline_params,
        enable_caching=False,
    )

    sa = params.get("service_account")
    job.submit(service_account=sa if sa else None)
    print(f"\n[OK] Pipeline submitted!")
    print(f"     Dashboard: {job._dashboard_uri()}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile and submit the GCS training pipeline to Vertex AI"
    )
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print resolved parameters and exit")
    parser.add_argument("--compile-only", action="store_true",
                        help="Compile the pipeline JSON but do not submit")
    parser.add_argument("--params",       type=Path, default=DEFAULT_PARAMS_FILE,
                        help="Path to the params YAML file")
    args = parser.parse_args()

    params = load_params(args.params)

    if args.dry_run:
        import json
        print(json.dumps(params, indent=2, default=str))
        return

    compile_pipeline()

    if not args.compile_only:
        submit(params)


if __name__ == "__main__":
    main()
