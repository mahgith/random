"""
Compile and submit the demo pipeline to Vertex AI.

Usage
-----
    # Print resolved parameters, do nothing
    python vertex/run_demo.py --dry-run

    # Compile the pipeline JSON only
    python vertex/run_demo.py --compile-only

    # Compile + submit to Vertex AI
    python vertex/run_demo.py

Before running
--------------
    1. Edit vertex/parameters/demo/params_v1.yaml (three TODOs)
    2. Upload a CSV to GCS:
           gsutil cp your_data.csv gs://<bucket>/demo/data.csv
    3. Authenticate:
           gcloud auth application-default login
"""

import argparse
import sys
from pathlib import Path

import yaml

PARAMS_FILE = Path(__file__).parent / "parameters" / "demo" / "params_v1.yaml"
COMPILED    = Path(__file__).parent / "pipelines" / "demo_pipeline.json"

# Make vertex/ importable so the pipeline can import its components
sys.path.insert(0, str(Path(__file__).parent))


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        cfg = yaml.safe_load(f)
    infra = cfg["infra"]
    demo  = cfg["demo"]
    return {
        "project_id":    infra["project_id"],
        "location":      infra["location"],
        "bucket_name":   infra["bucket_name"],
        "gcs_uri":       demo["gcs_uri"],
        "target_column": demo["target_column"],
    }


def compile_pipeline() -> None:
    from kfp.compiler import Compiler
    from pipelines.demo_pipeline import demo_pipeline

    COMPILED.parent.mkdir(parents=True, exist_ok=True)
    Compiler().compile(pipeline_func=demo_pipeline, package_path=str(COMPILED))
    print(f"[OK] Compiled → {COMPILED}")


def submit(params: dict) -> None:
    from google.cloud import aiplatform

    aiplatform.init(
        project=params["project_id"],
        location=params["location"],
    )

    job = aiplatform.PipelineJob(
        display_name="demo-pipeline",
        template_path=str(COMPILED),
        pipeline_root=f"gs://{params['bucket_name']}/pipeline_root",
        parameter_values={
            "project_id":    params["project_id"],
            "gcs_uri":       params["gcs_uri"],
            "target_column": params["target_column"],
        },
        enable_caching=False,
    )
    job.submit()
    print(f"\n[OK] Submitted!")
    print(f"     Dashboard: {job._dashboard_uri()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    params = load_params()

    if args.dry_run:
        import json
        print(json.dumps(params, indent=2))
        return

    compile_pipeline()

    if not args.compile_only:
        submit(params)


if __name__ == "__main__":
    main()
