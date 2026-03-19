# Forecasting MLOps Pipeline — Vertex AI + Kubeflow

End-to-end MLOps pipeline for package volume forecasting (inbound and outbound), built with KFP v1 SDK running on Vertex AI Pipelines.

---

## Architecture Overview

```
BigQuery  →  [ingest]  →  [preprocess]  →  [train]  →  [evaluate]  →  [champion/challenger]  →  [refit]  →  [register]
                                                              ↓                   ↓                   ↓
                                                         approved?          approved?           Vertex AI
                                                         gate 1/2           gate 2/2         Model Registry
```

### 3-Layer Forecasting Model

| Layer | What it does |
|---|---|
| **L1** | Exponential recency-weighted baseline — smooths recent volume with a configurable `half_life_days` |
| **L2A** | Seasonal multiplier table — `week_of_year × day_of_week` ratios learned from history |
| **L2B** | Prophet — captures yearly seasonality (multiplicative mode) on L1-normalised residuals |
| **L3** | LightGBM — log-residual correction on top of L1+L2, using lagged features and rolling stats |

All 3 layers are trained together, saved as a single bundle, and versioned in the Vertex AI Model Registry.

### Pipeline Steps (7 components)

| Step | Component | Description |
|---|---|---|
| 1 | `data_ingestion` | Queries BigQuery, applies UTC-5h offset, deduplicates, sums by day, folds weekends onto weekdays, outputs `[ds, y]` Parquet |
| 2 | `preprocessing` | Adds French holidays, ISO calendar features, rolling stats (10/20/30-day, no data leakage) |
| 3 | `training` | HP grid search (24 combos), each logged to Vertex AI Experiments; selects best by WAPE on rolling backtests; trains candidate bundle |
| 4 | `evaluation` | Rolling backtest across multiple historical cutoffs; gate: mean WAPE ≤ threshold AND mean MAPE ≤ threshold |
| 5 | `champion_vs_challenger` | Downloads current champion from Model Registry; runs same backtest; gate: `challenger_wape ≤ champion_wape - max(delta_abs, delta_rel × champion_wape)`; first run auto-passes |
| 6 | `refit` | Re-trains on 100% of data with winning hyperparams (only runs if both gates passed) |
| 7 | `model_registration` | Registers refitted model as new champion; demotes previous champion to `archived` in registry |

---

## Project Structure

```
.
├── common/                         ← Shared utilities, copied into every Docker image
│   └── core/
│       ├── logger.py               ← structlog setup (JSON in cloud, pretty locally)
│       └── settings.py             ← Pydantic-settings base class; auto-detects Vertex AI
│
├── configs/
│   ├── pipeline_config.yaml        ← Legacy config (kept for reference)
│   └── settings.py                 ← ProjectSettings + build_pipeline_params(direction)
│
├── docker/
│   ├── base/                       ← python:3.10-slim + common libs + common/ code
│   │   ├── Dockerfile
│   │   ├── Makefile                ← `make build` / `make push`
│   │   └── requirements.txt
│   ├── forecasting/                ← Extends base; adds Prophet, LightGBM, libgomp1
│   │   ├── Dockerfile
│   │   ├── Makefile
│   │   └── requirements.txt
│   └── serving/                    ← Extends forecasting; adds FastAPI + uvicorn
│       ├── Dockerfile
│       ├── Makefile
│       ├── main.py                 ← /health + /predict endpoints for Vertex AI Online Prediction
│       └── requirements.txt
│
├── parameters/                     ← Per-direction pipeline configuration (fill PLACEHOLDERs)
│   ├── inbound/
│   │   └── params_v1.yaml
│   └── outbound/
│       └── params_v1.yaml
│
├── pipelines/
│   ├── components/                 ← One file per pipeline step
│   │   ├── data_ingestion.py
│   │   ├── preprocessing.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── champion_vs_challenger.py
│   │   ├── refit.py
│   │   └── model_registration.py
│   ├── pipeline/
│   │   └── forecasting_pipeline.py ← Wires components into a DAG; sets CPU/RAM per step
│   └── compiled/                   ← Generated JSON (git-ignored); never edit manually
│
├── scripts/
│   └── run_pipeline.py             ← Compile + submit to Vertex AI
│
├── notebooks/                      ← Exploration notebooks (inbound / outbound)
│   ├── inbound.ipynb
│   ├── outbound.ipynb
│   ├── modeling_inbound.ipynb
│   └── modeling_outbound.ipynb
│
├── tests/
│   └── test_components_local.py    ← Run component logic locally without GCP
│
└── requirements.txt                ← Local dev deps only (kfp, google-cloud-aiplatform, etc.)
```

---

## Docker Image Hierarchy

```
python:3.10-slim
    └── docker/base          (common/ shared code, structlog, pydantic-settings)
            └── docker/forecasting   (Prophet, LightGBM, Stan build deps)
                        └── docker/serving       (FastAPI, uvicorn)
```

Images are pre-built and pushed to Artifact Registry. Components reference them by URI — no `packages_to_install` at runtime.

---

## Configuration: Fill in the PLACEHOLDERs

Every `PLACEHOLDER` in `parameters/inbound/params_v1.yaml` and `parameters/outbound/params_v1.yaml` must be replaced before running.

```yaml
infra:
  project_id: "your-gcp-project-id"
  region: "europe-west1"
  artifact_bucket: "your-ml-bucket"
  forecasting_image_uri: "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/forecasting:latest"
  serving_image_uri: "europe-west1-docker.pkg.dev/your-gcp-project-id/ml-images/serving:latest"
  experiment_name: "forecasting-inbound-v1"

bq_tables:
  - project: "your-gcp-project-id"
    dataset: "your_dataset"
    table: "your_table"
    date_column: "scan_date"
    volume_column: "package_count"
```

Also replace `your-gcp-project-id` in all component files (the `_FORECASTING_IMAGE` / `_SERVING_IMAGE` constants at the top of each component).

---

## First-Time GCP Setup

### 1. Install tools

```bash
# Google Cloud CLI: https://cloud.google.com/sdk/docs/install
# Docker Desktop: https://www.docker.com/products/docker-desktop/

pip install -r requirements.txt
```

### 2. Authenticate

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Enable APIs (one-time)

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  cloudlogging.googleapis.com
```

### 4. Create infrastructure (one-time)

```bash
# GCS bucket for pipeline artifacts and model bundles
gsutil mb -l europe-west1 gs://your-ml-bucket

# Artifact Registry repo for Docker images
gcloud artifacts repositories create ml-images \
  --repository-format=docker \
  --location=europe-west1
```

### 5. IAM permissions

Grant the Vertex AI default service account (`...@developer.gserviceaccount.com`) these roles:
- `BigQuery Data Viewer` + `BigQuery Job User`
- `Storage Object Admin`
- `Vertex AI User`
- `Artifact Registry Reader`

---

## Building and Pushing Docker Images

```bash
# Authenticate Docker to Artifact Registry
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Build and push (base must be built first)
cd docker/base        && make build && make push
cd docker/forecasting && make build && make push
cd docker/serving     && make build && make push
```

First build takes 10-15 minutes (Prophet/Stan compile from source). Subsequent builds use layer cache — usually seconds if only Python files changed.

---

## Running the Pipeline

```bash
# Dry run — prints resolved parameters, does not submit
python scripts/run_pipeline.py --direction inbound --dry-run

# Compile only — produces pipelines/compiled/forecasting_pipeline.json
python scripts/run_pipeline.py --direction inbound --compile-only

# Compile and submit to Vertex AI
python scripts/run_pipeline.py --direction inbound

# Outbound direction
python scripts/run_pipeline.py --direction outbound
```

The script prints a URL to the GCP Console to watch the run live.

---

## Debugging and Iteration

**Never iterate directly on Vertex AI — it's too slow.** Use this 3-layer approach:

| Layer | Where | Speed | Use for |
|---|---|---|---|
| 1 | Pure Python | ~2 sec | Logic bugs, calculation fixes, data issues |
| 2 | Local Docker | ~1 min | Missing packages, import errors, Dockerfile issues |
| 3 | Vertex AI | 5-30 min | Real BQ access, GCS writes, full DAG validation |

**Test a component as plain Python:**

```python
# Fake KFP artifact
class FakeArtifact:
    def __init__(self, path): self.path = path; self.metadata = {}

from pipelines.components.training import training_fn

training_fn(
    preprocessed_dataset=FakeArtifact("/tmp/data.parquet"),
    candidate_model=FakeArtifact("/tmp/model/"),
    hp_grid_json='{"half_life_days": [10], "n_changepoints": [5]}',
    ...
)
```

**Test inside Docker:**

```bash
docker build -f docker/forecasting/Dockerfile -t forecasting:dev .
docker run --rm -v $(pwd)/scratch:/scratch forecasting:dev python /scratch/test_component.py
```

---

## Monitoring

| What | Where |
|---|---|
| Pipeline DAG and step status | GCP Console → Vertex AI → Pipelines |
| HP search results | GCP Console → Vertex AI → Experiments |
| Component logs (structlog JSON) | GCP Console → Cloud Logging → Logs Explorer, filter by pipeline job ID |
| Registered models and versions | GCP Console → Vertex AI → Model Registry |

---

## Key Concepts

**KFP components** are regular Python functions decorated with `@component`. Each runs in its own container on a separate GCP machine. They do not share memory — data passes via GCS files (KFP manages the upload/download via `Input[Dataset]` / `Output[Dataset]` artifacts).

**Decision files** — gates (evaluation, champion/challenger) write `"approved"` or `"rejected"` to an `Output[str]` artifact. Downstream components read this file and skip their logic if rejected. This avoids `dsl.If()` which is not fully supported in KFP v1.

**Caching** — `enable_caching=True` on `ingest` and `preprocess` means re-runs with the same inputs reuse previous outputs. If only training code changes, KFP skips the first two steps automatically.

**Directions** — `inbound` and `outbound` share the same pipeline code. Different behaviour is driven entirely by `parameters/inbound/params_v1.yaml` vs `parameters/outbound/params_v1.yaml` (different lookback windows, different HP grids, different BQ tables).

**Logging** — `common/core/logger.py` uses structlog. On Vertex AI (`CLOUD_ML_JOB_ID` env var present), it emits structured JSON compatible with Cloud Logging. Locally, it emits coloured human-readable output.

---

## Common Errors

| Error | Fix |
|---|---|
| `PermissionDenied` on GCS/BQ | Check service account IAM roles |
| `Image not found` | Build and push Docker images first |
| `ModuleNotFoundError` inside component | Add the package to `docker/forecasting/requirements.txt` and rebuild |
| `PLACEHOLDER` validation error | Fill in all PLACEHOLDERs in `parameters/<direction>/params_v1.yaml` |
| Pipeline step immediately fails | Check Cloud Logging for the Python traceback |
| `Pipeline hangs at Pending` | Check billing is enabled on the GCP project |
