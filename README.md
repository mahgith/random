# Forecasting MLOps Pipeline — Vertex AI + Kubeflow

End-to-end MLOps pipeline for forecasting, built with KFP (Kubeflow Pipelines) running on Vertex AI.

---

## Project Structure

```
.
├── configs/
│   ├── pipeline_config.yaml    ← ALL settings live here (fill this in first)
│   └── settings.py             ← Loads the YAML, used by pipeline and scripts
│
├── pipelines/
│   ├── components/             ← Each file = one pipeline step
│   │   ├── data_ingestion.py   ← Pull raw data, write to GCS
│   │   ├── preprocessing.py    ← Feature engineering, train/test split
│   │   ├── training.py         ← Train model, save artifact
│   │   ├── evaluation.py       ← Compute metrics, approve/reject model
│   │   └── model_registration.py  ← Register in Vertex AI Model Registry
│   │
│   ├── pipeline/
│   │   └── forecasting_pipeline.py  ← Wires components into a pipeline graph
│   │
│   └── compiled/               ← Generated JSON (git-ignored), not edited manually
│
├── scripts/
│   └── run_pipeline.py         ← Compile + submit pipeline to Vertex AI
│
├── tests/
│   └── test_components_local.py  ← Run component logic locally (no GCP needed)
│
├── notebooks/                  ← Your existing exploration notebooks (source of truth)
└── requirements.txt
```

---

## Step-by-Step: How to Get This Running

### Step 1 — Prerequisites (one-time GCP setup)

```bash
# 1. Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# 2. Authenticate
gcloud auth login
gcloud auth application-default login

# 3. Set your project
gcloud config set project YOUR_PROJECT_ID

# 4. Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com  # if using BigQuery

# 5. Create a GCS bucket for pipeline artifacts
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME
```

### Step 2 — Fill in your configuration

Edit `configs/pipeline_config.yaml`:

```yaml
gcp:
  project_id: "my-actual-project-id"     # ← your GCP project
  region: "us-central1"
  artifact_bucket: "gs://my-bucket"      # ← bucket you created above
```

Verify it works:
```bash
python -c "from configs.settings import settings; print(settings.PROJECT_ID)"
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run tests locally (no GCP needed)

This runs your component logic on your machine using synthetic data.
**Do this before spending any GCP credits.**

```bash
pytest tests/test_components_local.py -v
```

All tests should pass. If not, fix the component code before moving on.

### Step 5 — Plug in your real notebook code

Open each component file and replace the placeholder sections with your actual logic.
The files tell you exactly where your code goes.

**Order to work in:**
1. `pipelines/components/data_ingestion.py` — replace the synthetic placeholder with your real data source (BigQuery or GCS)
2. `pipelines/components/preprocessing.py` — paste your feature engineering from your notebook
3. `pipelines/components/training.py` — paste your model training code
4. `pipelines/components/evaluation.py` — paste your evaluation metrics
5. Re-run tests after each component: `pytest tests/ -v`

### Step 6 — Compile and do a dry run

```bash
# Check configuration
python scripts/run_pipeline.py --dry-run

# Compile only (produces pipelines/compiled/forecasting_pipeline.json)
python scripts/run_pipeline.py --compile-only
```

### Step 7 — Submit to Vertex AI

```bash
python scripts/run_pipeline.py
```

Then open the printed URL in your browser to watch the pipeline run in the GCP console.

---

## Key Concepts (short version)

### What is a KFP Component?
A Python function decorated with `@component`. Each component:
- Runs in its own container (isolated environment)
- Gets its inputs as function arguments
- Writes its outputs to `Output[Dataset]` / `Output[Model]` artifacts
- Does NOT share memory with other components

### How do components pass data to each other?
They don't pass DataFrames. They pass **GCS file paths** (URIs).
KFP manages the upload/download automatically via the `Input[Dataset]` / `Output[Dataset]` types.

```
data_ingestion writes → gs://bucket/.../raw.parquet
preprocessing reads  ← gs://bucket/.../raw.parquet
preprocessing writes → gs://bucket/.../train.parquet
training reads       ← gs://bucket/.../train.parquet
```

### What is caching?
If you re-run the pipeline with the same inputs, KFP skips steps that already ran.
This means if only your training code changes, KFP reuses the preprocessing results — saving time and money.

### Where do I watch runs?
GCP Console → Vertex AI → Pipelines → Pipeline runs

### Where are metrics logged?
GCP Console → Vertex AI → Experiments

---

## Common Errors and Fixes

| Error | Fix |
|---|---|
| `google.api_core.exceptions.PermissionDenied` | Run `gcloud auth application-default login` |
| `ImportError: No module named 'kfp'` | Run `pip install -r requirements.txt` |
| `Bucket does not exist` | Create it: `gsutil mb gs://YOUR_BUCKET` |
| Component fails with `ModuleNotFoundError` | Add the missing package to `packages_to_install` in the `@component` decorator |
| Pipeline hangs at `Pending` | Check your GCP project has billing enabled |

---

## Next Steps (after the basic pipeline works)

1. **Hyperparameter tuning** — Vertex AI Vizier (`aiplatform.HyperparameterTuningJob`)
2. **Model monitoring** — Vertex AI Model Monitoring for data drift detection
3. **Scheduled runs** — Cloud Scheduler to trigger the pipeline on a cron
4. **Custom serving container** — When your serving needs custom pre/post-processing
5. **CI/CD** — GitHub Actions to retrain on every merge to main
