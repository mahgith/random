# 🚀 Vertex AI Pipeline starter-kit

This template provides a standardized and scalable architecture to develop Machine Learning pipelines on Google Cloud Vertex AI using Kubeflow Pipelines (KFP).

## 📋 Prerequisites
- Python 3.12 (recommended)
- Poetry 2.x
- Google Cloud SDK
- A Google Cloud project with billing enabled and Vertex AI / Vertex Pipelines APIs enabled
- make

> Optional: Docker (and access to Artifact Registry) if you plan to build custom images for use in KFP/Vertex components


## 📁 Project Architecture

The project follows a modular structure to cleanly separate infrastructure from business logic:

* **`vertex/`**: The core of the MLOps application.
  * **`common/`**: Shared libraries/utilities used by one or multiple components (e.g., shared/sarima → utilities for the SARIMA model).
  * **`common/core/`**: Cross-functional utilities shared across all components (Loggers, Settings, YAML parsers).
  * **`components/`**: Modularized business logic (Ingestion, Preprocessing, etc.).
  * **`docker/`**: Container definitions (`Dockerfile` and dependencies).
  * **`parameters/`**: Standardized YAML files to inject variables into pipelines.
  * **`pipelines/`**: KFP orchestration definitions.
* **`notebooks/`**: Environment for local experimentation and Proofs of Concept (PoC).
* **`scripts/`**: Utility scripts for local development only.


## 🛠️ Core Technologies
* **Python 3.12** + **Poetry** (Dependency management)
* **Typer** (CLI creation)
* **Kubeflow Pipelines (KFP)** (Orchestration)
* **Google Cloud Platform**: Vertex AI, BigQuery, Cloud Storage

## 📂 Project Structure (overview)
> ℹ️ **Directory‑level READMEs**
> Each major directory includes its own `README.md` with practical guidance on how to use, extend, and organize that part of the project—covering purpose, layout, naming conventions, configuration, and common Make targets.


```bash
  .                             
  ├── demo                                      # ⚠️ Illustrative example. NOT for production
  ├── notebooks                                 # Jupyter notebooks 
  ├── scripts                                   # Utility scripts for local development only.
  ├── vertex                                    # Main Vertex AI project structure
  │   ├── commons                               # Shared library utilities
  │   │   └── core                              # Core helper modules
  │   ├── components                            # Reusable pipeline components
  │   │   ├── data                              # Data-centric components
  │   │   │   ├── ingest                        # Batch/stream ingestion to raw/staging
  │   │   │   ├── preprocess                    # Cleaning, imputation, joins, schema fixes
  │   │   │   ├── featurize                     # Feature generation and selection
  │   │   │   └── split                         # Train/val/test splitting strategies
  │   │   ├── ml                                # Modeling components
  │   │   │   ├── train                         # Model training steps and routines
  │   │   │   ├── evaluate                      # Metrics, validation, and reporting
  │   │   │   ├── compare                       # Champion/challenger and model selection
  │   │   │   └── refit                         # Refit on full data for production
  │   │   ├── governance                        # Governance and lifecycle controls
  │   │   │   └── register                      # Model packaging and registry integration
  │   │   └── ops                               # Operational/infra components
  │   │   │   └── monitor                       # Data and model monitoring jobs
  │   ├── docker                                # Docker images used in pipelines
  │   │   ├── base                              # Base Docker image configuration
  │   │   │   ├── Dockerfile
  │   │   │   └── version.yaml
  │   ├── parameters                            # Pipeline parameter configuration files (arguments)
  │   ├── pipelines                             # Pipeline definitions and templates
  │   │   └── templates                         # Compiled pipeline templates (JSON) ready to submit
  ├── Makefile                           
  ├── poetry.lock                        
  ├── pyproject.toml                     
  └── README.md                       
```
## 🚀  Installation

### Development mode (main + dev dependencies)
Install all dependencies including development tools:
```bash
make install
```

Setup your GCP environment:
```bash
export PROJECT_ID=<gcp_project_id>
gcloud config set project $PROJECT_ID
gcloud auth login
gcloud auth application-default login
```
> Note: You must have all required permissions on the GCP project to access services from the command line (IAM roles for authentication, storage access, APIs, etc.).

## 🧪 Notebooks & Experimental Development (JupyterLab)
The project includes a `/notebooks` directory intended for:
- Data exploration
- Rapid prototyping
- Testing ideas before integrating them into the core codebase
- Developing scrapers, transformations, or analysis drafts
- Interactive workflows using marimo

You can open the Jupyter Lab environment as follows:
```bash
make jupyterlab
```
## ⚙️ Viewing available commands
You can run make help to see all available commands, which act as shortcuts for common or repetitive tasks.
```bash
make help
```

## 🧑‍💻 Demo pipeline (synthetic data) — Quickstart

> ⚠️ **Illustrative example. NOT for production.**  
> This section gives a brief overview. The **full step‑by‑step** and commands live in:
> - **Demo guide:** [`./demo/README.md`](./demo/README.md)  

### What this demo does
Runs an end‑to‑end pipeline on Vertex AI using synthetic data, mirroring the template layout (data → preprocess/featurize → split → train → evaluate/compare → register → monitor).