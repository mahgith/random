# Build on top of the base image so all GCP / structlog deps are inherited.
# Usage:
#   docker build -f docker/forecasting/Dockerfile \
#     --build-arg BASE_IMAGE_URI=europe-west1-docker.pkg.dev/your-project/ml-images/base:latest \
#     -t europe-west1-docker.pkg.dev/your-project/ml-images/forecasting:latest .
ARG BASE_IMAGE_URI=base:latest
FROM ${BASE_IMAGE_URI}

# libgomp1 is required by LightGBM (OpenMP parallel training)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY docker/forecasting/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
