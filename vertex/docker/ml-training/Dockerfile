ARG BASE_IMAGE_URI
FROM ${BASE_IMAGE_URI}

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY docker/ml-training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install cmdstan at build time so Prophet works offline inside Vertex AI containers.
# Without this, Prophet tries to download cmdstan at runtime and fails (no internet).
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
