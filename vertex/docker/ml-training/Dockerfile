ARG BASE_IMAGE_URI
FROM ${BASE_IMAGE_URI}

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY docker/ml-training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install cmdstan so Prophet can run offline inside Vertex AI containers.
# Installs to /root/.cmdstanpy/ — containers run as root so it's found at runtime.
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
