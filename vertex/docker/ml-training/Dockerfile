ARG BASE_IMAGE_URI
FROM ${BASE_IMAGE_URI}

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY docker/ml-training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install cmdstan to a fixed system path so Prophet can find it regardless
# of which user the container runs as at runtime (Vertex AI uses a non-root user).
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan('/opt/cmdstan')"
ENV CMDSTAN=/opt/cmdstan
