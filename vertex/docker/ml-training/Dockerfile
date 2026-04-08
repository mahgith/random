ARG BASE_IMAGE_URI
FROM ${BASE_IMAGE_URI}

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY docker/ml-training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install cmdstan to a fixed system directory
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan('/opt/cmdstan')"

# Record the exact versioned path (e.g. /opt/cmdstan/cmdstan-2.35.0) to a file.
# The component code reads this at runtime and calls set_cmdstan_path() directly,
# avoiding any dependency on the home directory or environment variables.
RUN find /opt/cmdstan -maxdepth 1 -name 'cmdstan-*' -type d | sort | tail -1 \
    > /opt/cmdstan_dir.txt && \
    echo "CmdStan installed at: $(cat /opt/cmdstan_dir.txt)"
