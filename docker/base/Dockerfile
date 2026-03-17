FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr (logs appear instantly in Cloud Logging)
# Prevent .pyc files accumulating in the container layer
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/base/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Make our shared utilities importable inside every component container
COPY common/ /app/common/

ENV PYTHONPATH="/app"
