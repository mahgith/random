FROM python:3.12-slim

# PYTHONUNBUFFERED: Ensures logs are sent directly to stdout without being buffered
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to the container
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/base/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire common package (including common/__init__.py)
# so that "from common.core.logger import get_logger" resolves correctly
COPY common/ /app/common/

ENV PYTHONPATH="/app"
