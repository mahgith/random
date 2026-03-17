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

# Set the working directory 
WORKDIR /app

COPY docker/base/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the src directory
COPY common/core/ /app/common/core

ENV PYTHONPATH="/app"