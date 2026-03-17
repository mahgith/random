ARG BASE_IMAGE_URI=base:latest
FROM ${BASE_IMAGE_URI}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY docker/prophet/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Copy directory
COPY common/prophet/ /app/common/prophet

EXPOSE 8080

CMD ["uvicorn", "common.prophet.api.main:app", "--host", "0.0.0.0", "--port", "8080"]