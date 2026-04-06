ARG BASE_IMAGE_URI
FROM ${BASE_IMAGE_URI}

COPY docker/ml-training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
