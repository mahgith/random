ARG BASE_IMAGE_URI=forecasting:latest
FROM ${BASE_IMAGE_URI}

COPY docker/serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the serving application
COPY docker/serving/main.py /app/main.py

EXPOSE 8080

# Vertex AI Online Prediction expects the server on port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
