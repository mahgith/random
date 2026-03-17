import os
import structlog
from fastapi import FastAPI, Request
from prophet.serialize import model_from_json
import pandas as pd
from common.core.logger import get_logger

app = FastAPI()
model = None

# Initialize the application structured logger
logger = get_logger("prophet-model-serving") 

# Bind context variables. All subsequent logs will include these fields.
# This is for filtering logs in GCP Cloud Logging.
structlog.contextvars.bind_contextvars(
)  

# Load the model into memory when starting the container
@app.on_event("startup")
def load_model():
    global model
    # Vertex AI automatically mounts the model artifact on this path.
    model_dir = os.environ.get("AIP_STORAGE_URI")
    
    if model_dir:
        # saved model name
        model_path = os.path.join(model_dir, "model.json") 
        with open(model_path, 'r') as f:
            model = model_from_json(f.read())
        logger.info("Prophet model successfully loaded into memory.")

# Health pathway (Required for Vertex AI)
@app.get("/health")
def health():
    return {"status": "healthy"}

# Prediction path (Required for Vertex AI)
@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    
    # Vertex AI sends the data in an array called "instances"
    # Example payload: {"instances": [{"date": "2026-03-01"}, {"date": "2026-03-02"}]}
    instances = body.get("instances", [])
    
    if not instances:
        return {"error": "No data provided in 'instances' key"}

    # Convert the JSON to a DataFrame that Prophet understands
    df = pd.DataFrame(instances)
    
    # Rename to 'ds' if the user sent 'date'
    if "date" in df.columns:
        df = df.rename(columns={"date": "ds"})
    
    # make the inference
    forecast = model.predict(df)
    
    # Format the output. Extract the date, the prediction (yhat), and the confidence intervals.
    results = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    
    # CConvert timestamps to strings so they can be serialized in JSON
    results["ds"] = results["ds"].astype(str)
    
    # Vertex AI requires that the response be under the "predictions" key
    predictions = results.to_dict(orient="records")
    
    return {"predictions": predictions}