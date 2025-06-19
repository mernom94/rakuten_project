from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rakuten Product Category API",
    description="API for training and predicting product categories using MLflow models",
    version="1.0.0"
)

# ----------- Pydantic Schemas -----------

class TrainingRequest(BaseModel):
    parameters: Dict[str, Any] = {}
    data_path: str = None  

class TrainingResponse(BaseModel):
    status: str
    message: str
    model_id: str = None
    metrics: Dict[str, float] = {}

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_id: str = None 

class PredictionResponse(BaseModel):
    predictions: List[Any]
    model_id: str
    status: str

# ----------- Training State -----------

training_status = {"is_training": False, "last_result": None}

# ----------- API Routes -----------

@app.get("/")
async def root():
    return {"message": "Rakuten ML API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "training_active": training_status["is_training"],
        "services": {
            "api": "running",
            "mlflow_tracking": mlflow.get_tracking_uri()
        }
    }

@app.get("/models/")
async def list_models():
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    models = client.list_registered_models()
    model_names = [m.name for m in models]
    return {
        "models": model_names,
        "message": "Model registry from MLflow tracking server"
    }

@app.post("/training/", response_model=TrainingResponse)
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    if training_status["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    try:
        training_status["is_training"] = True
        logger.info("Starting training...")

        # ✅ Replace below with call to your actual training function
        # For now, simulate a training run and log a dummy model
        import tempfile
        from sklearn.linear_model import LogisticRegression

        # Dummy data
        X = pd.DataFrame([[1, 2], [3, 4]], columns=["feat1", "feat2"])
        y = [0, 1]
        model = LogisticRegression().fit(X, y)

        mlflow.set_experiment("rakuten-product-classification")
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="rakuten_classifier")
            mlflow.log_metric("accuracy", 0.95)

        model_id = "rakuten_classifier"
        training_status["last_result"] = {"model_id": model_id, "accuracy": 0.95}
        training_status["is_training"] = False

        return TrainingResponse(
            status="success",
            message="Model trained and logged to MLflow",
            model_id=model_id,
            metrics={"accuracy": 0.95}
        )

    except Exception as e:
        training_status["is_training"] = False
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/training/status")
async def get_training_status():
    return {
        "is_training": training_status["is_training"],
        "last_result": training_status["last_result"]
    }

@app.post("/predict/", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    try:
        model_id = request.model_id or "rakuten_classifier"
        model_uri = f"models:/{model_id}/latest"

        logger.info(f"Loading model from MLflow: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        input_df = pd.DataFrame(request.data)
        preds = model.predict(input_df).tolist()

        return PredictionResponse(
            predictions=preds,
            model_id=model_id,
            status="success"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
