"""
FastAPI service for wind power prediction.
Serves predictions via REST API using trained Deep Learning model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import torch
import joblib
import json
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from src.model import DNNRegressor
from src.utils import SEED

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wind Power Prediction API",
    description="Deep Learning API for predicting wind turbine power output",
    version="1.0.0",
)

# Global variables for model artifacts
model = None
scaler = None
feature_cols = None
device = None


class PredictionRequest(BaseModel):
    """
    Request schema for prediction endpoint.
    Accepts a dictionary with all 75 feature values.
    """

    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and their values (75 features required)",
        example={
            "TEMPERATURE": 15.2,
            "WINDSPEED": 8.5,
            "PRESSURE": 1013.2,
            # ... (other features)
        },
    )

    @validator("features")
    def validate_features(cls, v):
        """Validate that all required features are present."""
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction_mw: float = Field(..., description="Predicted power output in MW")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    model_version: str = Field(default="1.0", description="Model version used")
    num_features: int = Field(..., description="Number of features used")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    num_features: int = Field(..., description="Expected number of features")
    device: str = Field(..., description="Computation device (cpu/cuda)")


@app.on_event("startup")
async def load_model_artifacts():
    """
    Load trained model and preprocessing artifacts on API startup.
    This runs once when the API starts, keeping artifacts in memory.
    """
    global model, scaler, feature_cols, device

    logger.info("=" * 50)
    logger.info("Starting Wind Power Prediction API")
    logger.info("=" * 50)

    try:
        # Paths to artifacts
        models_dir = Path("models")
        model_path = models_dir / "best_model.pth"
        scaler_path = models_dir / "scaler.joblib"
        features_path = models_dir / "features.json"

        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Features list not found: {features_path}")

        # Load scaler
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Load feature list
        logger.info(f"Loading feature list from {features_path}")
        with open(features_path, "r") as f:
            feature_cols = json.load(f)

        # Load model
        logger.info(f"Loading model from {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)

        # Initialize model with correct architecture
        input_size = len(feature_cols)
        model = DNNRegressor(
            input_size=input_size, hidden_sizes=[320, 160, 80, 40], dropout_rate=0.28
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode

        logger.info(f"Model loaded successfully!")
        logger.info(f"  Input features: {len(feature_cols)}")
        logger.info(f"  Device: {device}")
        logger.info("=" * 50)
        logger.info("API Ready!")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns API status and model information.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        num_features=len(feature_cols) if feature_cols else 0,
        device=str(device) if device else "unknown",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prediction endpoint.

    Accepts feature values and returns predicted power output in MW.

    Parameters:
    - features: Dictionary with all 75 feature names and their values

    Returns:
    - prediction_mw: Predicted power output in megawatts
    - timestamp: When the prediction was made
    - model_version: Version of the model used
    - num_features: Number of features processed
    """
    try:
        # Verify model is loaded
        if model is None or scaler is None or feature_cols is None:
            raise HTTPException(
                status_code=503, detail="Model not loaded. API is not ready."
            )

        # Extract features in correct order
        try:
            # Create feature array in the same order as training
            feature_values = [request.features.get(col, None) for col in feature_cols]

            # Check for missing features
            missing_features = [
                col for col, val in zip(feature_cols, feature_values) if val is None
            ]

            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required features: {missing_features[:5]}... ({len(missing_features)} total)",
                )

            # Convert to numpy array
            X = np.array([feature_values], dtype=np.float32)

        except KeyError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid feature name: {str(e)}"
            )

        # Scale features
        X_scaled = scaler.transform(X)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        # Make prediction
        with torch.no_grad():
            prediction_log = model(X_tensor).cpu().item()

        # Inverse log transform
        prediction_mw = float(np.expm1(prediction_log))

        # Create response
        response = PredictionResponse(
            prediction_mw=round(prediction_mw, 2),
            timestamp=datetime.now().isoformat(),
            model_version="1.0",
            num_features=len(feature_cols),
        )

        logger.info(f"Prediction made: {prediction_mw:.2f} MW")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/features")
async def get_required_features():
    """
    Returns the list of required feature names for predictions.
    Useful for API clients to know what features to send.
    """
    if feature_cols is None:
        raise HTTPException(status_code=503, detail="Feature list not loaded")

    return {"num_features": len(feature_cols), "features": feature_cols}


@app.get("/info")
async def get_model_info():
    """
    Returns detailed information about the loaded model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "Deep Neural Network (DNN)",
        "architecture": {
            "input_size": len(feature_cols),
            "hidden_layers": [320, 160, 80, 40],
            "dropout_rate": 0.28,
            "output_size": 1,
        },
        "target": "Wind Power Output (MW)",
        "preprocessing": {"scaling": "StandardScaler", "target_transform": "log1p"},
        "device": str(device),
        "num_features": len(feature_cols) if feature_cols else 0,
    }


if __name__ == "__main__":
    import uvicorn

    # Run the API
    uvicorn.run(
        "app_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info",
    )
