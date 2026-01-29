"""
FastAPI application for emotion prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
from pathlib import Path
from .predictor import EmotionPredictor
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger("api")

# Initialize FastAPI app
app = FastAPI(
    title="MoodNote AI - Emotion Classification API",
    description="Vietnamese emotion classification API using PhoBERT",
    version="1.0.0"
)

# Global predictor instance
predictor: Optional[EmotionPredictor] = None


# Pydantic models
class PredictionRequest(BaseModel):
    """Single prediction request"""
    text: str = Field(..., description="Vietnamese text to analyze", min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hôm nay tôi rất vui và hạnh phúc"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    texts: List[str] = Field(..., description="List of Vietnamese texts to analyze", min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Hôm nay tôi rất vui",
                    "Tôi cảm thấy buồn",
                    "Điều này khiến tôi tức giận"
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    text: str
    emotion: str
    confidence: float
    probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    model_path: str
    num_labels: int
    emotion_labels: Dict[int, str]
    device: str


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor

    logger.info("Starting API server...")

    try:
        # Load API config
        api_config = load_config("configs/api_config.yaml")
        model_config = load_config("configs/model_config.yaml")

        model_path = api_config['model']['path']
        device = api_config['model'].get('device', 'cpu')

        # Auto-detect device if set to cuda but not available
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device = 'cpu'

        # Initialize predictor
        logger.info(f"Loading model from {model_path}...")
        predictor = EmotionPredictor(
            model_path=model_path,
            device=device,
            segmenter=api_config['preprocessing']['segmenter'],
            emotion_labels=model_config['emotion_labels']
        )

        logger.info("API server started successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but predictions will fail until model is loaded")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MoodNote AI - Vietnamese Emotion Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    model_loaded = predictor is not None

    device = "unknown"
    if predictor is not None:
        device = str(predictor.device)

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        device=device
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model information"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name="PhoBERT",
        model_path=predictor.model_path,
        num_labels=len(predictor.emotion_labels),
        emotion_labels=predictor.emotion_labels,
        device=str(predictor.device)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict emotion for a single text

    Args:
        request: Prediction request with Vietnamese text

    Returns:
        Prediction result with emotion, confidence, and probabilities
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.predict(request.text, return_probabilities=True)
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict emotions for multiple texts

    Args:
        request: Batch prediction request with list of Vietnamese texts

    Returns:
        Batch prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = predictor.predict_batch(request.texts, return_probabilities=True)

        predictions = [PredictionResponse(**result) for result in results]

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run API server
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
