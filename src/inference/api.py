"""
FastAPI application for emotion prediction
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..utils.config import load_config
from ..utils.database import close_client, get_training_collection
from ..utils.logger import get_logger
from .predictor import MAX_SEQ_LENGTH, EmotionPredictor

logger = get_logger("api")

# Initialize FastAPI app
app = FastAPI(
    title="MoodNote AI - Emotion Classification API",
    description="Vietnamese emotion classification API using PhoBERT",
    version="1.0.0",
)

# Global predictor instance
predictor: EmotionPredictor | None = None


def _success(data: Any, message: str = "OK") -> dict[str, Any]:
    """Wrap a payload in the standard success envelope used by every endpoint."""
    return {"success": True, "message": message, "data": data}


def _require_predictor(detail: str = "Model not loaded") -> EmotionPredictor:
    """Return the loaded predictor or raise 503 if it failed to initialize."""
    if predictor is None:
        raise HTTPException(status_code=503, detail=detail)
    return predictor


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content={"success": False, "message": exc.detail}
    )


# Pydantic models
class PredictionRequest(BaseModel):
    """Single prediction request"""

    text: str = Field(..., description="Vietnamese text to analyze", min_length=1)

    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "Hôm nay tôi rất vui và hạnh phúc"}}
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""

    texts: list[str] = Field(..., description="List of Vietnamese texts to analyze", min_length=1)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": ["Hôm nay tôi rất vui", "Tôi cảm thấy buồn", "Điều này khiến tôi tức giận"]
            }
        }
    )


class PredictionResponse(BaseModel):
    """Prediction response"""

    text: str
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    sentiment_score: float
    intensity: float
    keywords: list[str]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""

    predictions: list[PredictionResponse]
    count: int


class SentencePrediction(BaseModel):
    """Kết quả phân tích từng câu trong đoạn nhật ký"""

    index: int
    text: str
    emotion: str
    confidence: float
    sentiment_score: float
    intensity: float
    probabilities: dict[str, float]


class TagItem(BaseModel):
    name: str
    type: str


class DiaryAnalysisRequest(BaseModel):
    """Yêu cầu phân tích đoạn nhật ký"""

    text: str = Field(..., description="Đoạn nhật ký tiếng Việt cần phân tích", min_length=1)
    keyword_count: int = Field(default=10, ge=3, le=10, description="Số từ khóa trích xuất (3-10)")
    other_threshold: float = Field(
        default=0.0, ge=0.0, lt=1.0, description="Ngưỡng tin cậy tối thiểu (0.0 = tắt)"
    )
    # Training fields — optional, ignored nếu không gửi
    allow_training: bool = False
    input_method: str = "TEXT"
    word_count: int = Field(default=0, ge=0)
    entry_date: str | None = None  # "YYYY-MM-DD"
    tags: list[TagItem] = []

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Hôm nay tôi rất mệt mỏi. Công việc quá nhiều khiến tôi căng thẳng. Nhưng tối về nhà thấy gia đình, tôi lại vui hơn.",
                "keyword_count": 5,
                "other_threshold": 0.0,
            }
        }
    )


class DiaryAnalysisResponse(BaseModel):
    """Kết quả phân tích toàn bộ đoạn nhật ký"""

    overall_emotion: str
    overall_confidence: float
    overall_sentiment: float
    overall_intensity: float
    emotion_distribution: dict[str, float]
    keywords: list[str]
    sentence_count: int
    sentences: list[SentencePrediction]


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
    emotion_labels: dict[int, str]
    device: str


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize model on startup"""
    global predictor

    logger.info("Starting API server...")

    try:
        # Load API config
        api_config = load_config("configs/api_config.yaml")
        model_config = load_config("configs/model_config.yaml")

        model_path = api_config["model"]["path"]
        device = api_config["model"].get("device", "cpu")
        preprocessing = api_config.get("preprocessing", {})

        # Auto-detect device if set to cuda but not available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device = "cpu"

        # Initialize predictor
        logger.info(f"Loading model from {model_path}...")
        predictor = EmotionPredictor(
            model_path=model_path,
            device=device,
            segmenter=preprocessing.get("segmenter", "pyvi"),
            emotion_labels=model_config["emotion_labels"],
            sentiment_scores=model_config.get("sentiment_scores"),
            max_length=preprocessing.get("max_length", MAX_SEQ_LENGTH),
        )

        logger.info("API server started successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but predictions will fail until model is loaded")


@app.get("/", tags=["General"])
async def root() -> dict[str, Any]:
    """Root endpoint"""
    return _success(
        {"version": "1.0.0", "docs": "/docs"},
        message="Welcome to MoodNote AI - Vietnamese Emotion Classification API",
    )


@app.get("/health", tags=["General"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint"""
    model_loaded = predictor is not None
    device = str(predictor.device) if predictor is not None else "unknown"

    return _success(
        HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            device=device,
        ).model_dump()
    )


@app.get("/model/info", tags=["Model"])
async def model_info() -> dict[str, Any]:
    """Get model information"""
    engine = _require_predictor()

    return _success(
        ModelInfoResponse(
            model_name="PhoBERT",
            model_path=engine.model_path,
            num_labels=len(engine.emotion_labels),
            emotion_labels=engine.emotion_labels,
            device=str(engine.device),
        ).model_dump()
    )


@app.post("/predict", tags=["Prediction"])
async def predict(request: PredictionRequest) -> dict[str, Any]:
    """
    Predict emotion for a single text

    Args:
        request: Prediction request with Vietnamese text

    Returns:
        Prediction result with emotion, confidence, and probabilities
    """
    engine = _require_predictor()

    try:
        result = engine.predict(request.text, return_probabilities=True)
        return _success(
            PredictionResponse(**result).model_dump(),
            message="Prediction successful",
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> dict[str, Any]:
    """
    Predict emotions for multiple texts

    Args:
        request: Batch prediction request with list of Vietnamese texts

    Returns:
        Batch prediction results
    """
    engine = _require_predictor()

    try:
        results = engine.predict_batch(request.texts, return_probabilities=True)
        predictions = [PredictionResponse(**result) for result in results]

        return _success(
            {
                "predictions": [p.model_dump() for p in predictions],
                "count": len(predictions),
            },
            message="Batch prediction successful",
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}") from e


@app.post("/predict/diary", tags=["Prediction"])
async def predict_diary(
    request: DiaryAnalysisRequest, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """
    Phân tích cảm xúc cho toàn bộ đoạn nhật ký.

    Tách đoạn nhật ký thành câu, phân loại cảm xúc từng câu,
    và trả về kết quả tổng hợp cùng timeline cảm xúc theo từng câu.

    - **text**: Đoạn nhật ký tiếng Việt (hỗ trợ nhiều đoạn văn)
    - **keyword_count**: Số từ khóa trích xuất từ toàn bộ đoạn (3-10)
    - **other_threshold**: Ngưỡng tin cậy tối thiểu trước khi fallback về "Other"
    """
    engine = _require_predictor(detail="Model chưa được tải")

    try:
        result = engine.predict_diary(
            text=request.text,
            other_threshold=request.other_threshold,
            keyword_count=request.keyword_count,
        )

        sentences = [SentencePrediction(index=i, **s) for i, s in enumerate(result["sentences"])]

        response = DiaryAnalysisResponse(
            overall_emotion=result["overall_emotion"],
            overall_confidence=result["overall_confidence"],
            overall_sentiment=result["overall_sentiment"],
            overall_intensity=result["overall_intensity"],
            emotion_distribution=result["emotion_distribution"],
            keywords=result["keywords"],
            sentence_count=result["sentence_count"],
            sentences=sentences,
        )

        if request.allow_training:
            background_tasks.add_task(
                _save_training_sample,
                text=request.text,
                input_method=request.input_method,
                word_count=request.word_count,
                entry_date=request.entry_date,
                tags=[t.model_dump() for t in request.tags],
                analysis=response.model_dump(),
            )

        return _success(response.model_dump(), message="Phân tích nhật ký thành công")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Lỗi phân tích nhật ký: {e}")
        raise HTTPException(status_code=500, detail=f"Phân tích thất bại: {str(e)}") from e


@app.on_event("shutdown")
def shutdown_event() -> None:
    close_client()


def _save_training_sample(
    text: str,
    input_method: str,
    word_count: int,
    entry_date: str | None,
    tags: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> None:
    """Persist a diary analysis as a training sample (best-effort, non-critical)."""
    try:
        doc = {
            "text": text,
            "wordCount": word_count,
            "inputMethod": input_method,
            "entryDate": entry_date,
            "tags": tags,
            "primaryEmotion": analysis["overall_emotion"],
            "sentimentScore": analysis["overall_sentiment"],
            "intensity": analysis["overall_intensity"],
            "confidence": analysis["overall_confidence"],
            "emotionDistribution": analysis["emotion_distribution"],
            "keywords": analysis["keywords"],
            "sentenceCount": analysis["sentence_count"],
            "createdAt": datetime.now(timezone.utc),
        }
        get_training_collection().insert_one(doc)
    except Exception as e:
        logger.warning(f"[Training] Failed to save sample (non-critical): {e}")
