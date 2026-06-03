"""Smoke tests for the FastAPI app using a fake predictor (no real model load).

A bare ``TestClient(app)`` (not used as a context manager) does not trigger the
startup event, so the heavyweight model is never loaded. We inject a fake
predictor via monkeypatching the module-level ``predictor`` global instead.
"""

import pytest
from fastapi.testclient import TestClient

from src.inference import api

EMOTION_LABELS = {0: "Enjoyment", 1: "Sadness", 6: "Other"}
PROBS = {"Enjoyment": 0.7, "Sadness": 0.2, "Other": 0.1}


class FakePredictor:
    """Minimal stand-in matching the interface api.py relies on."""

    device = "cpu"
    model_path = "models/best_model"
    emotion_labels = EMOTION_LABELS

    def _single(self, text):
        return {
            "text": text,
            "emotion": "Enjoyment",
            "confidence": 0.7,
            "sentiment_score": 0.5,
            "intensity": 42.0,
            "keywords": ["vui"],
            "probabilities": dict(PROBS),
        }

    def predict(self, text, return_probabilities=True):
        return self._single(text)

    def predict_batch(self, texts, return_probabilities=True):
        return [self._single(t) for t in texts]

    def predict_diary(self, text, other_threshold=0.0, keyword_count=10):
        return {
            "overall_emotion": "Enjoyment",
            "overall_confidence": 0.7,
            "overall_sentiment": 0.5,
            "overall_intensity": 42.0,
            "emotion_distribution": {"Enjoyment": 1.0},
            "keywords": ["vui"],
            "sentence_count": 1,
            "sentences": [
                {
                    "text": "Hôm nay tôi rất vui",
                    "emotion": "Enjoyment",
                    "confidence": 0.7,
                    "sentiment_score": 0.5,
                    "intensity": 42.0,
                    "probabilities": dict(PROBS),
                }
            ],
        }


@pytest.fixture
def client_with_model(monkeypatch):
    monkeypatch.setattr(api, "predictor", FakePredictor())
    return TestClient(api.app)


@pytest.fixture
def client_without_model(monkeypatch):
    monkeypatch.setattr(api, "predictor", None)
    return TestClient(api.app)


def test_root(client_with_model):
    resp = client_with_model.get("/")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_health_reports_loaded(client_with_model):
    body = client_with_model.get("/health").json()
    assert body["success"] is True
    assert body["data"]["model_loaded"] is True
    assert body["data"]["status"] == "healthy"


def test_health_reports_degraded_without_model(client_without_model):
    body = client_without_model.get("/health").json()
    assert body["data"]["model_loaded"] is False
    assert body["data"]["status"] == "degraded"


def test_predict_response_shape(client_with_model):
    resp = client_with_model.post("/predict", json={"text": "Hôm nay tôi rất vui"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert set(data) == {
        "text",
        "emotion",
        "confidence",
        "probabilities",
        "sentiment_score",
        "intensity",
        "keywords",
    }
    assert data["emotion"] == "Enjoyment"


def test_predict_requires_model(client_without_model):
    resp = client_without_model.post("/predict", json={"text": "xin chào"})
    assert resp.status_code == 503


def test_predict_validates_empty_text(client_with_model):
    resp = client_with_model.post("/predict", json={"text": ""})
    assert resp.status_code == 422  # violates min_length=1


def test_batch_predict(client_with_model):
    resp = client_with_model.post("/predict/batch", json={"texts": ["a", "b"]})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_diary_response_shape(client_with_model):
    resp = client_with_model.post("/predict/diary", json={"text": "Hôm nay tôi rất vui."})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["overall_emotion"] == "Enjoyment"
    assert data["sentence_count"] == 1
    assert data["sentences"][0]["index"] == 0
