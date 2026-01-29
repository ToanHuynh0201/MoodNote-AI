"""
Prediction utilities for emotion classification
"""
import torch
import numpy as np
from pathlib import Path
from ..models.model_utils import load_model
from ..data.preprocess import VietnamesePreprocessor


class EmotionPredictor:
    """Emotion predictor with Vietnamese preprocessing"""

    def __init__(
        self,
        model_path,
        device='cpu',
        segmenter='pyvi',
        emotion_labels=None
    ):
        """
        Initialize predictor

        Args:
            model_path: Path to saved model
            device: Device to run inference on
            segmenter: Vietnamese word segmenter to use
            emotion_labels: Dictionary mapping label indices to names
        """
        self.device = device
        self.model_path = model_path

        # Default emotion labels
        if emotion_labels is None:
            self.emotion_labels = {
                0: "Enjoyment",
                1: "Sadness",
                2: "Anger",
                3: "Fear",
                4: "Disgust",
                5: "Surprise",
                6: "Other"
            }
        else:
            self.emotion_labels = emotion_labels

        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load_model(model_path, device=device)

        # Initialize preprocessor
        self.preprocessor = VietnamesePreprocessor(segmenter=segmenter)

        print("Predictor initialized successfully!")

    def preprocess_text(self, text):
        """
        Preprocess Vietnamese text

        Args:
            text: Input text

        Returns:
            str: Preprocessed text
        """
        return self.preprocessor.segment_text(text)

    def predict(self, text, return_probabilities=True):
        """
        Predict emotion for a single text

        Args:
            text: Input Vietnamese text
            return_probabilities: Whether to return probabilities for all classes

        Returns:
            dict: Prediction results
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            logits = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Get prediction
        pred_idx = int(np.argmax(probs))
        pred_emotion = self.emotion_labels[pred_idx]
        confidence = float(probs[pred_idx])

        # Create result
        result = {
            'text': text,
            'emotion': pred_emotion,
            'confidence': confidence
        }

        if return_probabilities:
            result['probabilities'] = {
                self.emotion_labels[i]: float(probs[i])
                for i in range(len(probs))
            }

        return result

    def predict_batch(self, texts, return_probabilities=True):
        """
        Predict emotions for a batch of texts

        Args:
            texts: List of Vietnamese texts
            return_probabilities: Whether to return probabilities

        Returns:
            list: List of prediction results
        """
        results = []

        for text in texts:
            result = self.predict(text, return_probabilities=return_probabilities)
            results.append(result)

        return results


if __name__ == "__main__":
    # Test predictor
    print("Testing emotion predictor...")

    # Note: This requires a trained model
    model_path = "models/best_model"

    if Path(model_path).exists():
        predictor = EmotionPredictor(
            model_path=model_path,
            device='cpu'
        )

        # Test texts
        test_texts = [
            "Hôm nay tôi rất vui và hạnh phúc",
            "Tôi cảm thấy buồn và mệt mỏi",
            "Tôi rất tức giận về điều này"
        ]

        print("\nPredictions:")
        for text in test_texts:
            result = predictor.predict(text)
            print(f"\nText: {text}")
            print(f"Emotion: {result['emotion']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: {result['probabilities']}")
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first using scripts/train.py")
