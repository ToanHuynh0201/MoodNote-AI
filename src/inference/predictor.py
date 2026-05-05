"""
Prediction utilities for emotion classification
"""
import re
import unicodedata
import torch
import numpy as np
from pathlib import Path
from ..models.model_utils import load_model
from ..data.preprocess import VietnamesePreprocessor
from ..utils.keyword_extractor import VietnameseKeywordExtractor
from ..utils.logger import get_logger
from ..utils.emotion_constants import (
    normalize_emotion_labels,
    normalize_sentiment_scores,
    find_label_index_by_name,
)

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?…])\s+|\.\.\.\s+|\n+')


logger = get_logger("predictor")


class EmotionPredictor:
    """Emotion predictor with Vietnamese preprocessing"""

    def __init__(
        self,
        model_path,
        device='cpu',
        segmenter='pyvi',
        emotion_labels=None,
        sentiment_scores=None
    ):
        """
        Initialize predictor

        Args:
            model_path: Path to saved model
            device: Device to run inference on
            segmenter: Vietnamese word segmenter to use
            emotion_labels: Dictionary mapping label indices to names
            sentiment_scores: Dictionary mapping emotion names to sentiment values (-1 to +1)
        """
        self.device = device
        self.model_path = model_path

        self.emotion_labels = normalize_emotion_labels(emotion_labels)
        self.other_label_index = find_label_index_by_name(self.emotion_labels, "Other")
        if self.other_label_index is None:
            logger.warning('No "Other" label found in emotion_labels; low-confidence fallback is disabled.')

        self.sentiment_scores = normalize_sentiment_scores(sentiment_scores)

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load_model(model_path, device=device)

        # Initialize preprocessor
        self.preprocessor = VietnamesePreprocessor(segmenter=segmenter)

        # Initialize keyword extractor (FR-13)
        self.keyword_extractor = VietnameseKeywordExtractor(max_keywords=10)

        logger.info("Predictor initialized successfully!")

    def preprocess_text(self, text):
        """
        Preprocess Vietnamese text

        Args:
            text: Input text

        Returns:
            str: Preprocessed text
        """
        return self.preprocessor.segment_text(text)

    def _is_meaningful_text(self, text: str) -> bool:
        """Kiểm tra text có đủ nội dung chữ để phân tích cảm xúc không.
        Trả về False nếu text chỉ là ngày tháng, số, hoặc ký tự đặc biệt.
        """
        letter_count = sum(1 for c in text if unicodedata.category(c).startswith('L'))
        return letter_count >= 5

    def _empty_result(self, text: str, return_probabilities: bool = True) -> dict:
        """Trả về kết quả rỗng cho text không có nội dung cảm xúc."""
        result = {
            'text': text,
            'emotion': 'Other',
            'confidence': 0.0,
            'sentiment_score': 0.0,
            'intensity': 0.0,
            'keywords': []
        }
        if return_probabilities:
            result['probabilities'] = {name: 0.0 for name in self.emotion_labels.values()}
        return result

    def predict(self, text, return_probabilities=True, other_threshold=0.0):
        """
        Predict emotion for a single text

        Args:
            text: Input Vietnamese text
            return_probabilities: Whether to return probabilities for all classes
            other_threshold: If max(softmax) < threshold, predict "Other" instead.
                             Set > 0 to catch low-confidence predictions as "Other"
                             (e.g. 0.40 works well for this model).
                             Default 0.0 = disabled (original argmax behavior).

        Returns:
            dict: Prediction results
        """
        # Guard: skip non-meaningful text (dates, numbers, etc.)
        if not self._is_meaningful_text(text):
            return self._empty_result(text, return_probabilities)

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
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

        # Get probabilities
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probs = torch.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Get prediction
        max_prob = float(np.max(probs))
        if (
            other_threshold > 0
            and max_prob < other_threshold
            and self.other_label_index is not None
        ):
            pred_idx = self.other_label_index
        else:
            pred_idx = int(np.argmax(probs))
        pred_emotion = self.emotion_labels.get(pred_idx, str(pred_idx))
        confidence = float(probs[pred_idx])

        # FR-11: Sentiment score (-1.0 to +1.0) — weighted sum of emotion sentiments
        sentiment_score = float(sum(
            self.sentiment_scores.get(self.emotion_labels.get(i, ""), 0.0) * float(prob)
            for i, prob in enumerate(probs)
        ))
        sentiment_score = round(sentiment_score, 4)

        # FR-12: Intensity (0-100) — extremity of predicted emotion (|sentiment_weight| × 100)
        # Other → 0, Anger/Enjoyment → high, regardless of model confidence
        extremity = abs(self.sentiment_scores.get(pred_emotion, 0.0))
        intensity = round(float(extremity * 100), 2)

        # FR-13: Keyword extraction (3-10 keywords)
        keywords = self.keyword_extractor.extract(text, n=5)

        # Create result
        result = {
            'text': text,
            'emotion': pred_emotion,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'intensity': intensity,
            'keywords': keywords
        }

        if return_probabilities:
            result['probabilities'] = {
                self.emotion_labels.get(i, str(i)): float(prob)
                for i, prob in enumerate(probs)
            }

        return result

    def _split_sentences(self, text, min_length=3):
        """Tách văn bản nhật ký thành danh sách câu."""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        parts = _SENT_SPLIT_RE.split(text)
        return [p.strip() for p in parts if len(p.strip()) >= min_length]

    def _aggregate_diary_results(self, results):
        """Tổng hợp kết quả phân tích từng câu thành kết quả chung cho toàn đoạn nhật ký."""
        emotion_names = list(self.emotion_labels.values())

        # Trọng số theo độ tin cậy của từng câu
        weights = [r['confidence'] for r in results]
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(results)
            total_weight = float(len(results))

        # Trung bình xác suất có trọng số
        num_classes = len(emotion_names)
        p_avg = [0.0] * num_classes
        for w, r in zip(weights, results):
            for k, name in enumerate(emotion_names):
                p_avg[k] += w * r['probabilities'].get(name, 0.0)
        p_avg = [v / total_weight for v in p_avg]

        best_idx = int(p_avg.index(max(p_avg)))
        overall_emotion = emotion_names[best_idx]
        overall_confidence = round(p_avg[best_idx], 4)

        # Sentiment và intensity có trọng số
        overall_sentiment = round(
            sum(w * r['sentiment_score'] for w, r in zip(weights, results)) / total_weight, 4
        )
        overall_intensity = round(
            sum(w * r['intensity'] for w, r in zip(weights, results)) / total_weight, 2
        )

        emotion_distribution = {name: round(p_avg[k], 2) for k, name in enumerate(emotion_names)}

        return {
            'overall_emotion':      overall_emotion,
            'overall_confidence':   overall_confidence,
            'overall_sentiment':    overall_sentiment,
            'overall_intensity':    overall_intensity,
            'emotion_distribution': emotion_distribution,
        }

    def predict_diary(self, text, other_threshold=0.0, min_sentence_length=3, keyword_count=10):
        """
        Phân tích cảm xúc cho toàn bộ đoạn nhật ký.

        Tách đoạn nhật ký thành câu, dự đoán cảm xúc từng câu,
        sau đó tổng hợp kết quả chung cho toàn bộ đoạn.

        Args:
            text: Văn bản nhật ký tiếng Việt (nhiều câu, có thể có xuống dòng).
            other_threshold: Ngưỡng tin cậy tối thiểu, dưới ngưỡng thì trả về "Other".
            min_sentence_length: Số ký tự tối thiểu để coi là câu hợp lệ.
            keyword_count: Số từ khóa trích xuất từ toàn bộ đoạn (3-10).

        Returns:
            dict với các trường: overall_emotion, overall_confidence, overall_sentiment,
            overall_intensity, emotion_distribution, keywords, sentence_count, sentences.
        """
        if not text.strip():
            raise ValueError("Văn bản nhật ký không được rỗng")

        sentences = self._split_sentences(text, min_length=min_sentence_length)
        sentences = [s for s in sentences if self._is_meaningful_text(s)]
        if not sentences:
            raise ValueError("Không tìm thấy câu hợp lệ trong đoạn nhật ký")

        results = [
            self.predict(s, return_probabilities=True, other_threshold=other_threshold)
            for s in sentences
        ]

        agg = self._aggregate_diary_results(results)
        keywords = self.keyword_extractor.extract(text, n=keyword_count)

        return {
            'overall_emotion':      agg['overall_emotion'],
            'overall_confidence':   agg['overall_confidence'],
            'overall_sentiment':    agg['overall_sentiment'],
            'overall_intensity':    agg['overall_intensity'],
            'emotion_distribution': agg['emotion_distribution'],
            'keywords':             keywords,
            'sentence_count':       len(results),
            'sentences':            results,
        }

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
