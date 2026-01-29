# Quick Start Guide - MoodNote AI

Hướng dẫn nhanh để bắt đầu sử dụng MoodNote AI.

## Bước 1: Cài đặt Dependencies

```bash
# Kích hoạt virtual environment
.venv\Scripts\activate  # Windows
# hoặc: source .venv/bin/activate  # Linux/Mac

# Cài đặt packages
pip install -r requirements.txt
```

## Bước 2: Download và Preprocess Dataset

```bash
# Download UIT-VSMEC dataset từ Hugging Face
python -m src.data.download_dataset

# Preprocess data với Vietnamese word segmentation
python -m src.data.preprocess
```

**Lưu ý:** Bước preprocess rất quan trọng vì PhoBERT yêu cầu text đã được word-segmented!

## Bước 3: Setup Weights & Biases (Tùy chọn)

Nếu muốn track experiments với W&B:

```bash
# Login W&B
wandb login

# Hoặc thêm API key vào .env
cp .env.example .env
# Sửa WANDB_API_KEY trong .env
```

Nếu không dùng W&B:

```bash
# Train without W&B
python scripts/train.py --no-wandb
```

## Bước 4: Train Model

```bash
# Train model (mặc định 5 epochs)
python scripts/train.py
```

Training sẽ:
- Tải PhoBERT-base model
- Train trên UIT-VSMEC dataset
- Lưu best model vào `models/best_model/`
- Tạo confusion matrix
- Log metrics to W&B (nếu enabled)

**Thời gian:** ~30-60 phút trên GPU, ~2-3 giờ trên CPU

## Bước 5: Test API

```bash
# Chạy API server
python scripts/run_api.py
```

Truy cập API documentation: http://localhost:8000/docs

### Test với cURL:

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hôm nay tôi rất vui"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Tôi vui", "Tôi buồn", "Tôi giận"]}'
```

### Test với Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Hôm nay tôi rất vui và hạnh phúc"}
)

result = response.json()
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Cấu trúc File Quan Trọng

```
MoodNote-AI/
├── configs/
│   ├── model_config.yaml       # Model settings
│   ├── training_config.yaml    # Training hyperparameters
│   └── api_config.yaml         # API settings
├── data/
│   ├── raw/                    # Dataset gốc
│   └── processed/              # Dataset đã preprocess
├── models/
│   └── best_model/             # Model đã train xong
├── scripts/
│   ├── train.py                # Script train model
│   └── run_api.py              # Script chạy API
└── README.md                   # Documentation đầy đủ
```

## Troubleshooting

### "pyvi not found"
```bash
pip install pyvi
```

### "CUDA out of memory"
Giảm batch_size trong `configs/training_config.yaml`:
```yaml
training:
  batch_size: 8  # thay vì 16
```

### "Model not loaded" trong API
Đảm bảo đã train model:
```bash
python scripts/train.py
```

### Dataset không tải được
Kiểm tra internet connection và thử lại:
```bash
python -m src.data.download_dataset
```

## Kết quả Dự kiến

Sau khi train, model sẽ đạt:
- **Accuracy:** 55-65%
- **F1-Macro:** 50-60%
- **F1-Weighted:** 55-65%

Đây là kết quả tốt cho bài toán emotion classification (7 classes) trên Vietnamese text!

## Next Steps

1. **Thử nghiệm với text của bạn:**
   ```bash
   python -m src.inference.predictor
   ```

2. **Tune hyperparameters:**
   - Sửa `configs/training_config.yaml`
   - Thử learning_rate khác, batch_size, num_epochs

3. **Deploy API:**
   - Containerize với Docker
   - Deploy lên cloud (Heroku, AWS, GCP)

4. **Improve model:**
   - Thử PhoBERT-large thay vì base
   - Data augmentation
   - Ensemble models

## Tài liệu

- **README.md:** Documentation đầy đủ
- **API Docs:** http://localhost:8000/docs (khi chạy API)
- **PhoBERT:** https://github.com/VinAIResearch/PhoBERT
- **UIT-VSMEC:** https://huggingface.co/datasets/tridm/UIT-VSMEC

Chúc bạn thành công! 🚀
