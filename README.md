# MoodNote AI - Vietnamese Emotion Classification

PhoBERT-based emotion classification system for Vietnamese diary entries using the UIT-VSMEC dataset.

## Overview

This project fine-tunes PhoBERT (Vietnamese BERT) on the UIT-VSMEC dataset to classify Vietnamese text into 7 emotion categories:

- **Enjoyment** (Vui vẻ)
- **Sadness** (Buồn bã)
- **Anger** (Tức giận)
- **Fear** (Sợ hãi)
- **Disgust** (Ghê tởm)
- **Surprise** (Ngạc nhiên)
- **Other** (Khác)

## Features

- PhoBERT-based emotion classification model
- Vietnamese word segmentation preprocessing (critical for PhoBERT)
- Automatic dataset download from Hugging Face
- Training with Weights & Biases experiment tracking
- Comprehensive evaluation metrics (F1-macro, F1-weighted, per-class metrics)
- FastAPI REST API for inference
- Confusion matrix visualization

## Project Structure

```
MoodNote-AI/
├── configs/                  # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── api_config.yaml
├── data/                     # Dataset directory
│   ├── raw/                  # Raw downloaded data
│   ├── processed/            # Preprocessed data
│   └── cache/                # Cached tokenized data
├── src/                      # Source code
│   ├── data/                 # Data processing
│   ├── models/               # Model definitions
│   ├── training/             # Training logic
│   ├── inference/            # Inference & API
│   └── utils/                # Utilities
├── scripts/                  # Executable scripts
│   ├── train.py
│   └── run_api.py
├── models/                   # Saved models
│   ├── checkpoints/
│   └── best_model/
└── logs/                     # Training logs
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## Installation

### 1. Clone Repository

```bash
cd MoodNote-AI
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Weights & Biases (Optional)

If you want to use W&B for experiment tracking:

```bash
wandb login
```

Create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your WANDB_API_KEY
```

## Quick Start

### Step 1: Download Dataset

```bash
python -m src.data.download_dataset
```

This will download the UIT-VSMEC dataset from Hugging Face and save it to `data/raw/`.

### Step 2: Preprocess Data

```bash
python -m src.data.preprocess
```

This applies Vietnamese word segmentation (critical for PhoBERT) and saves preprocessed data to `data/processed/`.

**Note:** This step uses `pyvi` for word segmentation. Example:
- Input: "Hôm nay tôi rất vui"
- Output: "Hôm_nay tôi rất vui"

### Step 3: Train Model

```bash
python scripts/train.py
```

Training options:

```bash
# Without W&B logging
python scripts/train.py --no-wandb

# Custom output directory
python scripts/train.py --output-dir custom_checkpoints
```

Training will:
- Load preprocessed data
- Initialize PhoBERT model
- Train for 5 epochs (default)
- Evaluate on validation set
- Save best model to `models/best_model/`
- Generate confusion matrix

**Expected Results:**
- Accuracy: 55-65%
- F1-Macro: 50-60%
- F1-Weighted: 55-65%

### Step 4: Run Inference API

```bash
python scripts/run_api.py
```

The API will be available at:
- **API Endpoint:** http://localhost:8000
- **Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hôm nay tôi rất vui và hạnh phúc"
  }'
```

Response:

```json
{
  "text": "Hôm nay tôi rất vui và hạnh phúc",
  "emotion": "Enjoyment",
  "confidence": 0.92,
  "probabilities": {
    "Enjoyment": 0.92,
    "Sadness": 0.03,
    "Anger": 0.01,
    "Fear": 0.01,
    "Disgust": 0.01,
    "Surprise": 0.01,
    "Other": 0.01
  }
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Tôi rất vui",
      "Tôi buồn lắm",
      "Điều này khiến tôi tức giận"
    ]
  }'
```

### Python Client Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Hôm nay tôi rất vui"}
)
result = response.json()
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2f}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": [
            "Tôi vui quá",
            "Tôi buồn",
            "Tôi giận lắm"
        ]
    }
)
results = response.json()
for pred in results['predictions']:
    print(f"{pred['text']} -> {pred['emotion']}")
```

## Configuration

### Model Configuration ([configs/model_config.yaml](configs/model_config.yaml))

```yaml
model:
  name: "vinai/phobert-base"      # PhoBERT model
  num_labels: 7                    # Number of emotions
  max_seq_length: 128              # Max sequence length
  dropout: 0.1                     # Dropout rate

preprocessing:
  segmenter: "pyvi"                # Word segmenter
  lowercase: false
```

### Training Configuration ([configs/training_config.yaml](configs/training_config.yaml))

```yaml
training:
  learning_rate: 3e-5              # Learning rate
  batch_size: 16                   # Batch size
  num_epochs: 5                    # Number of epochs
  warmup_steps: 500                # Warmup steps
  weight_decay: 0.01               # Weight decay
  fp16: true                       # Mixed precision

wandb:
  project: "moodnote-ai"
  enabled: true
```

## Dataset

**UIT-VSMEC** (Vietnamese Social Media Emotion Corpus)
- **Source:** Hugging Face (`tridm/UIT-VSMEC`)
- **Total samples:** 6,927 Vietnamese Facebook posts
- **Splits:**
  - Train: 5,550 samples
  - Validation: 686 samples
  - Test: 693 samples
- **Labels:** 7 emotion classes

## Model Architecture

```
PhoBERT Base (vinai/phobert-base)
    ↓
[CLS] token representation (768-dim)
    ↓
Dropout (0.1)
    ↓
Linear Layer (768 → 7)
    ↓
Softmax
```

**Parameters:** ~135M (PhoBERT-base)

## Training Details

- **Optimizer:** AdamW
- **Learning Rate:** 3e-5
- **Scheduler:** Linear with warmup
- **Batch Size:** 16
- **Epochs:** 5
- **Mixed Precision:** FP16 (optional)
- **Early Stopping:** Patience = 3

## Evaluation Metrics

The model is evaluated using:

- **Accuracy:** Overall correctness
- **F1-Macro:** Equal weight to all classes
- **F1-Weighted:** Weighted by class frequency
- **Per-class Precision/Recall/F1**
- **Confusion Matrix**

## Important Notes

### Vietnamese Word Segmentation

PhoBERT **requires** word-segmented input. This project uses `pyvi` for fast segmentation:

- Input: "Hôm nay tôi học bài"
- Output: "Hôm_nay tôi học bài"

**Do not skip this step!** The model will not work correctly without proper word segmentation.

### GPU Support

The code automatically detects CUDA availability:

```python
# Training uses GPU if available
python scripts/train.py  # Auto-detects GPU

# Force CPU
python scripts/train.py --device cpu
```

### Experiment Tracking

Training metrics are logged to Weights & Biases by default. To disable:

```bash
python scripts/train.py --no-wandb
```

## Troubleshooting

### Issue: "pyvi not found"

```bash
pip install pyvi
```

### Issue: "CUDA out of memory"

Reduce batch size in [configs/training_config.yaml](configs/training_config.yaml):

```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Issue: "Dataset not found"

Make sure you've downloaded and preprocessed the data:

```bash
python -m src.data.download_dataset
python -m src.data.preprocess
```

### Issue: "Model not loaded" in API

Ensure the model has been trained and saved to `models/best_model/`:

```bash
python scripts/train.py
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/
```

## License

This project is for educational purposes.

## Acknowledgments

- **PhoBERT:** [VinAI Research](https://github.com/VinAIResearch/PhoBERT)
- **UIT-VSMEC Dataset:** [UIT NLP Group](https://nlp.uit.edu.vn/datasets)
- **Hugging Face Transformers:** [Hugging Face](https://huggingface.co/)

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{phobert,
  title={PhoBERT: Pre-trained language models for Vietnamese},
  author={Nguyen, Dat Quoc and Nguyen, Anh Tuan},
  booktitle={Findings of EMNLP},
  year={2020}
}

@article{uit-vsmec,
  title={Emotion Recognition for Vietnamese Social Media Text},
  author={Ho, Vong Anh and Nguyen, Duong Huynh-Cong and others},
  year={2020}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
