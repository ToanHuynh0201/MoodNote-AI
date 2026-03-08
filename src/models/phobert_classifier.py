"""
PhoBERT model for emotion classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced classification.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=0 → standard CrossEntropyLoss
    gamma=2 → focuses training on hard/misclassified examples
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class PhoBERTEmotionClassifier(nn.Module):
    """
    PhoBERT-based emotion classifier

    Architecture:
        PhoBERT Base/Large
            ↓
        [CLS] token representation
            ↓
        Dropout (0.1)
            ↓
        Linear Layer (hidden_size → num_labels)
    """

    def __init__(
        self,
        model_name="vinai/phobert-base",
        num_labels=7,
        dropout=0.1,
        freeze_bert=False,
        class_weights=None,
        label_smoothing=0.0,
        focal_gamma=2.0
    ):
        """
        Initialize PhoBERT classifier

        Args:
            model_name: Name of pretrained PhoBERT model
            num_labels: Number of emotion classes
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT parameters
            label_smoothing: Label smoothing factor (set 0.0 to disable)
            focal_gamma: Gamma for Focal Loss (set 0.0 to use CrossEntropyLoss)
        """
        super(PhoBERTEmotionClassifier, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma

        # Load PhoBERT model
        self.bert = AutoModel.from_pretrained(model_name)

        # Get hidden size from config (768 for base, 1024 for large)
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # Class weights for imbalanced dataset
        self.class_weights = class_weights

        # Optionally freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Ground truth labels (batch_size,) — optional

        Returns:
            SequenceClassifierOutput with loss (if labels provided) and logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            if self.focal_gamma > 0:
                loss = FocalLoss(gamma=self.focal_gamma, weight=weight)(logits, labels)
            else:
                loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def get_num_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config):
    """
    Create PhoBERT model from config

    Args:
        config: Model configuration dictionary

    Returns:
        PhoBERTEmotionClassifier: Initialized model
    """
    model = PhoBERTEmotionClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels'],
        dropout=config['model']['dropout']
    )

    num_params = model.get_num_parameters()
    print(f"Model created: {config['model']['name']}")
    print(f"Trainable parameters: {num_params:,}")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing PhoBERT model...")

    model = PhoBERTEmotionClassifier(
        model_name="vinai/phobert-base",
        num_labels=7,
        dropout=0.1
    )

    print(f"\nModel architecture:")
    print(model)

    print(f"\nNumber of parameters: {model.get_num_parameters():,}")

    # Test forward pass
    batch_size = 2
    seq_len = 128

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)

    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output (logits): {logits}")
