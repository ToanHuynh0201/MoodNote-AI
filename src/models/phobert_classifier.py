"""
PhoBERT model for emotion classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers import logging as hf_logging
from transformers.modeling_outputs import SequenceClassifierOutput


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced classification, with optional label smoothing.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=0 → standard CrossEntropyLoss
    gamma=2 → focuses training on hard/misclassified examples
    label_smoothing > 0 → soft targets to prevent overconfidence
    """

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)

        if self.label_smoothing > 0:
            # Build soft targets: (1 - eps) for true class, eps/(C-1) for others
            with torch.no_grad():
                soft_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(logits, dim=-1)
            # CE with soft targets
            ce_loss = -(soft_targets * log_probs).sum(dim=-1)
            # Apply class weights per sample (fix: was silently ignored before)
            if self.weight is not None:
                sample_weights = self.weight.to(logits.device)[targets]
                ce_loss = ce_loss * sample_weights
            # Focal weight based on true class probability
            pt = torch.exp(log_probs).gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
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
        Dropout
            ↓
        Linear (hidden_size → hidden_size // 2)
            ↓
        GELU + LayerNorm + Dropout
            ↓
        Linear (hidden_size // 2 → num_labels)
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

        # Load PhoBERT base model (without CLS pooler — we use mean pooling in forward)
        _prev_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hf_logging.set_verbosity(_prev_level)

        # Get hidden size from config (768 for base, 1024 for large)
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size

        # Input dropout
        self.dropout = nn.Dropout(dropout)

        # Multi-layer classification head: hidden → hidden//2 → num_labels
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_labels)
        )

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
        # Mean pooling over non-padding tokens (better than CLS-only for short social media text)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # Multi-sample Dropout: average K predictions during training for better regularization
        if self.training:
            K = 5
            logits = torch.stack([self.classifier(self.dropout(mean_pooled)) for _ in range(K)]).mean(0)
        else:
            logits = self.classifier(self.dropout(mean_pooled))

        loss = None
        if labels is not None:
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            if self.focal_gamma > 0:
                loss_fn = FocalLoss(gamma=self.focal_gamma, weight=weight, label_smoothing=self.label_smoothing)
                loss = loss_fn(logits, labels)
            else:
                loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
                loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)  # type: ignore[arg-type]

    def get_num_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_groups(self, base_lr, llrd_factor=0.9):
        """
        Get optimizer parameter groups with Layer-wise LR Decay (LLRD).

        Lower BERT layers get smaller LR to preserve pretrained features.
        Classifier head gets full base_lr.

        Args:
            base_lr: Base learning rate (for classifier head)
            llrd_factor: Decay factor per layer (0.9 → each layer gets 90% of layer above)

        Returns:
            list: Parameter groups with different learning rates
        """
        # Classifier head: full base_lr
        param_groups = [
            {"params": self.classifier.parameters(), "lr": base_lr, "name": "classifier"}
        ]

        # BERT encoder layers: top layers get higher LR, bottom layers get lower LR
        num_layers = self.bert.config.num_hidden_layers
        for layer_idx in range(num_layers - 1, -1, -1):
            # Distance from top: layer 11 (top) gets factor^0, layer 0 gets factor^11
            distance_from_top = (num_layers - 1) - layer_idx
            layer_lr = base_lr * (llrd_factor ** distance_from_top)
            param_groups.append({
                "params": self.bert.encoder.layer[layer_idx].parameters(),
                "lr": layer_lr,
                "name": f"bert_layer_{layer_idx}"
            })

        # Embeddings: very small LR (preserve pretrained embeddings)
        param_groups.append({
            "params": self.bert.embeddings.parameters(),
            "lr": base_lr * (llrd_factor ** num_layers),
            "name": "bert_embeddings"
        })

        return param_groups


def create_model(config, class_weights=None):
    """
    Create PhoBERT model from config

    Args:
        config: Model configuration dictionary
        class_weights: Optional tensor of class weights for imbalanced dataset

    Returns:
        PhoBERTEmotionClassifier: Initialized model
    """
    model = PhoBERTEmotionClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels'],
        dropout=config['model']['dropout'],
        label_smoothing=config['model'].get('label_smoothing', 0.0),
        focal_gamma=config['model'].get('focal_gamma', 2.0),
        class_weights=class_weights
    )

    num_params = model.get_num_parameters()
    print(f"Model created: {config['model']['name']}")
    print(f"Trainable parameters: {num_params:,}")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing PhoBERT model...")

    model = PhoBERTEmotionClassifier(
        model_name="uitnlp/visobert",
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
    print(f"Output shape: {logits.logits.shape}")
    print(f"Output (logits): {logits.logits}")
