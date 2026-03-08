"""
PhoBERT model for emotion classification
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


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
        Linear Layer (768 → num_labels)
    """

    def __init__(
        self,
        model_name="vinai/phobert-base",
        num_labels=7,
        dropout=0.1,
        freeze_bert=False
    ):
        """
        Initialize PhoBERT classifier

        Args:
            model_name: Name of pretrained PhoBERT model
            num_labels: Number of emotion classes
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT parameters
        """
        super(PhoBERTEmotionClassifier, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # Load PhoBERT model
        self.bert = AutoModel.from_pretrained(model_name)

        # Get hidden size from config
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_labels)

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
            loss = nn.CrossEntropyLoss()(logits, labels)

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
