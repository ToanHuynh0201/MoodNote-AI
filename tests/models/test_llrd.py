"""Tests cho Layer-wise LR Decay — dựng encoder ngẫu nhiên tí hon, không tải PhoBERT thật."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.models.phobert_classifier import PhoBERTEmotionClassifier  # noqa: E402

NUM_LAYERS = 3
BASE_LR = 2e-5
LLRD_FACTOR = 0.85


@pytest.fixture
def tiny_model(monkeypatch):
    """PhoBERTEmotionClassifier gắn encoder RoBERTa random-init 3 lớp (không dùng mạng)."""
    config = transformers.RobertaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )

    monkeypatch.setattr(
        "src.models.phobert_classifier.AutoModel.from_pretrained",
        lambda *args, **kwargs: transformers.RobertaModel(config, add_pooling_layer=False),
    )
    monkeypatch.setattr(
        "src.models.phobert_classifier.AutoConfig.from_pretrained",
        lambda *args, **kwargs: config,
    )

    return PhoBERTEmotionClassifier(model_name="fake/tiny", num_labels=7, dropout=0.1)


def test_get_parameter_groups_returns_one_group_per_layer_plus_head_and_embeddings(tiny_model):
    groups = tiny_model.get_parameter_groups(base_lr=BASE_LR, llrd_factor=LLRD_FACTOR)

    assert [g["name"] for g in groups] == [
        "classifier",
        "bert_layer_2",
        "bert_layer_1",
        "bert_layer_0",
        "bert_embeddings",
    ]


def test_get_parameter_groups_decays_learning_rate_monotonically_downwards(tiny_model):
    lrs = [g["lr"] for g in tiny_model.get_parameter_groups(BASE_LR, LLRD_FACTOR)]

    assert lrs[0] == BASE_LR
    assert lrs == sorted(lrs, reverse=True)
    assert lrs[-1] == pytest.approx(BASE_LR * LLRD_FACTOR**NUM_LAYERS)


def test_forward_returns_logits_for_every_label(tiny_model):
    input_ids = torch.randint(0, 64, (2, 16))
    attention_mask = torch.ones_like(input_ids)

    output = tiny_model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (2, 7)
    assert output.loss is None


def test_forward_computes_loss_when_labels_are_given(tiny_model):
    input_ids = torch.randint(0, 64, (2, 16))
    attention_mask = torch.ones_like(input_ids)

    output = tiny_model(input_ids, attention_mask, labels=torch.tensor([0, 3]))

    assert torch.isfinite(output.loss)
