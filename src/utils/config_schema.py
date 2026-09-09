"""
Pydantic schemas validating the shape of configs/*.yaml.
"""

from __future__ import annotations

from pydantic import BaseModel

from .config import load_all_configs


class ModelSection(BaseModel):
    name: str
    num_labels: int
    max_seq_length: int
    dropout: float
    label_smoothing: float
    focal_gamma: float


class PreprocessingSection(BaseModel):
    segmenter: str
    lowercase: bool


class ModelConfigSchema(BaseModel):
    model: ModelSection
    emotion_labels: dict[int, str]
    sentiment_scores: dict[str, float]
    preprocessing: PreprocessingSection


class TrainingSection(BaseModel):
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int
    warmup_ratio: float
    weight_decay: float
    fp16: bool
    seed: int
    early_stopping_patience: int
    use_llrd: bool
    llrd_factor: float
    use_class_weights: bool
    rdrop_alpha: float


class OptimizerSection(BaseModel):
    type: str
    betas: tuple[float, float]
    eps: float


class SchedulerSection(BaseModel):
    type: str


class LoggingSection(BaseModel):
    log_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int


class WandbSection(BaseModel):
    project: str
    name: str
    enabled: bool


class AblationSection(BaseModel):
    scenarios: list[str]
    baseline: str
    metrics: list[str]
    real_dir: str
    synthetic_dir: str
    ablation_dir: str
    validation_path: str
    test_path: str
    results_dir: str


class TrainingConfigSchema(BaseModel):
    training: TrainingSection
    optimizer: OptimizerSection
    scheduler: SchedulerSection
    logging: LoggingSection
    wandb: WandbSection
    ablation: AblationSection


class ApiSection(BaseModel):
    host: str
    port: int
    reload: bool
    workers: int


class ApiModelSection(BaseModel):
    path: str
    device: str
    max_batch_size: int


class ApiPreprocessingSection(BaseModel):
    segmenter: str
    max_length: int


class ApiConfigSchema(BaseModel):
    api: ApiSection
    model: ApiModelSection
    preprocessing: ApiPreprocessingSection


def validate_configs(config_dir: str = "configs") -> dict[str, BaseModel]:
    """
    Load configs/*.yaml and validate each against its pydantic schema.

    Args:
        config_dir: Directory containing model/training/api config files

    Returns:
        dict: Validated schema instances keyed by "model"/"training"/"api"

    Raises:
        pydantic.ValidationError: if a config file doesn't match its schema
    """
    raw = load_all_configs(config_dir)

    return {
        "model": ModelConfigSchema(**raw["model"]),
        "training": TrainingConfigSchema(**raw["training"]),
        "api": ApiConfigSchema(**raw["api"]),
    }
