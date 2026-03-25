"""
Training utilities using Hugging Face Trainer API
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any
from torch.optim import AdamW
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup
)
from pathlib import Path
import wandb
from ..utils.metrics import compute_metrics_for_trainer
from ..utils.logger import get_logger

logger = get_logger()


def create_training_arguments(
    output_dir,
    num_epochs=5,
    batch_size=16,
    learning_rate=3e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,
    seed=42,
    log_steps=100,
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    gradient_accumulation_steps=1,
    wandb_config=None
):
    """
    Create TrainingArguments for Hugging Face Trainer

    Args:
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for optimizer
        fp16: Whether to use mixed precision training
        seed: Random seed
        log_steps: Log every N steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        wandb_config: Weights & Biases configuration

    Returns:
        TrainingArguments: Training arguments object
    """
    # Determine report_to
    report_to = []
    if wandb_config and wandb_config.get('enabled', True):
        report_to.append('wandb')

    import os
    os.environ['TENSORBOARD_LOGGING_DIR'] = str(Path(output_dir) / "logs")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16 and torch.cuda.is_available(),
        seed=seed,
        warmup_steps=warmup_steps,
        logging_steps=log_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=report_to,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
    )

    return args


def init_wandb(config, project_name, run_name=None):
    """
    Initialize Weights & Biases

    Args:
        config: Training configuration
        project_name: W&B project name
        run_name: W&B run name (optional)
    """
    if run_name is None:
        from datetime import datetime
        run_name = f"phobert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project=project_name,
        name=run_name,
        config=config
    )

    logger.info(f"W&B initialized: {project_name}/{run_name}")


class EmotionTrainer(Trainer):
    """
    Custom Trainer with Layer-wise LR Decay (LLRD) support.
    When llrd_factor is set and model has get_parameter_groups(),
    each BERT layer gets a progressively smaller learning rate.
    """

    def __init__(self, *args, llrd_factor=None, rdrop_alpha=0.0, **kwargs):
        self.llrd_factor = llrd_factor
        self.rdrop_alpha = rdrop_alpha
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.rdrop_alpha > 0 and model.training:
            # R-Drop: two forward passes with different dropout masks + KL divergence
            # Ref: "R-Drop: Regularized Dropout for Neural Networks" (NeurIPS 2021)
            outputs1 = model(**inputs)
            outputs2 = model(**inputs)
            loss = (outputs1.loss + outputs2.loss) / 2
            p1 = F.softmax(outputs1.logits, dim=-1)
            p2 = F.softmax(outputs2.logits, dim=-1)
            kl = (F.kl_div(F.log_softmax(outputs1.logits, dim=-1), p2, reduction='batchmean') +
                  F.kl_div(F.log_softmax(outputs2.logits, dim=-1), p1, reduction='batchmean')) / 2
            loss = loss + self.rdrop_alpha * kl
            return (loss, outputs1) if return_outputs else loss
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def create_optimizer(self, model=None):  # noqa: ARG002
        if self.llrd_factor and hasattr(self.model, 'get_parameter_groups'):
            assert self.model is not None
            _model: Any = self.model
            param_groups = _model.get_parameter_groups(
                base_lr=self.args.learning_rate,
                llrd_factor=self.llrd_factor
            )
            # Build param_id → name map for correct no-decay detection
            no_decay_names = ["bias", "LayerNorm.weight", "layer_norm.weight"]
            param_to_name = {id(p): n for n, p in self.model.named_parameters()}

            optimizer_grouped_params = []
            for group in param_groups:
                wd_group = {"params": [], "lr": group["lr"], "weight_decay": self.args.weight_decay}
                no_wd_group = {"params": [], "lr": group["lr"], "weight_decay": 0.0}
                for param in group["params"]:
                    name = param_to_name.get(id(param), "")
                    if any(nd in name for nd in no_decay_names):
                        no_wd_group["params"].append(param)
                    else:
                        wd_group["params"].append(param)
                if wd_group["params"]:
                    optimizer_grouped_params.append(wd_group)
                if no_wd_group["params"]:
                    optimizer_grouped_params.append(no_wd_group)

            self.optimizer = AdamW(optimizer_grouped_params, eps=1e-8)
            logger.info(f"LLRD optimizer created with {len(param_groups)} layer groups (factor={self.llrd_factor})")
            return self.optimizer
        return super().create_optimizer()


def train_model(
    model,
    train_dataset,
    eval_dataset,
    training_config,
    output_dir="models/checkpoints",
    use_wandb=True
):
    """
    Train PhoBERT model

    Args:
        model: PhoBERT model
        train_dataset: Training dataset
        eval_dataset: Validation dataset
        training_config: Training configuration dictionary
        output_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases

    Returns:
        Trainer: Trained model trainer
    """
    logger.info("Starting model training...")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize W&B if enabled
    if use_wandb and training_config.get('wandb', {}).get('enabled', True):
        init_wandb(
            config=training_config,
            project_name=training_config['wandb']['project'],
            run_name=training_config['wandb'].get('name')
        )

    # Create training arguments
    t_cfg = training_config['training']
    grad_accum = t_cfg.get('gradient_accumulation_steps', 1)
    steps_per_epoch = len(train_dataset) // (t_cfg['batch_size'] * grad_accum)
    total_steps = steps_per_epoch * t_cfg['num_epochs']
    warmup_ratio = t_cfg.get('warmup_ratio', 0.0)
    warmup_steps = int(total_steps * warmup_ratio) if warmup_ratio > 0 else t_cfg.get('warmup_steps', 100)

    training_args = create_training_arguments(
        output_dir=output_dir,
        num_epochs=t_cfg['num_epochs'],
        batch_size=t_cfg['batch_size'],
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(t_cfg['learning_rate']),
        warmup_steps=warmup_steps,
        weight_decay=t_cfg['weight_decay'],
        fp16=t_cfg.get('fp16', True),
        seed=t_cfg.get('seed', 42),
        log_steps=training_config['logging']['log_steps'],
        eval_steps=training_config['logging']['eval_steps'],
        save_steps=training_config['logging']['save_steps'],
        save_total_limit=training_config['logging']['save_total_limit'],
        wandb_config=training_config.get('wandb')
    )

    # LLRD config
    llrd_factor = None
    if t_cfg.get('use_llrd', False):
        llrd_factor = t_cfg.get('llrd_factor', 0.9)
        logger.info(f"Layer-wise LR Decay enabled (factor={llrd_factor})")

    # R-Drop config (default disabled, enable in Stage 2 by setting rdrop_alpha > 0)
    rdrop_alpha = t_cfg.get('rdrop_alpha', 0.0)
    if rdrop_alpha > 0:
        logger.info(f"R-Drop enabled (alpha={rdrop_alpha})")

    # Early stopping patience from config
    patience = t_cfg.get('early_stopping_patience', 5)

    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        llrd_factor=llrd_factor,
        rdrop_alpha=rdrop_alpha
    )

    # Train
    logger.info("Training started...")
    train_result = trainer.train()

    # Log training results
    logger.info(f"Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()

    logger.info(f"Validation results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")

    return trainer


if __name__ == "__main__":
    # Test trainer setup
    print("Testing trainer setup...")

    config = {
        'training': {
            'num_epochs': 3,
            'batch_size': 16,
            'learning_rate': 3e-5,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'fp16': False,
            'seed': 42
        },
        'logging': {
            'log_steps': 100,
            'eval_steps': 500,
            'save_steps': 500,
            'save_total_limit': 3
        },
        'wandb': {
            'project': 'moodnote-ai',
            'name': 'test-run',
            'enabled': False
        }
    }

    args = create_training_arguments(
        output_dir="test_output",
        **config['training'],
        **config['logging'],
        wandb_config=config['wandb']
    )

    print("\nTraining Arguments:")
    print(f"Output dir: {args.output_dir}")
    print(f"Num epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
