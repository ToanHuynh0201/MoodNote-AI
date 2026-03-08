"""
Training utilities using Hugging Face Trainer API
"""
import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
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
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    seed=42,
    log_steps=100,
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
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

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        fp16=fp16 and torch.cuda.is_available(),
        seed=seed,
        logging_dir=str(Path(output_dir) / "logs"),
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
        lr_scheduler_type="cosine"
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


EmotionTrainer = Trainer


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
    training_args = create_training_arguments(
        output_dir=output_dir,
        num_epochs=training_config['training']['num_epochs'],
        batch_size=training_config['training']['batch_size'],
        learning_rate=float(training_config['training']['learning_rate']),
        warmup_steps=training_config['training']['warmup_steps'],
        weight_decay=training_config['training']['weight_decay'],
        fp16=training_config['training'].get('fp16', True),
        seed=training_config['training'].get('seed', 42),
        log_steps=training_config['logging']['log_steps'],
        eval_steps=training_config['logging']['eval_steps'],
        save_steps=training_config['logging']['save_steps'],
        save_total_limit=training_config['logging']['save_total_limit'],
        wandb_config=training_config.get('wandb')
    )

    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
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
