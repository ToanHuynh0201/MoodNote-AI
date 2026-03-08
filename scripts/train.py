"""
Main training script for PhoBERT emotion classification
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from src.data.dataset import EmotionDataset
from src.models.phobert_classifier import PhoBERTEmotionClassifier
from src.models.model_utils import save_model, get_device, print_model_summary
from src.training.trainer import train_model
from src.utils.config import load_all_configs
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metrics, print_metrics, plot_confusion_matrix
import numpy as np


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train PhoBERT for emotion classification")

    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing config files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing preprocessed data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--best-model-dir",
        type=str,
        default="models/best_model",
        help="Directory to save best model"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Setup logger
    logger = setup_logger(name="training")

    logger.info("=" * 60)
    logger.info("PhoBERT Emotion Classification Training")
    logger.info("=" * 60)

    # Load configurations
    logger.info("\n1. Loading configurations...")
    configs = load_all_configs(args.config_dir)
    model_config = configs['model']
    training_config = configs['training']

    logger.info(f"Model: {model_config['model']['name']}")
    logger.info(f"Num labels: {model_config['model']['num_labels']}")
    logger.info(f"Learning rate: {training_config['training']['learning_rate']}")
    logger.info(f"Batch size: {training_config['training']['batch_size']}")
    logger.info(f"Num epochs: {training_config['training']['num_epochs']}")

    # Get device
    device = get_device()

    # Load datasets
    logger.info("\n2. Loading datasets...")
    data_path = Path(args.data_dir)

    train_dataset = EmotionDataset(
        data_path=data_path / "train.csv",
        tokenizer_name=model_config['model']['name'],
        max_length=model_config['model']['max_seq_length']
    )

    val_dataset = EmotionDataset(
        data_path=data_path / "validation.csv",
        tokenizer_name=model_config['model']['name'],
        max_length=model_config['model']['max_seq_length']
    )

    test_dataset = EmotionDataset(
        data_path=data_path / "test.csv",
        tokenizer_name=model_config['model']['name'],
        max_length=model_config['model']['max_seq_length']
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create model
    logger.info("\n3. Creating model...")
    model = PhoBERTEmotionClassifier(
        model_name=model_config['model']['name'],
        num_labels=model_config['model']['num_labels'],
        dropout=model_config['model']['dropout'],
        label_smoothing=model_config['model'].get('label_smoothing', 0.0),
        focal_gamma=model_config['model'].get('focal_gamma', 2.0)
    )

    model.to(device)
    print_model_summary(model)

    # Train model
    logger.info("\n4. Training model...")
    use_wandb = not args.no_wandb and training_config.get('wandb', {}).get('enabled', True)

    trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_config=training_config,
        output_dir=args.output_dir,
        use_wandb=use_wandb
    )

    # Evaluate on test set
    logger.info("\n5. Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    logger.info("Test set results:")
    for key, value in test_results.items():
        logger.info(f"  {key}: {value:.4f}")

    # Get predictions for detailed metrics
    logger.info("\n6. Computing detailed metrics...")
    predictions = trainer.predict(test_dataset)

    test_preds = predictions.predictions
    test_labels = predictions.label_ids

    # Compute detailed metrics
    detailed_metrics = compute_metrics(test_preds, test_labels)
    print_metrics(detailed_metrics, model_config['emotion_labels'])

    # Plot confusion matrix
    logger.info("\n7. Generating confusion matrix...")
    cm_path = Path(args.output_dir) / "confusion_matrix.png"
    plot_confusion_matrix(
        test_preds,
        test_labels,
        emotion_labels=model_config['emotion_labels'],
        save_path=cm_path
    )

    # Save best model
    logger.info("\n8. Saving best model...")
    save_model(
        model=trainer.model,
        tokenizer=train_dataset.tokenizer,
        save_dir=args.best_model_dir,
        config={
            'model_config': model_config,
            'training_config': training_config,
            'test_results': {
                'accuracy': detailed_metrics['accuracy'],
                'f1_macro': detailed_metrics['f1_macro'],
                'f1_weighted': detailed_metrics['f1_weighted']
            }
        }
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"\nBest model saved to: {args.best_model_dir}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info(f"Confusion matrix saved to: {cm_path}")

    logger.info(f"\nFinal Test Results:")
    logger.info(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Macro: {detailed_metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Weighted: {detailed_metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
