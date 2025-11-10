"""
Main script to run the entire ensemble pipeline
Usage:
    python main.py --mode train    # Train all models
    python main.py --mode predict  # Generate predictions
    python main.py --mode all      # Train and predict
"""

import argparse
import torch
from pathlib import Path

import config
from train import train_all_models
from ensemble import EnsemblePredictor, evaluate_ensemble, create_submission
from data_preprocessing import load_and_preprocess_data, NFLDataset
from torch.utils.data import DataLoader


def train_pipeline():
    """Train all three models"""
    print("\n" + "="*80)
    print("TRAINING PIPELINE")
    print("="*80)

    models, trainers = train_all_models()

    print("\n✓ All models trained successfully!")
    print(f"  Models saved in: {config.TRAIN_CONFIG['save_dir']}")

    return models, trainers


def predict_pipeline(use_test_data=False):
    """Generate predictions using ensemble"""
    print("\n" + "="*80)
    print("PREDICTION PIPELINE")
    print("="*80)

    # Load validation data for evaluation
    print("\nLoading validation data...")
    _, val_dataset, feat_eng = load_and_preprocess_data()

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # Create ensemble
    print("\nCreating ensemble predictor...")
    ensemble = EnsemblePredictor(device=config.DEVICE)

    # Evaluate on validation set
    val_rmse = evaluate_ensemble(ensemble, val_loader)

    print(f"\n✓ Ensemble evaluation complete!")
    print(f"  Validation RMSE: {val_rmse:.4f}")

    # Generate submission (if test data available)
    if use_test_data:
        print("\nGenerating test predictions...")
        # TODO: Load test data and create submission
        # This would require the actual test data from Kaggle
        print("  ⚠ Test data loading not implemented yet")
        print("  → Use the Kaggle API submission format once available")

    return ensemble, val_rmse


def main():
    parser = argparse.ArgumentParser(description='NFL Big Data Bowl 2026 - Ensemble Pipeline')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'predict', 'all'],
                       help='Pipeline mode: train, predict, or all')
    parser.add_argument('--test', action='store_true',
                       help='Generate predictions for test data')

    args = parser.parse_args()

    print("="*80)
    print("NFL BIG DATA BOWL 2026 - ENSEMBLE TRAJECTORY PREDICTION")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Mode: {args.mode}")
    print("="*80)

    if args.mode in ['train', 'all']:
        # Train models
        models, trainers = train_pipeline()

    if args.mode in ['predict', 'all']:
        # Generate predictions
        ensemble, val_rmse = predict_pipeline(use_test_data=args.test)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

    if args.mode == 'all':
        print(f"\n✓ All models trained and evaluated")
        print(f"  → Check {config.TRAIN_CONFIG['save_dir']} for model checkpoints")
        print(f"  → Check {config.TRAIN_CONFIG['log_dir']} for training logs")


if __name__ == '__main__':
    main()
