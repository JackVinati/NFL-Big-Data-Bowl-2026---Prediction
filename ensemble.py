"""
Ensemble prediction combining all three models
Optimized for competition submission
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import config
from models import create_model


class EnsemblePredictor:
    """
    Ensemble predictor that combines physics, LSTM, and transformer models
    """

    def __init__(self, model_paths: Dict[str, str] = None, device='cuda'):
        """
        Args:
            model_paths: Dict with keys 'physics', 'lstm', 'transformer'
                        pointing to model checkpoint paths
            device: Device to run inference on
        """
        self.device = device
        self.models = {}
        self.weights = config.ENSEMBLE_CONFIG['weights']

        print("Initializing ensemble predictor...")
        print(f"  Device: {device}")
        print(f"  Weights: {self.weights}")

        # Load models
        if model_paths is None:
            # Default paths
            model_paths = {
                'physics': None,  # Physics model has no trained weights
                'lstm': 'models/lstm/best_model.pth',
                'transformer': 'models/transformer/best_model.pth'
            }

        # Initialize physics model (no loading needed)
        self.models['physics'] = create_model('physics').to(device)
        self.models['physics'].eval()
        print("  ✓ Physics model loaded")

        # Load LSTM
        if model_paths['lstm'] and Path(model_paths['lstm']).exists():
            self.models['lstm'] = create_model('lstm', input_size=26).to(device)
            checkpoint = torch.load(model_paths['lstm'], map_location=device)
            self.models['lstm'].load_state_dict(checkpoint['model_state_dict'])
            self.models['lstm'].eval()
            print(f"  ✓ LSTM model loaded (RMSE: {checkpoint['metrics']['rmse']:.4f})")
        else:
            print(f"  ⚠ LSTM model not found at {model_paths['lstm']}")
            self.weights['lstm'] = 0
            self.weights['transformer'] += self.weights['lstm']

        # Load Transformer
        if model_paths['transformer'] and Path(model_paths['transformer']).exists():
            self.models['transformer'] = create_model('transformer', input_size=26).to(device)
            checkpoint = torch.load(model_paths['transformer'], map_location=device)
            self.models['transformer'].load_state_dict(checkpoint['model_state_dict'])
            self.models['transformer'].eval()
            print(f"  ✓ Transformer model loaded (RMSE: {checkpoint['metrics']['rmse']:.4f})")
        else:
            print(f"  ⚠ Transformer model not found at {model_paths['transformer']}")
            self.weights['transformer'] = 0

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            print(f"  Normalized weights: {self.weights}")

    @torch.no_grad()
    def predict(self, input_seq, num_frames_to_predict):
        """
        Generate ensemble predictions

        Args:
            input_seq: [batch, seq_len, features] tensor
            num_frames_to_predict: int or list of ints

        Returns:
            predictions: [batch, num_frames, 2] tensor (x, y positions)
        """

        predictions = []

        # Get predictions from each model
        for model_name, model in self.models.items():
            if self.weights.get(model_name, 0) > 0:
                pred = model(input_seq, num_frames_to_predict)
                predictions.append((self.weights[model_name], pred))

        # Weighted average
        if len(predictions) > 0:
            ensemble_pred = sum(w * pred for w, pred in predictions)
        else:
            raise ValueError("No models available for prediction")

        return ensemble_pred

    @torch.no_grad()
    def predict_batch(self, batch):
        """
        Predict for a full batch from dataloader

        Args:
            batch: Dict from dataloader

        Returns:
            predictions: Dict with game_id, play_id, nfl_id, and predictions
        """

        input_seq = batch['input'].to(self.device)  # [batch, num_players, seq_len, features]
        game_ids = batch['game_id']
        play_ids = batch['play_id']
        player_ids = batch['player_ids']

        batch_size, num_players, seq_len, num_features = input_seq.shape

        # Reshape for models
        input_seq = input_seq.view(-1, seq_len, num_features)  # [batch*players, seq_len, features]

        # Determine max frames
        if 'num_output_frames' in batch:
            max_frames = int(batch['num_output_frames'].max())
        else:
            max_frames = 11  # Default

        # Get ensemble predictions
        predictions = self.predict(input_seq, max_frames)  # [batch*players, max_frames, 2]

        # Reshape back
        predictions = predictions.view(batch_size, num_players, max_frames, 2)

        return {
            'game_ids': game_ids,
            'play_ids': play_ids,
            'player_ids': player_ids,
            'predictions': predictions.cpu().numpy()
        }


def denormalize_positions(predictions, scaler=None):
    """
    Denormalize predicted positions back to field coordinates

    Args:
        predictions: Array of predictions in normalized space
        scaler: StandardScaler used during preprocessing

    Returns:
        Denormalized predictions
    """

    # Simple denormalization (multiply by field dimensions)
    # This assumes predictions are in 0-1 range
    predictions_denorm = predictions.copy()
    predictions_denorm[..., 0] *= config.FIELD_LENGTH  # x
    predictions_denorm[..., 1] *= config.FIELD_WIDTH   # y

    return predictions_denorm


def create_submission(ensemble_predictor, test_loader, output_file='submission.csv'):
    """
    Create submission file for Kaggle

    Args:
        ensemble_predictor: EnsemblePredictor instance
        test_loader: DataLoader for test data
        output_file: Path to save submission CSV
    """

    print(f"\nGenerating predictions for submission...")

    all_predictions = []

    for batch in tqdm(test_loader, desc="Predicting"):
        # Get ensemble predictions
        results = ensemble_predictor.predict_batch(batch)

        # Process each play in batch
        for b in range(len(results['game_ids'])):
            game_id = results['game_ids'][b]
            play_id = results['play_ids'][b]
            player_ids = results['player_ids'][b]
            predictions = results['predictions'][b]  # [num_players, max_frames, 2]

            # Create rows for submission
            for p, player_id in enumerate(player_ids):
                for frame in range(predictions.shape[1]):
                    x_pred, y_pred = predictions[p, frame]

                    all_predictions.append({
                        'game_id': game_id,
                        'play_id': play_id,
                        'nfl_id': player_id,
                        'frame_id': frame + 1,  # Frames start at 1
                        'x': x_pred,
                        'y': y_pred
                    })

    # Create dataframe
    submission_df = pd.DataFrame(all_predictions)

    # Denormalize if needed (check if values are in 0-1 range)
    if submission_df['x'].max() <= 1.5:
        submission_df[['x', 'y']] = denormalize_positions(
            submission_df[['x', 'y']].values
        )

    # Clip to field boundaries
    submission_df['x'] = submission_df['x'].clip(0, config.FIELD_LENGTH)
    submission_df['y'] = submission_df['y'].clip(0, config.FIELD_WIDTH)

    # Save
    submission_df.to_csv(output_file, index=False)
    print(f"\nSubmission saved to {output_file}")
    print(f"  Total predictions: {len(submission_df):,} rows")
    print(f"  Unique plays: {submission_df.groupby(['game_id', 'play_id']).ngroups:,}")
    print(f"  Unique players: {submission_df['nfl_id'].nunique():,}")

    # Display sample
    print("\nSample predictions:")
    print(submission_df.head(10))

    return submission_df


def evaluate_ensemble(ensemble_predictor, val_loader):
    """
    Evaluate ensemble on validation data

    Args:
        ensemble_predictor: EnsemblePredictor instance
        val_loader: DataLoader for validation data

    Returns:
        rmse: Root mean squared error
    """

    print("\nEvaluating ensemble on validation set...")

    total_squared_error = 0
    total_count = 0

    for batch in tqdm(val_loader, desc="Evaluating"):
        # Get predictions
        results = ensemble_predictor.predict_batch(batch)

        # Get targets
        targets = batch['target'].numpy()  # [batch, num_players, max_frames, 2]
        masks = batch['mask'].numpy()      # [batch, num_players, max_frames]

        predictions = results['predictions']

        # Calculate squared error
        squared_diff = (predictions - targets) ** 2
        squared_diff = squared_diff * masks[..., np.newaxis]  # Apply mask

        total_squared_error += squared_diff.sum()
        total_count += masks.sum() * 2  # *2 for x and y

    rmse = np.sqrt(total_squared_error / total_count)

    print(f"\nEnsemble Validation RMSE: {rmse:.4f}")

    return rmse


if __name__ == '__main__':
    # Test ensemble
    print("Testing ensemble predictor...")

    # Create dummy ensemble
    ensemble = EnsemblePredictor()

    # Test with dummy data
    dummy_input = torch.randn(4, 20, 26).to(config.DEVICE)
    predictions = ensemble.predict(dummy_input, 11)

    print(f"\nTest successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Output range: x=[{predictions[..., 0].min():.2f}, {predictions[..., 0].max():.2f}], "
          f"y=[{predictions[..., 1].min():.2f}, {predictions[..., 1].max():.2f}]")
