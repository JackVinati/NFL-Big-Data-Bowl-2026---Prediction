"""
Training script for ensemble models
Can train all 3 models in parallel on RTX 5090
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import json

import config
from models import create_model
from data_preprocessing import load_and_preprocess_data, custom_collate_fn


# ============================================================================
# CUSTOM LOSS FUNCTIONS
# ============================================================================

class TrajectoryLoss(nn.Module):
    """
    Custom loss function based on competition metric + physics constraints
    """

    def __init__(self):
        super(TrajectoryLoss, self).__init__()
        self.rmse_weight = config.LOSS_CONFIG['rmse_weight']
        self.smoothness_weight = config.LOSS_CONFIG['smoothness_weight']
        self.physics_weight = config.LOSS_CONFIG['physics_weight']

    def forward(self, predictions, targets, mask=None):
        """
        Args:
            predictions: [batch, num_frames, 2]
            targets: [batch, num_frames, 2]
            mask: [batch, num_frames] - 1 for valid frames, 0 for padding

        Returns:
            total_loss, rmse_loss, smoothness_loss, physics_loss
        """

        # RMSE loss (competition metric)
        squared_diff = (predictions - targets) ** 2

        if mask is not None:
            mask = mask.unsqueeze(-1)  # [batch, num_frames, 1]
            squared_diff = squared_diff * mask
            num_valid = mask.sum()
        else:
            num_valid = predictions.numel() / 2  # Divide by 2 for x,y

        rmse_loss = torch.sqrt(squared_diff.sum() / (2 * num_valid))

        # Smoothness loss (frame-to-frame consistency)
        if predictions.shape[1] > 1:
            frame_diff = predictions[:, 1:, :] - predictions[:, :-1, :]
            frame_diff_squared = (frame_diff ** 2).sum(-1)  # [batch, num_frames-1]

            if mask is not None:
                smooth_mask = mask[:, :-1, 0]  # [batch, num_frames-1]
                frame_diff_squared = frame_diff_squared * smooth_mask
                smoothness_loss = frame_diff_squared.sum() / smooth_mask.sum()
            else:
                smoothness_loss = frame_diff_squared.mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)

        # Physics loss (velocity constraints)
        if predictions.shape[1] > 1:
            velocities = (predictions[:, 1:, :] - predictions[:, :-1, :]) * config.FRAMES_PER_SECOND
            speeds = torch.sqrt((velocities ** 2).sum(-1))  # [batch, num_frames-1]

            # Penalize speeds exceeding max
            max_speed_violation = torch.relu(speeds - config.LOSS_CONFIG['max_speed'])

            if mask is not None:
                physics_mask = mask[:, :-1, 0]
                physics_loss = (max_speed_violation * physics_mask).sum() / physics_mask.sum()
            else:
                physics_loss = max_speed_violation.mean()
        else:
            physics_loss = torch.tensor(0.0, device=predictions.device)

        # Total loss
        total_loss = (self.rmse_weight * rmse_loss +
                     self.smoothness_weight * smoothness_loss +
                     self.physics_weight * physics_loss)

        return total_loss, rmse_loss, smoothness_loss, physics_loss


# ============================================================================
# TRAINER CLASS
# ============================================================================

class Trainer:
    """
    Trainer for individual model
    """

    def __init__(self, model, model_name, train_loader, val_loader,
                 device='cuda', use_amp=True):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp

        # Loss function
        self.criterion = TrajectoryLoss()

        # Optimizer (different for different models)
        if 'physics' in model_name:
            # Physics model has no learnable parameters
            self.optimizer = None
            self.scheduler = None
            self.scaler = None
        elif 'lstm' in model_name:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LSTM_CONFIG['learning_rate'],
                weight_decay=config.LSTM_CONFIG['weight_decay']
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.TRAIN_CONFIG['reduce_lr_patience'],
                factor=config.TRAIN_CONFIG['reduce_lr_factor'],
                min_lr=config.TRAIN_CONFIG['min_lr']
            )
            self.scaler = GradScaler() if use_amp else None
        else:  # transformer
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.TRANSFORMER_CONFIG['learning_rate'],
                weight_decay=config.TRANSFORMER_CONFIG['weight_decay']
            )
            # Warmup scheduler
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRANSFORMER_CONFIG['learning_rate'],
                epochs=config.TRANSFORMER_CONFIG['epochs'],
                steps_per_epoch=len(train_loader),
                pct_start=0.1
            )
            self.scaler = GradScaler() if use_amp else None

        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        # Create save directory
        self.save_dir = Path(config.TRAIN_CONFIG['save_dir']) / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Trainer initialized for {model_name}")
        print(f"  Device: {device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def train_epoch(self):
        """Train for one epoch"""

        if 'physics' in self.model_name:
            # Physics model doesn't need training
            return 0.0, {}

        self.model.train()
        epoch_loss = 0.0
        epoch_rmse = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Training {self.model_name}")

        for batch in pbar:
            input_seq = batch['input'].to(self.device)  # [batch, num_players, seq_len, features]
            target = batch['target'].to(self.device)    # [batch, num_players, max_frames, 2]
            mask = batch['mask'].to(self.device)        # [batch, num_players, max_frames]

            # Reshape for model
            batch_size, num_players, seq_len, num_features = input_seq.shape
            input_seq = input_seq.view(-1, seq_len, num_features)  # [batch*players, seq_len, features]
            target = target.view(-1, target.shape[2], 2)            # [batch*players, max_frames, 2]
            mask = mask.view(-1, mask.shape[2])                     # [batch*players, max_frames]

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    predictions = self.model(input_seq, num_frames_to_predict=target.shape[1])
                    loss, rmse, smooth, phys = self.criterion(predictions, target, mask)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                              config.LSTM_CONFIG.get('gradient_clip', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(input_seq, num_frames_to_predict=target.shape[1])
                loss, rmse, smooth, phys = self.criterion(predictions, target, mask)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                              config.LSTM_CONFIG.get('gradient_clip', 1.0))
                self.optimizer.step()

            # Update scheduler if OneCycleLR
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            epoch_loss += loss.item()
            epoch_rmse += rmse.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'rmse': f'{rmse.item():.4f}'})

        avg_loss = epoch_loss / num_batches
        avg_rmse = epoch_rmse / num_batches

        return avg_loss, {'rmse': avg_rmse}

    @torch.no_grad()
    def validate(self):
        """Validate the model"""

        self.model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc=f"Validating {self.model_name}"):
            input_seq = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Reshape
            batch_size, num_players, seq_len, num_features = input_seq.shape
            input_seq = input_seq.view(-1, seq_len, num_features)
            target = target.view(-1, target.shape[2], 2)
            mask = mask.view(-1, mask.shape[2])

            # Forward pass
            if self.use_amp:
                with autocast():
                    predictions = self.model(input_seq, num_frames_to_predict=target.shape[1])
                    loss, rmse, _, _ = self.criterion(predictions, target, mask)
            else:
                predictions = self.model(input_seq, num_frames_to_predict=target.shape[1])
                loss, rmse, _, _ = self.criterion(predictions, target, mask)

            val_loss += loss.item()
            val_rmse += rmse.item()
            num_batches += 1

        avg_loss = val_loss / num_batches
        avg_rmse = val_rmse / num_batches

        return avg_loss, {'rmse': avg_rmse}

    def train(self, num_epochs):
        """Full training loop"""

        print(f"\nStarting training for {self.model_name}")
        print("=" * 80)

        if 'physics' in self.model_name:
            print("Physics model doesn't require training. Evaluating...")
            val_loss, metrics = self.validate()
            print(f"Validation RMSE: {metrics['rmse']:.4f}")
            self.save_checkpoint(0, val_loss, metrics)
            return

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)

            # Update scheduler if ReduceLROnPlateau
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            # Print metrics
            print(f"  Train Loss: {train_loss:.4f}, Train RMSE: {train_metrics['rmse']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_metrics, is_best=True)
                print(f"  ✓ New best model saved (RMSE: {val_metrics['rmse']:.4f})")
            else:
                self.patience_counter += 1

            # Regular checkpoint
            if (epoch + 1) % config.TRAIN_CONFIG['checkpoint_freq'] == 0:
                self.save_checkpoint(epoch, val_loss, val_metrics)

            # Early stopping
            if self.patience_counter >= config.TRAIN_CONFIG['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.1f} minutes")
        print(f"Best validation RMSE: {self.best_val_loss:.4f}")

    def save_checkpoint(self, epoch, val_loss, metrics, is_best=False):
        """Save model checkpoint"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        if is_best:
            save_path = self.save_dir / 'best_model.pth'
        else:
            save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, save_path)

        # Save metrics
        metrics_dict = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_rmse': metrics.get('rmse', 0),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_all_models():
    """
    Train all three models
    Can be run in parallel on different GPUs or sequentially
    """

    print("="*80)
    print("NFL BIG DATA BOWL 2026 - ENSEMBLE TRAINING")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Mixed Precision: {config.USE_AMP}")
    print()

    # Load data
    print("Loading and preprocessing data...")
    train_dataset, val_dataset, feat_eng = load_and_preprocess_data()

    # Create data loaders for each model (with custom collate function)
    physics_train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn
    )

    physics_val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn
    )

    lstm_train_loader = DataLoader(
        train_dataset,
        batch_size=config.LSTM_CONFIG['batch_size'],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn
    )

    lstm_val_loader = DataLoader(
        val_dataset,
        batch_size=config.LSTM_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn
    )

    transformer_train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRANSFORMER_CONFIG['batch_size'],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn
    )

    transformer_val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRANSFORMER_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn
    )

    # Create models
    print("\nInitializing models...")
    models = {
        'physics': create_model('physics'),
        'lstm': create_model('lstm', input_size=26),
        'transformer': create_model('transformer', input_size=26)
    }

    # Create trainers
    trainers = {
        'physics': Trainer(models['physics'], 'physics', physics_train_loader, physics_val_loader),
        'lstm': Trainer(models['lstm'], 'lstm', lstm_train_loader, lstm_val_loader),
        'transformer': Trainer(models['transformer'], 'transformer',
                              transformer_train_loader, transformer_val_loader)
    }

    # Train models sequentially (can be parallelized with multiprocessing)
    print("\n" + "="*80)
    print("TRAINING PHYSICS MODEL")
    print("="*80)
    trainers['physics'].train(1)  # Just evaluate

    print("\n" + "="*80)
    print("TRAINING LSTM MODEL")
    print("="*80)
    trainers['lstm'].train(config.LSTM_CONFIG['epochs'])

    print("\n" + "="*80)
    print("TRAINING TRANSFORMER MODEL")
    print("="*80)
    trainers['transformer'].train(config.TRANSFORMER_CONFIG['epochs'])

    print("\n" + "="*80)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)

    return models, trainers


if __name__ == '__main__':
    train_all_models()
