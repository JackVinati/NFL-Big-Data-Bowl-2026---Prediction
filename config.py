"""
Configuration file for NFL Big Data Bowl 2026 - Ensemble Models
Optimized for RTX 5090 with 32GB VRAM
"""

import torch

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 8
PIN_MEMORY = True
USE_AMP = True  # Automatic Mixed Precision (FP16)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_DIR = 'train/'
TEST_INPUT = 'test_input.csv'
TEST_OUTPUT = 'test.csv'

# Field dimensions
FIELD_LENGTH = 120.0  # yards
FIELD_WIDTH = 53.3    # yards

# Temporal
FRAMES_PER_SECOND = 10

# ============================================================================
# FEATURE CONFIGURATION (Based on EDA insights)
# ============================================================================
INPUT_FEATURES = [
    'x', 'y',           # Position
    's', 'a',           # Speed, acceleration
    'dir', 'o',         # Direction, orientation
]

# Feature engineering based on EDA correlations
ENGINEERED_FEATURES = [
    'dx_to_ball',       # Distance x to ball landing
    'dy_to_ball',       # Distance y to ball landing
    'dist_to_ball',     # Euclidean distance to ball
    'velocity_x',       # Speed component in x
    'velocity_y',       # Speed component in y
    'angle_to_ball',    # Angle toward ball
    'speed_to_ball',    # Speed component toward ball
]

# Player role embeddings
PLAYER_ROLES = [
    'Defensive Coverage',
    'Other Route Runner',
    'Targeted Receiver',
    'Passer'
]

PLAYER_POSITIONS = [
    'WR', 'CB', 'FS', 'TE', 'QB', 'SS', 'RB',
    'ILB', 'OLB', 'MLB', 'FB', 'DE', 'S', 'DT', 'NT'
]

# ============================================================================
# MODEL 1: PHYSICS-BASED MODEL
# ============================================================================
PHYSICS_CONFIG = {
    'name': 'physics_baseline',
    'ball_attraction_weight': 0.3,  # Based on EDA: targeted receivers move toward ball
    'velocity_weight': 0.7,         # Maintain current velocity
    'smoothing_factor': 0.1,        # Frame-to-frame smoothing
    'role_speed_multipliers': {     # From EDA speed analysis
        'Targeted Receiver': 1.0,
        'Other Route Runner': 1.0,
        'Defensive Coverage': 0.64,  # 2.54 / 3.99
        'Passer': 0.45,              # 1.80 / 3.99
    }
}

# ============================================================================
# MODEL 2: LSTM MODEL
# ============================================================================
LSTM_CONFIG = {
    'name': 'lstm_attention',
    'input_size': 32,               # After embedding
    'hidden_size': 256,
    'num_layers': 3,
    'dropout': 0.2,
    'bidirectional': True,
    'attention_heads': 8,
    'batch_size': 512,              # Large batch for 32GB VRAM
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 50,
    'gradient_clip': 1.0,
    'sequence_length': 20,          # Last 20 frames of input
}

# ============================================================================
# MODEL 3: TRANSFORMER MODEL
# ============================================================================
TRANSFORMER_CONFIG = {
    'name': 'transformer_trajectory',
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'activation': 'gelu',
    'batch_size': 256,              # Transformers use more memory
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'epochs': 50,
    'warmup_steps': 1000,
    'gradient_clip': 1.0,
    'max_sequence_length': 30,      # Max input frames
    'use_social_attention': True,   # Cross-player attention
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAIN_CONFIG = {
    'train_split': 0.85,
    'val_split': 0.15,
    'random_seed': 42,
    'save_dir': 'models/',
    'log_dir': 'logs/',
    'checkpoint_freq': 5,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-6,
}

# Data augmentation (based on EDA insights)
AUGMENTATION_CONFIG = {
    'horizontal_flip_prob': 0.5,    # Play direction left/right is balanced
    'position_noise_std': 0.1,      # Small Gaussian noise
    'velocity_noise_std': 0.05,
    'temporal_shift_max': 2,        # Shift sequence by ±2 frames
}

# ============================================================================
# ENSEMBLE CONFIGURATION
# ============================================================================
ENSEMBLE_CONFIG = {
    'weights': {
        'physics': 0.2,             # Simple baseline
        'lstm': 0.3,                # Good at sequences
        'transformer': 0.5,         # Best for complex interactions
    },
    'use_weighted_average': True,
    'use_stacking': False,          # Can enable later if needed
}

# ============================================================================
# LOSS CONFIGURATION
# ============================================================================
LOSS_CONFIG = {
    'rmse_weight': 1.0,             # Competition metric
    'smoothness_weight': 0.1,       # Frame-to-frame smoothness
    'physics_weight': 0.05,         # Physics constraints (max speed/accel)
    'max_speed': 12.53,             # From EDA
    'max_accel': 17.12,             # From EDA
}

# ============================================================================
# SUBMISSION CONFIGURATION
# ============================================================================
SUBMISSION_CONFIG = {
    'output_file': 'submission.csv',
    'batch_size': 512,
}

# ============================================================================
# CONSTANTS FROM EDA
# ============================================================================
EDA_CONSTANTS = {
    'avg_displacement_per_frame': 0.46,
    'avg_total_displacement': 5.84,
    'avg_speed': 3.02,
    'avg_time_in_air': 1.13,
    'avg_output_frames': 11.29,

    # Correlations
    'x_to_ball_land_x': 0.86,
    'x_to_yardline': 0.94,
    'speed_to_accel': 0.21,
}

print(f"Configuration loaded. Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
