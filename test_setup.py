"""
Quick test script to verify all components work
Run this before starting full training
"""

import torch
import numpy as np
from pathlib import Path

print("="*80)
print("TESTING NFL BIG DATA BOWL 2026 - ENSEMBLE SETUP")
print("="*80)

# Test 1: Imports
print("\n1. Testing imports...")
try:
    import config
    from models import create_model
    from data_preprocessing import FeatureEngineering, NFLDataset
    from train import Trainer, TrajectoryLoss
    from ensemble import EnsemblePredictor
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    exit(1)

# Test 2: GPU
print("\n2. Testing GPU...")
if torch.cuda.is_available():
    print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = 'cuda'
else:
    print("   ⚠ GPU not available, will use CPU")
    device = 'cpu'

# Test 3: Model creation
print("\n3. Testing model creation...")
try:
    physics_model = create_model('physics')
    lstm_model = create_model('lstm', input_size=26)
    transformer_model = create_model('transformer', input_size=26)

    print(f"   ✓ Physics model: {sum(p.numel() for p in physics_model.parameters()):,} params")
    print(f"   ✓ LSTM model: {sum(p.numel() for p in lstm_model.parameters()):,} params")
    print(f"   ✓ Transformer model: {sum(p.numel() for p in transformer_model.parameters()):,} params")
except Exception as e:
    print(f"   ✗ Model creation error: {e}")
    exit(1)

# Test 4: Model forward pass
print("\n4. Testing model forward pass...")
try:
    batch_size = 4
    seq_len = 20
    num_features = 26
    num_frames = 11

    dummy_input = torch.randn(batch_size, seq_len, num_features).to(device)

    physics_model = physics_model.to(device)
    lstm_model = lstm_model.to(device)
    transformer_model = transformer_model.to(device)

    with torch.no_grad():
        physics_out = physics_model(dummy_input, num_frames)
        lstm_out = lstm_model(dummy_input, num_frames)
        transformer_out = transformer_model(dummy_input, num_frames)

    assert physics_out.shape == (batch_size, num_frames, 2), f"Physics output shape mismatch: {physics_out.shape}"
    assert lstm_out.shape == (batch_size, num_frames, 2), f"LSTM output shape mismatch: {lstm_out.shape}"
    assert transformer_out.shape == (batch_size, num_frames, 2), f"Transformer output shape mismatch: {transformer_out.shape}"

    print(f"   ✓ Physics model output: {physics_out.shape}")
    print(f"   ✓ LSTM model output: {lstm_out.shape}")
    print(f"   ✓ Transformer model output: {transformer_out.shape}")
except Exception as e:
    print(f"   ✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Loss function
print("\n5. Testing loss function...")
try:
    criterion = TrajectoryLoss()

    predictions = torch.randn(batch_size, num_frames, 2).to(device)
    targets = torch.randn(batch_size, num_frames, 2).to(device)
    mask = torch.ones(batch_size, num_frames).to(device)

    loss, rmse, smooth, phys = criterion(predictions, targets, mask)

    print(f"   ✓ Loss computed: {loss.item():.4f}")
    print(f"     - RMSE: {rmse.item():.4f}")
    print(f"     - Smoothness: {smooth.item():.4f}")
    print(f"     - Physics: {phys.item():.4f}")
except Exception as e:
    print(f"   ✗ Loss function error: {e}")
    exit(1)

# Test 6: Feature engineering
print("\n6. Testing feature engineering...")
try:
    import pandas as pd

    # Create dummy data
    dummy_df = pd.DataFrame({
        'game_id': [1] * 100,
        'play_id': [1] * 100,
        'nfl_id': [1] * 100,
        'frame_id': list(range(100)),
        'x': np.random.uniform(0, 120, 100),
        'y': np.random.uniform(0, 53.3, 100),
        's': np.random.uniform(0, 10, 100),
        'a': np.random.uniform(0, 5, 100),
        'dir': np.random.uniform(0, 360, 100),
        'o': np.random.uniform(0, 360, 100),
        'ball_land_x': [60] * 100,
        'ball_land_y': [26.65] * 100,
        'player_role': ['Defensive Coverage'] * 100,
        'player_side': ['Defense'] * 100,
        'play_direction': ['right'] * 100,
    })

    feat_eng = FeatureEngineering()
    result_df = feat_eng.calculate_relative_features(dummy_df)

    required_features = ['dx_to_ball', 'dy_to_ball', 'dist_to_ball',
                        'velocity_x', 'velocity_y', 'angle_to_ball', 'speed_to_ball']

    for feat in required_features:
        assert feat in result_df.columns, f"Missing feature: {feat}"

    print(f"   ✓ Feature engineering successful")
    print(f"   ✓ Generated features: {len(result_df.columns)} columns")
except Exception as e:
    print(f"   ✗ Feature engineering error: {e}")
    exit(1)

# Test 7: Directory structure
print("\n7. Checking directory structure...")
required_dirs = ['train', 'eda_outputs', 'eda_outputs/plots']
for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"   ✓ {dir_path}/ exists")
    else:
        print(f"   ✗ {dir_path}/ not found")

# Test 8: Data files
print("\n8. Checking data files...")
import glob
input_files = glob.glob('train/input_2023_w*.csv')
output_files = glob.glob('train/output_2023_w*.csv')

if len(input_files) > 0:
    print(f"   ✓ Found {len(input_files)} input files")
else:
    print(f"   ✗ No input files found in train/")

if len(output_files) > 0:
    print(f"   ✓ Found {len(output_files)} output files")
else:
    print(f"   ✗ No output files found in train/")

# Test 9: Config
print("\n9. Checking configuration...")
print(f"   Device: {config.DEVICE}")
print(f"   Mixed Precision: {config.USE_AMP}")
print(f"   LSTM batch size: {config.LSTM_CONFIG['batch_size']}")
print(f"   Transformer batch size: {config.TRANSFORMER_CONFIG['batch_size']}")
print(f"   Ensemble weights: {config.ENSEMBLE_CONFIG['weights']}")

# Summary
print("\n" + "="*80)
print("SETUP TEST SUMMARY")
print("="*80)
print("✓ All core components working!")
print("\nReady to start training:")
print("  python main.py --mode train")
print("\nOr run full pipeline:")
print("  python main.py --mode all")
print("="*80)
