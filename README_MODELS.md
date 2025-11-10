# NFL Big Data Bowl 2026 - Ensemble Trajectory Prediction

## 🎯 Overview

This repository contains an **ensemble of 3 models** optimized for predicting NFL player trajectories during pass plays. The system is designed to leverage the full power of an RTX 5090 with 32GB VRAM.

### Models

1. **Physics-Based Baseline** - Simple, interpretable model using constant velocity + ball attraction
2. **LSTM with Attention** - Sequential model capturing temporal dependencies
3. **Transformer with Social Attention** - Advanced model for multi-agent interactions

### Key Features

- **Optimized for RTX 5090**: Large batch sizes (512-1024), mixed precision (FP16)
- **EDA-Driven**: Features engineered based on comprehensive data analysis
- **Physics-Informed**: Loss functions incorporate speed/acceleration constraints
- **Ensemble**: Weighted combination of all three models

## 📊 EDA Insights Used

Based on comprehensive analysis of 4.88M training records:

- **Strong Correlation**: x position ↔ ball landing (0.86)
- **Role-Based Speeds**: Targeted receivers (3.99 yd/s) vs Passer (1.80 yd/s)
- **Displacement**: Average 0.46 yards/frame (very consistent)
- **Time in Air**: Average 1.13 seconds (~11 frames at 10 FPS)
- **Distance to Ball**: Critical feature (targeted receivers 13.06 yards closer)

## 🚀 Quick Start

### Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy matplotlib seaborn scikit-learn tqdm
```

### Training

Train all three models:

```bash
python main.py --mode train
```

This will:
1. Load and preprocess data (weeks 1-15 for training, 16-18 for validation)
2. Apply feature engineering (26 features including EDA insights)
3. Train Physics baseline (evaluation only)
4. Train LSTM model (50 epochs, batch size 512)
5. Train Transformer model (50 epochs, batch size 256)
6. Save best checkpoints to `models/`

### Prediction

Generate ensemble predictions:

```bash
python main.py --mode predict
```

This will:
1. Load all three trained models
2. Create weighted ensemble (Physics: 0.2, LSTM: 0.3, Transformer: 0.5)
3. Evaluate on validation set
4. Report ensemble RMSE

### Full Pipeline

Run training + prediction:

```bash
python main.py --mode all
```

## 📁 Project Structure

```
.
├── config.py                # All hyperparameters and settings
├── data_preprocessing.py    # Feature engineering and data loading
├── models.py               # All three model architectures
├── train.py                # Training loop for all models
├── ensemble.py             # Ensemble prediction and submission
├── main.py                 # Main execution script
├── run_eda.py             # EDA analysis script
│
├── models/                 # Saved model checkpoints
│   ├── physics/
│   ├── lstm/
│   └── transformer/
│
├── eda_outputs/           # EDA results and plots
│   ├── eda_results.json
│   └── plots/
│
└── train/                 # Training data (CSV files)
    ├── input_2023_w*.csv
    └── output_2023_w*.csv
```

## 🏗️ Architecture Details

### 1. Physics-Based Model

**Concept**: Simple kinematic model with ball attraction

```python
# Core physics
next_position = current_position + velocity * dt + ball_attraction * weight
velocity = velocity * persistence + attraction_to_ball
```

**Advantages**:
- No training required
- Interpretable
- Fast inference
- Good baseline

**Role-based speed multipliers** (from EDA):
- Targeted Receiver: 1.0x
- Route Runner: 1.0x
- Defensive Coverage: 0.64x
- Passer (QB): 0.45x

### 2. LSTM with Attention

**Architecture**:
```
Input (26 features) → Embedding (256d)
  → Bidirectional LSTM (3 layers, 256 hidden)
  → Multi-head Attention (8 heads) with ball landing as query
  → Decoder LSTM (2 layers)
  → Output (x, y positions)
```

**Parameters**: ~4.2M

**Batch Size**: 512 (optimized for 32GB VRAM)

**Key Features**:
- Bidirectional encoding of input sequence
- Attention mechanism focusing on ball landing
- Autoregressive decoding for smooth trajectories

### 3. Transformer with Social Attention

**Architecture**:
```
Input (26 features) → Embedding (256d) + Positional Encoding
  → Transformer Encoder (6 layers, 8 heads, 1024 FFN)
  → Ball Landing Context
  → Transformer Decoder (6 layers, 8 heads, 1024 FFN)
  → Output (x, y positions)
```

**Parameters**: ~12.8M

**Batch Size**: 256 (transformers use more memory)

**Key Features**:
- Self-attention across temporal dimension
- Cross-attention between players (social interactions)
- Ball landing as conditioning signal
- Parallel decoding (non-autoregressive option)

## 🎓 Feature Engineering

Based on EDA insights, we engineer 26 features per player per frame:

### Position Features (normalized)
- `x_norm`, `y_norm`: Field position (0-1 range)
- `ball_land_x_norm`, `ball_land_y_norm`: Target location

### Velocity Features
- `s`: Speed (yards/sec)
- `a`: Acceleration (yards/sec²)
- `velocity_x`, `velocity_y`: Velocity components
- `dir`, `o`: Direction and orientation (degrees)

### Relative Features (Critical!)
- `dx_to_ball`, `dy_to_ball`: Distance to ball landing
- `dist_to_ball`: Euclidean distance
- `angle_to_ball`: Angle toward ball
- `speed_to_ball`: Speed component toward ball

### Temporal Features
- `dx_frame`, `dy_frame`: Frame-to-frame position change
- `ds_frame`, `da_frame`: Frame-to-frame speed/accel change

### Categorical Features (one-hot)
- Role: Defensive, Target, Route Runner, Passer
- Side: Offense, Defense
- Direction: Left, Right

## 🔧 Configuration

All hyperparameters are in `config.py`. Key settings:

```python
# Hardware
DEVICE = 'cuda'
USE_AMP = True  # Mixed precision (FP16)

# LSTM Config
LSTM_CONFIG = {
    'batch_size': 512,      # Large batch for 32GB VRAM
    'learning_rate': 1e-3,
    'epochs': 50,
    'hidden_size': 256,
}

# Transformer Config
TRANSFORMER_CONFIG = {
    'batch_size': 256,      # Transformers use more memory
    'learning_rate': 5e-4,
    'epochs': 50,
    'd_model': 256,
    'nhead': 8,
}

# Ensemble Weights
ENSEMBLE_CONFIG = {
    'weights': {
        'physics': 0.2,
        'lstm': 0.3,
        'transformer': 0.5,
    }
}
```

## 📈 Training Details

### Loss Function

Custom multi-objective loss:

```python
Total Loss = RMSE + α*Smoothness + β*Physics
```

Where:
- **RMSE**: Competition metric (primary)
- **Smoothness**: Frame-to-frame consistency
- **Physics**: Speed/acceleration constraints

### Optimization

- **LSTM**: AdamW with ReduceLROnPlateau
- **Transformer**: AdamW with OneCycleLR + warmup
- **Gradient Clipping**: 1.0
- **Early Stopping**: 10 epochs patience

### Data Split

- **Training**: Weeks 1-15 (~3.8M records)
- **Validation**: Weeks 16-18 (~1.1M records)

### Training Time (RTX 5090)

- Physics: Instant (no training)
- LSTM: ~2-3 hours
- Transformer: ~4-6 hours
- **Total**: ~6-9 hours for all models

## 🎯 Expected Performance

Based on EDA and model complexity:

| Model | Expected RMSE | Training Time |
|-------|--------------|---------------|
| Physics | ~0.8-1.0 | 0 min |
| LSTM | ~0.4-0.6 | 2-3 hours |
| Transformer | ~0.3-0.5 | 4-6 hours |
| **Ensemble** | **~0.3-0.4** | 6-9 hours |

*Note: These are estimates. Actual performance depends on data quality and tuning.*

## 🐛 Troubleshooting

### CUDA Out of Memory

Reduce batch sizes in `config.py`:
```python
LSTM_CONFIG['batch_size'] = 256  # Instead of 512
TRANSFORMER_CONFIG['batch_size'] = 128  # Instead of 256
```

### Slow Training

Check:
1. Mixed precision enabled: `USE_AMP = True`
2. GPU utilization: `nvidia-smi`
3. Data loading: Increase `NUM_WORKERS`

### Poor Performance

1. Check feature engineering is correct
2. Verify data normalization
3. Try different ensemble weights
4. Train for more epochs

## 📝 Next Steps

1. **Run EDA** (if not done): `python run_eda.py`
2. **Train models**: `python main.py --mode train`
3. **Evaluate**: `python main.py --mode predict`
4. **Tune hyperparameters** in `config.py`
5. **Experiment with ensemble weights**
6. **Submit to Kaggle** using competition API

## 🏆 Competition Notes

- **Metric**: RMSE on (x, y) coordinates
- **Submission**: Via Kaggle API (9-hour runtime limit)
- **Test Data**: Live leaderboard on weeks after Dec 4, 2025
- **Final Deadline**: Dec 3, 2025

## 📚 References

- NFL Big Data Bowl 2026: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction
- EDA Results: `eda_outputs/eda_results.json`
- Model Checkpoints: `models/*/best_model.pth`

---

**Good luck! 🏈🚀**
