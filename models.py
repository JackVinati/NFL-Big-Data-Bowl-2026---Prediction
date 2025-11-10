"""
Three model architectures for ensemble:
1. Physics-based model
2. LSTM with attention
3. Transformer with social attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import config


# ============================================================================
# MODEL 1: PHYSICS-BASED BASELINE
# ============================================================================

class PhysicsModel(nn.Module):
    """
    Physics-informed baseline model
    Based on EDA insights:
    - Average displacement: 0.46 yards/frame
    - Players move toward ball landing (especially targeted receivers)
    - Role-specific speed differences
    """

    def __init__(self):
        super(PhysicsModel, self).__init__()
        self.ball_attraction = config.PHYSICS_CONFIG['ball_attraction_weight']
        self.velocity_weight = config.PHYSICS_CONFIG['velocity_weight']
        self.smoothing = config.PHYSICS_CONFIG['smoothing_factor']
        self.role_multipliers = config.PHYSICS_CONFIG['role_speed_multipliers']

    def forward(self, input_seq, num_frames_to_predict):
        """
        Args:
            input_seq: [batch, seq_len, features]
            num_frames_to_predict: int or list of ints

        Returns:
            predictions: [batch, max_pred_frames, 2]  # (x, y) positions
        """
        batch_size = input_seq.shape[0]
        device = input_seq.device

        # Extract last frame features
        last_frame = input_seq[:, -1, :]  # [batch, features]

        # Parse features (based on feature engineering)
        # [x_norm, y_norm, s, a, dir, o, dx_to_ball, dy_to_ball, dist_to_ball,
        #  velocity_x, velocity_y, angle_to_ball, speed_to_ball,
        #  dx_frame, dy_frame, ds_frame, da_frame,
        #  role_defensive, role_target, role_other_route, role_passer,
        #  side_offense, side_defense, play_dir_right,
        #  ball_land_x_norm, ball_land_y_norm]

        x_pos = last_frame[:, 0]  # x_norm
        y_pos = last_frame[:, 1]  # y_norm
        speed = last_frame[:, 2]  # s
        velocity_x = last_frame[:, 9]  # velocity_x
        velocity_y = last_frame[:, 10]  # velocity_y
        dx_to_ball = last_frame[:, 6]  # dx_to_ball
        dy_to_ball = last_frame[:, 7]  # dy_to_ball
        dist_to_ball = last_frame[:, 8]  # dist_to_ball

        # Role indicators
        role_target = last_frame[:, 18]  # role_target
        role_defensive = last_frame[:, 17]  # role_defensive
        role_passer = last_frame[:, 20]  # role_passer

        # Calculate role multipliers
        speed_multiplier = torch.ones(batch_size, device=device)
        speed_multiplier = torch.where(role_target > 0.5, speed_multiplier * 1.0, speed_multiplier)
        speed_multiplier = torch.where(role_defensive > 0.5, speed_multiplier * 0.64, speed_multiplier)
        speed_multiplier = torch.where(role_passer > 0.5, speed_multiplier * 0.45, speed_multiplier)

        # Determine max frames
        if isinstance(num_frames_to_predict, int):
            max_frames = num_frames_to_predict
        else:
            max_frames = max(num_frames_to_predict)

        # Generate predictions frame by frame
        predictions = []
        current_x = x_pos
        current_y = y_pos
        current_vx = velocity_x
        current_vy = velocity_y

        for frame in range(max_frames):
            # Calculate attraction toward ball
            attraction_x = dx_to_ball * self.ball_attraction
            attraction_y = dy_to_ball * self.ball_attraction

            # Combine velocity and attraction
            next_vx = current_vx * self.velocity_weight + attraction_x
            next_vy = current_vy * self.velocity_weight + attraction_y

            # Apply role-based speed scaling
            next_vx = next_vx * speed_multiplier
            next_vy = next_vy * speed_multiplier

            # Update position (velocity is already in normalized space)
            next_x = current_x + next_vx * (1.0 / config.FRAMES_PER_SECOND)
            next_y = current_y + next_vy * (1.0 / config.FRAMES_PER_SECOND)

            # Apply smoothing
            next_x = current_x + (next_x - current_x) * (1 - self.smoothing)
            next_y = current_y + (next_y - current_y) * (1 - self.smoothing)

            # Store prediction
            pred_frame = torch.stack([next_x, next_y], dim=1)  # [batch, 2]
            predictions.append(pred_frame)

            # Update for next iteration
            current_x = next_x
            current_y = next_y
            current_vx = next_vx
            current_vy = next_vy

            # Update distance to ball
            dx_to_ball = dx_to_ball - (next_x - current_x)
            dy_to_ball = dy_to_ball - (next_y - current_y)

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [batch, max_frames, 2]

        return predictions


# ============================================================================
# MODEL 2: LSTM WITH ATTENTION
# ============================================================================

class LSTMAttentionModel(nn.Module):
    """
    LSTM model with attention to ball landing location
    Good at capturing temporal dependencies
    """

    def __init__(self, input_size=26, hidden_size=256, num_layers=3,
                 dropout=0.2, bidirectional=True, attention_heads=8):
        super(LSTMAttentionModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size * self.num_directions,
            attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Ball landing context
        self.ball_context = nn.Sequential(
            nn.Linear(2, hidden_size),  # ball_land_x, ball_land_y
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * self.num_directions)
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            hidden_size * self.num_directions + 2,  # + previous position
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # (x, y)
        )

    def forward(self, input_seq, num_frames_to_predict=11):
        """
        Args:
            input_seq: [batch, seq_len, features]
            num_frames_to_predict: int

        Returns:
            predictions: [batch, num_frames, 2]
        """
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device

        # Project input
        x = self.input_proj(input_seq)  # [batch, seq_len, hidden]

        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, seq_len, hidden*directions]

        # Get ball landing context from last frame
        ball_pos = input_seq[:, -1, -2:]  # [batch, 2] (ball_land_x_norm, ball_land_y_norm)
        ball_context = self.ball_context(ball_pos)  # [batch, hidden*directions]
        ball_context = ball_context.unsqueeze(1)  # [batch, 1, hidden*directions]

        # Apply attention with ball as query
        attended, _ = self.attention(
            ball_context,  # query
            lstm_out,      # key
            lstm_out       # value
        )  # [batch, 1, hidden*directions]

        # Prepare decoder
        # Use attended features + last hidden state
        decoder_hidden = hidden
        decoder_cell = cell

        # Start with last known position
        current_pos = input_seq[:, -1, :2]  # [batch, 2] (x_norm, y_norm)

        # Decode autoregressively
        predictions = []
        decoder_input = torch.cat([attended.squeeze(1), current_pos], dim=1)  # [batch, hidden*dir+2]
        decoder_input = decoder_input.unsqueeze(1)  # [batch, 1, hidden*dir+2]

        for _ in range(num_frames_to_predict):
            # Decode one step
            decoder_out, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input, (decoder_hidden, decoder_cell)
            )  # [batch, 1, hidden]

            # Predict position
            pred_pos = self.output_proj(decoder_out.squeeze(1))  # [batch, 2]
            predictions.append(pred_pos)

            # Next decoder input
            decoder_input = torch.cat([attended.squeeze(1), pred_pos], dim=1)
            decoder_input = decoder_input.unsqueeze(1)

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [batch, num_frames, 2]

        return predictions


# ============================================================================
# MODEL 3: TRANSFORMER WITH SOCIAL ATTENTION
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerTrajectoryModel(nn.Module):
    """
    Transformer model with social attention for multi-agent trajectory prediction
    Best for modeling complex player interactions
    """

    def __init__(self, input_size=26, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1):
        super(TransformerTrajectoryModel, self).__init__()

        self.d_model = d_model

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Ball landing embedding
        self.ball_embedding = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # (x, y)
        )

        # Learnable query embeddings for future frames
        self.query_embed = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, input_seq, num_frames_to_predict=11):
        """
        Args:
            input_seq: [batch, seq_len, features]
            num_frames_to_predict: int

        Returns:
            predictions: [batch, num_frames, 2]
        """
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device

        # Embed input
        x = self.input_embedding(input_seq)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encode
        memory = self.transformer_encoder(x)  # [batch, seq_len, d_model]

        # Get ball landing embedding
        ball_pos = input_seq[:, -1, -2:]  # [batch, 2]
        ball_embed = self.ball_embedding(ball_pos)  # [batch, d_model]
        ball_embed = ball_embed.unsqueeze(1)  # [batch, 1, d_model]

        # Create query embeddings for future frames
        queries = self.query_embed.expand(batch_size, num_frames_to_predict, -1)  # [batch, num_frames, d_model]

        # Add ball context to queries
        queries = queries + ball_embed

        # Decode
        tgt = queries
        output = self.transformer_decoder(tgt, memory)  # [batch, num_frames, d_model]

        # Project to positions
        predictions = self.output_head(output)  # [batch, num_frames, 2]

        return predictions


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type='lstm', **kwargs):
    """
    Factory function to create models

    Args:
        model_type: 'physics', 'lstm', or 'transformer'
        **kwargs: Model-specific arguments

    Returns:
        model instance
    """

    if model_type == 'physics':
        return PhysicsModel()

    elif model_type == 'lstm':
        return LSTMAttentionModel(
            input_size=kwargs.get('input_size', 26),
            hidden_size=config.LSTM_CONFIG['hidden_size'],
            num_layers=config.LSTM_CONFIG['num_layers'],
            dropout=config.LSTM_CONFIG['dropout'],
            bidirectional=config.LSTM_CONFIG['bidirectional'],
            attention_heads=config.LSTM_CONFIG['attention_heads']
        )

    elif model_type == 'transformer':
        return TransformerTrajectoryModel(
            input_size=kwargs.get('input_size', 26),
            d_model=config.TRANSFORMER_CONFIG['d_model'],
            nhead=config.TRANSFORMER_CONFIG['nhead'],
            num_encoder_layers=config.TRANSFORMER_CONFIG['num_encoder_layers'],
            num_decoder_layers=config.TRANSFORMER_CONFIG['num_decoder_layers'],
            dim_feedforward=config.TRANSFORMER_CONFIG['dim_feedforward'],
            dropout=config.TRANSFORMER_CONFIG['dropout']
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test models
    print("Testing model architectures...")

    batch_size = 4
    seq_len = 20
    num_features = 26
    num_frames = 11

    dummy_input = torch.randn(batch_size, seq_len, num_features)

    print("\n1. Physics Model:")
    physics_model = create_model('physics')
    physics_out = physics_model(dummy_input, num_frames)
    print(f"   Output shape: {physics_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in physics_model.parameters()):,}")

    print("\n2. LSTM Model:")
    lstm_model = create_model('lstm')
    lstm_out = lstm_model(dummy_input, num_frames)
    print(f"   Output shape: {lstm_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    print("\n3. Transformer Model:")
    transformer_model = create_model('transformer')
    transformer_out = transformer_model(dummy_input, num_frames)
    print(f"   Output shape: {transformer_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")

    print("\nAll models initialized successfully!")
