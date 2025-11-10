"""
Data Preprocessing and Feature Engineering
Based on EDA insights with strong correlations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import glob
import config

class FeatureEngineering:
    """
    Feature engineering based on EDA insights:
    - x ↔ ball_land_x correlation: 0.86 (very strong!)
    - Role-based speed differences
    - Distance to ball is critical for prediction
    """

    def __init__(self):
        self.scalers = {
            'position': StandardScaler(),
            'velocity': StandardScaler(),
            'angles': StandardScaler(),
        }
        self.fitted = False

    def calculate_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relative features to ball landing location"""

        # Distance to ball (critical feature based on EDA)
        df['dx_to_ball'] = df['ball_land_x'] - df['x']
        df['dy_to_ball'] = df['ball_land_y'] - df['y']
        df['dist_to_ball'] = np.sqrt(df['dx_to_ball']**2 + df['dy_to_ball']**2)

        # Velocity components (from speed and direction)
        df['velocity_x'] = df['s'] * np.cos(np.radians(df['dir']))
        df['velocity_y'] = df['s'] * np.sin(np.radians(df['dir']))

        # Angle to ball
        df['angle_to_ball'] = np.degrees(np.arctan2(df['dy_to_ball'], df['dx_to_ball']))

        # Speed component toward ball
        angle_diff = np.radians(df['angle_to_ball'] - df['dir'])
        df['speed_to_ball'] = df['s'] * np.cos(angle_diff)

        return df

    def normalize_by_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize positions by field dimensions"""
        df['x_norm'] = df['x'] / config.FIELD_LENGTH
        df['y_norm'] = df['y'] / config.FIELD_WIDTH
        df['ball_land_x_norm'] = df['ball_land_x'] / config.FIELD_LENGTH
        df['ball_land_y_norm'] = df['ball_land_y'] / config.FIELD_WIDTH
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""

        # Role encoding (one-hot)
        df['role_defensive'] = (df['player_role'] == 'Defensive Coverage').astype(float)
        df['role_target'] = (df['player_role'] == 'Targeted Receiver').astype(float)
        df['role_other_route'] = (df['player_role'] == 'Other Route Runner').astype(float)
        df['role_passer'] = (df['player_role'] == 'Passer').astype(float)

        # Side encoding
        df['side_offense'] = (df['player_side'] == 'Offense').astype(float)
        df['side_defense'] = (df['player_side'] == 'Defense').astype(float)

        # Play direction
        df['play_dir_right'] = (df['play_direction'] == 'right').astype(float)

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features (differences between frames)"""

        # Sort by game, play, player, frame
        df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])

        # Calculate frame-to-frame changes
        for group_cols in [['game_id', 'play_id', 'nfl_id']]:
            df['dx_frame'] = df.groupby(group_cols)['x'].diff().fillna(0)
            df['dy_frame'] = df.groupby(group_cols)['y'].diff().fillna(0)
            df['ds_frame'] = df.groupby(group_cols)['s'].diff().fillna(0)
            df['da_frame'] = df.groupby(group_cols)['a'].diff().fillna(0)

        return df

    def fit(self, df: pd.DataFrame):
        """Fit scalers on training data"""

        # Fit position scaler
        position_features = ['x_norm', 'y_norm', 'ball_land_x_norm', 'ball_land_y_norm',
                           'dx_to_ball', 'dy_to_ball', 'dist_to_ball']
        self.scalers['position'].fit(df[position_features])

        # Fit velocity scaler
        velocity_features = ['s', 'a', 'velocity_x', 'velocity_y', 'speed_to_ball',
                            'dx_frame', 'dy_frame', 'ds_frame', 'da_frame']
        self.scalers['velocity'].fit(df[velocity_features])

        # Fit angle scaler
        angle_features = ['dir', 'o', 'angle_to_ball']
        self.scalers['angles'].fit(df[angle_features])

        self.fitted = True
        print("Feature scalers fitted successfully")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers"""

        if not self.fitted:
            raise ValueError("Scalers not fitted. Call fit() first.")

        # Apply scalers
        position_features = ['x_norm', 'y_norm', 'ball_land_x_norm', 'ball_land_y_norm',
                           'dx_to_ball', 'dy_to_ball', 'dist_to_ball']
        df[position_features] = self.scalers['position'].transform(df[position_features])

        velocity_features = ['s', 'a', 'velocity_x', 'velocity_y', 'speed_to_ball',
                            'dx_frame', 'dy_frame', 'ds_frame', 'da_frame']
        df[velocity_features] = self.scalers['velocity'].transform(df[velocity_features])

        angle_features = ['dir', 'o', 'angle_to_ball']
        df[angle_features] = self.scalers['angles'].transform(df[angle_features])

        return df

    def process(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Full feature engineering pipeline"""

        print(f"Processing {len(df):,} records...")

        # Add engineered features
        df = self.calculate_relative_features(df)
        df = self.normalize_by_field(df)
        df = self.encode_categorical(df)
        df = self.add_temporal_features(df)

        # Fit or transform
        if fit:
            self.fit(df)

        df = self.transform(df)

        print(f"Feature engineering complete. Shape: {df.shape}")
        return df


class NFLDataset(Dataset):
    """
    PyTorch Dataset for NFL player tracking data
    Returns sequences of player positions for trajectory prediction
    """

    def __init__(self, input_df: pd.DataFrame, output_df: pd.DataFrame = None,
                 sequence_length: int = 20, is_test: bool = False):
        """
        Args:
            input_df: Input tracking data (pre-throw)
            output_df: Output tracking data (post-throw), None for test
            sequence_length: Number of input frames to use
            is_test: Whether this is test data
        """
        self.input_df = input_df
        self.output_df = output_df
        self.sequence_length = sequence_length
        self.is_test = is_test

        # Get unique plays
        self.plays = input_df.groupby(['game_id', 'play_id']).size().reset_index()[['game_id', 'play_id']]

        print(f"Dataset created with {len(self.plays)} plays")

    def __len__(self):
        return len(self.plays)

    def __getitem__(self, idx):
        game_id, play_id = self.plays.iloc[idx][['game_id', 'play_id']]

        # Get input data for this play
        play_input = self.input_df[
            (self.input_df['game_id'] == game_id) &
            (self.input_df['play_id'] == play_id)
        ].sort_values(['nfl_id', 'frame_id'])

        # Get players in this play
        players = play_input['nfl_id'].unique()

        # Feature columns
        feature_cols = [
            'x_norm', 'y_norm', 's', 'a', 'dir', 'o',
            'dx_to_ball', 'dy_to_ball', 'dist_to_ball',
            'velocity_x', 'velocity_y', 'angle_to_ball', 'speed_to_ball',
            'dx_frame', 'dy_frame', 'ds_frame', 'da_frame',
            'role_defensive', 'role_target', 'role_other_route', 'role_passer',
            'side_offense', 'side_defense', 'play_dir_right',
            'ball_land_x_norm', 'ball_land_y_norm'
        ]

        # Build input sequence for each player
        input_sequences = []
        target_sequences = []
        player_ids = []
        num_output_frames_list = []

        for player_id in players:
            player_input = play_input[play_input['nfl_id'] == player_id].sort_values('frame_id')

            # Take last sequence_length frames (or all if fewer)
            if len(player_input) >= self.sequence_length:
                player_seq = player_input.iloc[-self.sequence_length:]
            else:
                # Pad with first frame if needed
                pad_length = self.sequence_length - len(player_input)
                first_frame = player_input.iloc[[0]].copy()
                padding = pd.concat([first_frame] * pad_length, ignore_index=True)
                player_seq = pd.concat([padding, player_input], ignore_index=True)

            # Get features
            features = player_seq[feature_cols].values
            input_sequences.append(features)

            # Get target if not test
            if not self.is_test and self.output_df is not None:
                player_output = self.output_df[
                    (self.output_df['game_id'] == game_id) &
                    (self.output_df['play_id'] == play_id) &
                    (self.output_df['nfl_id'] == player_id)
                ].sort_values('frame_id')

                if len(player_output) > 0:
                    # Target is future positions (x, y)
                    target = player_output[['x', 'y']].values
                    target_sequences.append(target)
                    player_ids.append(player_id)
                    num_output_frames_list.append(len(player_output))

        # Convert to tensors
        input_tensor = torch.FloatTensor(np.array(input_sequences))  # [num_players, seq_len, features]

        if self.is_test or self.output_df is None:
            return {
                'input': input_tensor,
                'game_id': game_id,
                'play_id': play_id,
                'player_ids': np.array(player_ids) if player_ids else np.array(players),
                'num_players': len(players)
            }
        else:
            # Pad targets to same length (max frames in this play)
            if len(target_sequences) > 0:
                max_frames = max(len(t) for t in target_sequences)
                padded_targets = []
                masks = []

                for target in target_sequences:
                    pad_length = max_frames - len(target)
                    if pad_length > 0:
                        padding = np.zeros((pad_length, 2))
                        target_padded = np.vstack([target, padding])
                        mask = np.array([1] * len(target) + [0] * pad_length)
                    else:
                        target_padded = target
                        mask = np.ones(len(target))

                    padded_targets.append(target_padded)
                    masks.append(mask)

                target_tensor = torch.FloatTensor(np.array(padded_targets))  # [num_players, max_frames, 2]
                mask_tensor = torch.FloatTensor(np.array(masks))  # [num_players, max_frames]
            else:
                target_tensor = torch.zeros(len(players), 1, 2)
                mask_tensor = torch.zeros(len(players), 1)

            return {
                'input': input_tensor,
                'target': target_tensor,
                'mask': mask_tensor,
                'game_id': game_id,
                'play_id': play_id,
                'player_ids': np.array(player_ids),
                'num_players': len(player_ids),
                'num_output_frames': np.array(num_output_frames_list)
            }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable number of players per play
    Pads to maximum number of players in the batch
    """
    # Find max number of players in this batch
    max_players = max(item['num_players'] for item in batch)

    batch_size = len(batch)
    seq_len = batch[0]['input'].shape[1]
    num_features = batch[0]['input'].shape[2]

    # Initialize padded tensors
    padded_inputs = torch.zeros(batch_size, max_players, seq_len, num_features)

    game_ids = []
    play_ids = []
    player_ids_list = []
    num_players_list = []

    # Handle targets if present
    has_targets = 'target' in batch[0]

    if has_targets:
        max_frames = max(item['target'].shape[1] for item in batch)
        padded_targets = torch.zeros(batch_size, max_players, max_frames, 2)
        padded_masks = torch.zeros(batch_size, max_players, max_frames)
        num_output_frames_list = []

    # Pad each item
    for i, item in enumerate(batch):
        num_players = item['num_players']

        # Pad input
        padded_inputs[i, :num_players] = item['input']

        # Store metadata
        game_ids.append(item['game_id'])
        play_ids.append(item['play_id'])
        player_ids_list.append(item['player_ids'])
        num_players_list.append(num_players)

        if has_targets:
            num_frames = item['target'].shape[1]
            padded_targets[i, :num_players, :num_frames] = item['target']
            padded_masks[i, :num_players, :num_frames] = item['mask']
            num_output_frames_list.append(item['num_output_frames'])

    result = {
        'input': padded_inputs,
        'game_id': game_ids,
        'play_id': play_ids,
        'player_ids': player_ids_list,
        'num_players': num_players_list,
    }

    if has_targets:
        result['target'] = padded_targets
        result['mask'] = padded_masks
        result['num_output_frames'] = num_output_frames_list

    return result


def load_and_preprocess_data(train_weeks: List[int] = None,
                             val_weeks: List[int] = None) -> Tuple[NFLDataset, NFLDataset]:
    """
    Load and preprocess training data

    Args:
        train_weeks: List of weeks to use for training (1-18)
        val_weeks: List of weeks to use for validation

    Returns:
        train_dataset, val_dataset
    """

    print("Loading training data...")

    # Load all input and output files
    input_files = sorted(glob.glob(f'{config.DATA_DIR}input_2023_w*.csv'))
    output_files = sorted(glob.glob(f'{config.DATA_DIR}output_2023_w*.csv'))

    # Default split if not specified
    if train_weeks is None:
        train_weeks = list(range(1, 16))  # Weeks 1-15 for training
    if val_weeks is None:
        val_weeks = [16, 17, 18]  # Weeks 16-18 for validation

    print(f"Training weeks: {train_weeks}")
    print(f"Validation weeks: {val_weeks}")

    # Load training data
    train_input_dfs = []
    train_output_dfs = []
    for week in train_weeks:
        inp_df = pd.read_csv(input_files[week-1])
        out_df = pd.read_csv(output_files[week-1])
        train_input_dfs.append(inp_df)
        train_output_dfs.append(out_df)

    train_input = pd.concat(train_input_dfs, ignore_index=True)
    train_output = pd.concat(train_output_dfs, ignore_index=True)

    # Load validation data
    val_input_dfs = []
    val_output_dfs = []
    for week in val_weeks:
        inp_df = pd.read_csv(input_files[week-1])
        out_df = pd.read_csv(output_files[week-1])
        val_input_dfs.append(inp_df)
        val_output_dfs.append(out_df)

    val_input = pd.concat(val_input_dfs, ignore_index=True)
    val_output = pd.concat(val_output_dfs, ignore_index=True)

    print(f"Train input: {len(train_input):,} records")
    print(f"Train output: {len(train_output):,} records")
    print(f"Val input: {len(val_input):,} records")
    print(f"Val output: {len(val_output):,} records")

    # Feature engineering
    feat_eng = FeatureEngineering()

    # Process training data (fit and transform)
    train_input = feat_eng.process(train_input, fit=True)
    train_output_processed = train_output.copy()

    # Process validation data (transform only)
    val_input = feat_eng.process(val_input, fit=False)
    val_output_processed = val_output.copy()

    # Create datasets
    train_dataset = NFLDataset(train_input, train_output_processed,
                               sequence_length=config.LSTM_CONFIG['sequence_length'])
    val_dataset = NFLDataset(val_input, val_output_processed,
                            sequence_length=config.LSTM_CONFIG['sequence_length'])

    return train_dataset, val_dataset, feat_eng


if __name__ == '__main__':
    # Test data loading
    print("Testing data preprocessing...")
    train_ds, val_ds, feat_eng = load_and_preprocess_data()

    print(f"\nTrain dataset: {len(train_ds)} plays")
    print(f"Val dataset: {len(val_ds)} plays")

    # Test batch loading
    sample = train_ds[0]
    print(f"\nSample batch:")
    print(f"  Input shape: {sample['input'].shape}")
    print(f"  Target shape: {sample['target'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Num players: {sample['num_players']}")

    print("\nData preprocessing test successful!")
