#!/usr/bin/env python3
"""
NFL Big Data Bowl 2026 - Comprehensive EDA Script
Analyzes player tracking data and saves results to JSON with plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directories
Path('eda_outputs').mkdir(exist_ok=True)
Path('eda_outputs/plots').mkdir(exist_ok=True)

print("="*80)
print("NFL BIG DATA BOWL 2026 - COMPREHENSIVE EDA")
print("="*80)
print(f"Started at: {datetime.now()}\n")

# Initialize results dictionary
eda_results = {
    'metadata': {'analysis_date': str(datetime.now())},
    'data_structure': {},
    'data_quality': {},
    'feature_analysis': {},
    'play_characteristics': {},
    'player_analysis': {},
    'movement_patterns': {},
    'ball_trajectory': {},
    'correlations': {}
}

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("1. LOADING DATA...")
input_files = sorted(glob.glob('train/input_2023_w*.csv'))
output_files = sorted(glob.glob('train/output_2023_w*.csv'))

print(f"   Found {len(input_files)} input files")
print(f"   Found {len(output_files)} output files")

input_dfs = []
output_dfs = []

for i, (inp_file, out_file) in enumerate(zip(input_files, output_files), 1):
    print(f"   Loading week {i:02d}...", end='\r')
    inp_df = pd.read_csv(inp_file)
    out_df = pd.read_csv(out_file)
    inp_df['week'] = i
    out_df['week'] = i
    input_dfs.append(inp_df)
    output_dfs.append(out_df)

df_input = pd.concat(input_dfs, ignore_index=True)
df_output = pd.concat(output_dfs, ignore_index=True)

print(f"\n   Total input records: {len(df_input):,}")
print(f"   Total output records: {len(df_output):,}")
print(f"   Memory: Input={df_input.memory_usage(deep=True).sum()/1024**2:.1f}MB, Output={df_output.memory_usage(deep=True).sum()/1024**2:.1f}MB")

eda_results['metadata']['total_weeks'] = len(input_files)
eda_results['data_structure'] = {
    'input_shape': list(df_input.shape),
    'output_shape': list(df_output.shape),
    'input_columns': list(df_input.columns),
    'output_columns': list(df_output.columns)
}

# =============================================================================
# 2. DATA QUALITY
# =============================================================================
print("\n2. ANALYZING DATA QUALITY...")

unique_games = df_input['game_id'].nunique()
unique_plays = df_input.groupby(['game_id', 'play_id']).ngroups
unique_players = df_input['nfl_id'].nunique()

eda_results['data_quality']['unique_counts'] = {
    'unique_games': int(unique_games),
    'unique_plays': int(unique_plays),
    'unique_players': int(unique_players)
}

print(f"   Unique games: {unique_games:,}")
print(f"   Unique plays: {unique_plays:,}")
print(f"   Unique players: {unique_players:,}")

# Missing values
input_missing = df_input.isnull().sum()
if input_missing.sum() > 0:
    print(f"   Input missing values: {input_missing[input_missing > 0].to_dict()}")

# =============================================================================
# 3. PLAY CHARACTERISTICS
# =============================================================================
print("\n3. ANALYZING PLAY CHARACTERISTICS...")

players_per_play = df_input.groupby(['game_id', 'play_id'])['nfl_id'].nunique()
input_frames_per_play = df_input.groupby(['game_id', 'play_id'])['frame_id'].max()
output_frames = df_input.groupby(['game_id', 'play_id', 'nfl_id'])['num_frames_output'].first()
time_in_air = output_frames / 10  # Convert to seconds

eda_results['play_characteristics'] = {
    'players_per_play': {
        'mean': float(players_per_play.mean()),
        'median': float(players_per_play.median()),
        'min': int(players_per_play.min()),
        'max': int(players_per_play.max())
    },
    'input_frames_per_play': {
        'mean': float(input_frames_per_play.mean()),
        'median': float(input_frames_per_play.median()),
        'min': int(input_frames_per_play.min()),
        'max': int(input_frames_per_play.max())
    },
    'output_frames_to_predict': {
        'mean': float(output_frames.mean()),
        'median': float(output_frames.median()),
        'min': int(output_frames.min()),
        'max': int(output_frames.max())
    },
    'ball_time_in_air_seconds': {
        'mean': float(time_in_air.mean()),
        'median': float(time_in_air.median()),
        'min': float(time_in_air.min()),
        'max': float(time_in_air.max())
    }
}

print(f"   Avg players per play: {players_per_play.mean():.1f}")
print(f"   Avg input frames: {input_frames_per_play.mean():.1f}")
print(f"   Avg output frames: {output_frames.mean():.1f}")
print(f"   Avg time in air: {time_in_air.mean():.2f}s")

# Plot 1: Players per play
fig, ax = plt.subplots(figsize=(10, 6))
players_per_play.value_counts().sort_index().plot(kind='bar', ax=ax)
ax.set_title('Distribution of Players per Play', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Players')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('eda_outputs/plots/01_players_per_play.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Output frames
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(output_frames, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(output_frames.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {output_frames.mean():.1f}')
axes[0].set_title('Output Frames to Predict', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Frames')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].hist(time_in_air, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(time_in_air.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {time_in_air.mean():.2f}s')
axes[1].set_title('Ball Time in Air', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
plt.tight_layout()
plt.savefig('eda_outputs/plots/02_output_frames_and_time.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. PLAYER ANALYSIS
# =============================================================================
print("\n4. ANALYZING PLAYERS...")

position_counts = df_input['player_position'].value_counts()
side_counts = df_input['player_side'].value_counts()
role_counts = df_input['player_role'].value_counts()

eda_results['player_analysis'] = {
    'position_distribution': position_counts.head(20).to_dict(),
    'side_distribution': side_counts.to_dict(),
    'role_distribution': role_counts.to_dict(),
    'players_to_predict': {
        'total_to_predict': int(df_input['player_to_predict'].sum()),
        'percentage': float(df_input['player_to_predict'].mean() * 100)
    }
}

print(f"   Top positions: {list(position_counts.head(5).index)}")
print(f"   Offense: {side_counts.get('Offense', 0):,} | Defense: {side_counts.get('Defense', 0):,}")
print(f"   Players to predict: {df_input['player_to_predict'].sum():,} ({df_input['player_to_predict'].mean()*100:.1f}%)")

# Plot 3: Player roles
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
position_counts.head(15).plot(kind='barh', ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Top 15 Player Positions', fontsize=12, fontweight='bold')

side_counts.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
axes[0, 1].set_title('Offense vs Defense', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=0)

role_counts.plot(kind='barh', ax=axes[1, 0], color='coral')
axes[1, 0].set_title('Player Roles', fontsize=12, fontweight='bold')

predict_counts = df_input['player_to_predict'].value_counts()
axes[1, 1].pie(predict_counts, labels=['Not Predicted', 'To Predict'], autopct='%1.1f%%',
               colors=['#95a5a6', '#3498db'], startangle=90)
axes[1, 1].set_title('Players to Predict', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_outputs/plots/03_player_roles_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 5. TRACKING FEATURES
# =============================================================================
print("\n5. ANALYZING TRACKING FEATURES...")

tracking_features = ['x', 'y', 's', 'a', 'o', 'dir']
for feat in tracking_features:
    eda_results['feature_analysis'][f'{feat}_stats'] = {
        'mean': float(df_input[feat].mean()),
        'median': float(df_input[feat].median()),
        'std': float(df_input[feat].std()),
        'min': float(df_input[feat].min()),
        'max': float(df_input[feat].max())
    }

print(f"   Speed: mean={df_input['s'].mean():.2f}, max={df_input['s'].max():.2f} yards/sec")
print(f"   Acceleration: mean={df_input['a'].mean():.2f}, max={df_input['a'].max():.2f} yards/sec²")

# Plot 4: Tracking features
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()
for idx, feat in enumerate(tracking_features):
    sample_data = df_input[feat].dropna().sample(min(100000, len(df_input)), random_state=42)
    axes[idx].hist(sample_data, bins=100, edgecolor='black', alpha=0.7)
    axes[idx].axvline(sample_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sample_data.mean():.2f}')
    axes[idx].set_title(f'Distribution of {feat.upper()}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feat)
    axes[idx].legend()
plt.tight_layout()
plt.savefig('eda_outputs/plots/04_tracking_features.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Speed by role
fig, ax = plt.subplots(figsize=(12, 6))
df_input.boxplot(column='s', by='player_role', ax=ax)
ax.set_title('Speed Distribution by Player Role', fontsize=12, fontweight='bold')
ax.set_xlabel('Player Role')
ax.set_ylabel('Speed (yards/sec)')
ax.get_figure().suptitle('')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_outputs/plots/05_speed_by_role.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. BALL TRAJECTORY
# =============================================================================
print("\n6. ANALYZING BALL TRAJECTORY...")

ball_positions = df_input.groupby(['game_id', 'play_id'])[['ball_land_x', 'ball_land_y']].first()
play_info = df_input.groupby(['game_id', 'play_id']).first()
play_info['pass_distance'] = np.abs(play_info['ball_land_x'] - play_info['absolute_yardline_number'])

eda_results['ball_trajectory'] = {
    'ball_land_x': {
        'mean': float(ball_positions['ball_land_x'].mean()),
        'std': float(ball_positions['ball_land_x'].std())
    },
    'ball_land_y': {
        'mean': float(ball_positions['ball_land_y'].mean()),
        'std': float(ball_positions['ball_land_y'].std())
    },
    'pass_distance': {
        'mean': float(play_info['pass_distance'].mean()),
        'median': float(play_info['pass_distance'].median()),
        'max': float(play_info['pass_distance'].max())
    }
}

print(f"   Avg pass distance: {play_info['pass_distance'].mean():.1f} yards")
print(f"   Ball landing X: {ball_positions['ball_land_x'].mean():.1f} ± {ball_positions['ball_land_x'].std():.1f}")
print(f"   Ball landing Y: {ball_positions['ball_land_y'].mean():.1f} ± {ball_positions['ball_land_y'].std():.1f}")

# Plot 6: Ball trajectory
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes[0, 0].hist(ball_positions['ball_land_x'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Ball Landing X Position', fontsize=12, fontweight='bold')
axes[0, 1].hist(ball_positions['ball_land_y'], bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_title('Ball Landing Y Position', fontsize=12, fontweight='bold')
axes[1, 0].hexbin(ball_positions['ball_land_x'], ball_positions['ball_land_y'], gridsize=40, cmap='YlOrRd', mincnt=1)
axes[1, 0].set_xlim(0, 120)
axes[1, 0].set_ylim(0, 53.3)
axes[1, 0].set_title('Ball Landing Heatmap', fontsize=12, fontweight='bold')
axes[1, 0].set_aspect('equal')
axes[1, 1].hist(play_info['pass_distance'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1, 1].axvline(play_info['pass_distance'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {play_info["pass_distance"].mean():.1f}')
axes[1, 1].set_title('Pass Distance', fontsize=12, fontweight='bold')
axes[1, 1].legend()
plt.tight_layout()
plt.savefig('eda_outputs/plots/06_ball_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 7. MOVEMENT PATTERNS
# =============================================================================
print("\n7. ANALYZING MOVEMENT PATTERNS...")

last_input = df_input.groupby(['game_id', 'play_id', 'nfl_id']).last()[['x', 'y']]
first_output = df_output.groupby(['game_id', 'play_id', 'nfl_id']).first()[['x', 'y']]
transition = last_input.join(first_output, how='inner', rsuffix='_out')
transition['displacement'] = np.sqrt((transition['x_out'] - transition['x'])**2 + (transition['y_out'] - transition['y'])**2)

output_first = df_output.groupby(['game_id', 'play_id', 'nfl_id']).first()[['x', 'y']]
output_last = df_output.groupby(['game_id', 'play_id', 'nfl_id']).last()[['x', 'y']]
total_movement = output_first.join(output_last, how='inner', rsuffix='_end')
total_movement['total_displacement'] = np.sqrt((total_movement['x_end'] - total_movement['x'])**2 +
                                               (total_movement['y_end'] - total_movement['y'])**2)

eda_results['movement_patterns'] = {
    'frame_to_frame_displacement': {
        'mean': float(transition['displacement'].mean()),
        'std': float(transition['displacement'].std())
    },
    'total_displacement_during_flight': {
        'mean': float(total_movement['total_displacement'].mean()),
        'std': float(total_movement['total_displacement'].std())
    }
}

print(f"   Avg frame-to-frame displacement: {transition['displacement'].mean():.3f} yards")
print(f"   Avg total displacement: {total_movement['total_displacement'].mean():.2f} yards")

# Plot 7: Displacement
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(transition['displacement'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(transition['displacement'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {transition["displacement"].mean():.3f}')
axes[0].set_title('Frame-to-Frame Displacement', fontsize=12, fontweight='bold')
axes[0].legend()
axes[1].hist(total_movement['total_displacement'], bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(total_movement['total_displacement'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {total_movement["total_displacement"].mean():.2f}')
axes[1].set_title('Total Displacement During Ball Flight', fontsize=12, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig('eda_outputs/plots/07_displacement_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 8. CORRELATION ANALYSIS
# =============================================================================
print("\n8. COMPUTING CORRELATIONS...")

numerical_features = ['x', 'y', 's', 'a', 'o', 'dir', 'num_frames_output', 'ball_land_x', 'ball_land_y']
sample_corr = df_input[numerical_features].dropna().sample(min(50000, len(df_input)), random_state=42)
corr_matrix = sample_corr.corr()

eda_results['correlations']['feature_correlation_matrix'] = {
    k: {k2: float(v2) for k2, v2 in v.items()}
    for k, v in corr_matrix.to_dict().items()
}

# Plot 8: Correlation matrix
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_outputs/plots/08_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 9. FIELD VISUALIZATION
# =============================================================================
print("\n9. CREATING FIELD VISUALIZATIONS...")

# Position heatmap
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
sample_viz = df_input.sample(min(50000, len(df_input)), random_state=42)
offense_data = sample_viz[sample_viz['player_side'] == 'Offense']
defense_data = sample_viz[sample_viz['player_side'] == 'Defense']

axes[0].hexbin(offense_data['x'], offense_data['y'], gridsize=50, cmap='Blues', mincnt=1)
axes[0].set_xlim(0, 120)
axes[0].set_ylim(0, 53.3)
axes[0].set_title('Offensive Player Positions', fontsize=14, fontweight='bold')
axes[0].set_aspect('equal')

axes[1].hexbin(defense_data['x'], defense_data['y'], gridsize=50, cmap='Reds', mincnt=1)
axes[1].set_xlim(0, 120)
axes[1].set_ylim(0, 53.3)
axes[1].set_title('Defensive Player Positions', fontsize=14, fontweight='bold')
axes[1].set_aspect('equal')
plt.tight_layout()
plt.savefig('eda_outputs/plots/09_position_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 10. SUMMARY
# =============================================================================
print("\n10. GENERATING SUMMARY...")

summary = {
    'dataset_overview': {
        'total_input_records': int(len(df_input)),
        'total_output_records': int(len(df_output)),
        'unique_games': int(unique_games),
        'unique_plays': int(unique_plays),
        'unique_players': int(unique_players),
        'weeks_covered': len(input_files)
    },
    'key_statistics': {
        'avg_players_per_play': float(players_per_play.mean()),
        'avg_output_frames': float(output_frames.mean()),
        'avg_ball_time_in_air_sec': float(time_in_air.mean()),
        'avg_speed_yards_per_sec': float(df_input['s'].mean()),
        'avg_pass_distance_yards': float(play_info['pass_distance'].mean()),
        'avg_displacement_per_frame': float(transition['displacement'].mean()),
        'avg_total_displacement': float(total_movement['total_displacement'].mean())
    }
}

eda_results['summary'] = summary

# Save results
output_file = 'eda_outputs/eda_results.json'
with open(output_file, 'w') as f:
    json.dump(eda_results, f, indent=2)

print(f"\n{'='*80}")
print("EDA COMPLETE!")
print(f"{'='*80}")
print(f"Results saved to: {output_file}")
print(f"Plots saved to: eda_outputs/plots/ ({len(list(Path('eda_outputs/plots').glob('*.png')))} images)")
print(f"Completed at: {datetime.now()}")
print(f"\n{'='*80}")
print("SUMMARY STATISTICS:")
print(f"{'='*80}")
for key, val in summary['key_statistics'].items():
    print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")
