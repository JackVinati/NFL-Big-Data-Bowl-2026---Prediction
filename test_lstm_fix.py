"""Quick test for LSTM model fix"""
import torch
import sys
sys.path.append('.')

from models import create_model

print("Testing LSTM model with fixed hidden states...")

# Test parameters
batch_size = 4
seq_len = 20
num_features = 26
num_frames = 11
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model
lstm_model = create_model('lstm', input_size=26).to(device)

# Create dummy input
dummy_input = torch.randn(batch_size, seq_len, num_features).to(device)

# Forward pass
try:
    with torch.no_grad():
        output = lstm_model(dummy_input, num_frames)

    print(f"✓ Success!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: ({batch_size}, {num_frames}, 2)")

    assert output.shape == (batch_size, num_frames, 2), f"Shape mismatch: {output.shape}"
    print(f"\n✓ All tests passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
