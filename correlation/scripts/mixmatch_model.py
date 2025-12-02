#!/usr/bin/env python3
"""
MixMatch Drift Classifier Model
Based on the PoPETs 2024.2 paper implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimplifiedDriftModel(nn.Module):
    """
    Simplified version of the MixMatch drift model for traffic correlation.
    Based on the PoPETs 2024.2 paper implementation.
    """
    
    def __init__(self, kernel_size=8, pool_size=8, sequence_length=500):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.sequence_length = sequence_length
        
        # Convolutional layers for drift detection
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 4, kernel_size, padding='same'),
            nn.Conv1d(4, 8, kernel_size, padding='same'), 
            nn.Conv1d(8, 16, kernel_size, padding='same')
        ])
        
        # Pooling layers
        self.pool_layers = nn.ModuleList([
            nn.AvgPool1d(pool_size, stride=2),
            nn.AvgPool1d(pool_size, stride=2),
            nn.AvgPool1d(pool_size, stride=2)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Will be initialized during first forward pass
        self.fc_layers = None
        self.flattened_size = None

    def _calculate_conv_output_size(self):
        """Calculate the output size after conv and pooling layers."""
        size = self.sequence_length
        for _ in range(len(self.conv_layers)):
            # After convolution (same padding)
            # After pooling
            size = size // 2
        return size * 16  # 16 is the number of channels after last conv layer

    def _initialize_fc_layers(self, flattened_size):
        """Initialize fully connected layers based on flattened size."""
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, inflow, outflow):
        """Compute drift score from inflow and outflow traffic."""
        
        # Convert to tensors if needed
        if not isinstance(inflow, torch.Tensor):
            inflow = torch.tensor(inflow, dtype=torch.float32)
        if not isinstance(outflow, torch.Tensor):
            outflow = torch.tensor(outflow, dtype=torch.float32)
        
        # Ensure tensors are on the same device as the model
        device = next(self.parameters()).device
        inflow = inflow.to(device)
        outflow = outflow.to(device)
        
        # Calculate drift score (difference between outflow and inflow)
        drift_score = outflow - inflow
        
        # Ensure correct shape for Conv1d (batch_size, channels, sequence_length)
        if drift_score.dim() == 2:
            drift_score = drift_score.unsqueeze(1)  # Add channel dimension
        elif drift_score.dim() == 1:
            drift_score = drift_score.unsqueeze(0).unsqueeze(1)  # Add batch and channel dims
        
        # Apply conv + pool layers with dropout
        for i, (conv_layer, pool_layer) in enumerate(zip(self.conv_layers, self.pool_layers)):
            drift_score = F.relu(conv_layer(drift_score))
            drift_score = pool_layer(drift_score)
            if i > 0:  # Add dropout after first layer
                drift_score = self.dropout(drift_score)
        
        # Flatten
        drift_score = drift_score.view(drift_score.size(0), -1)
        
        # Initialize FC layers if not done yet
        if self.fc_layers is None:
            self.flattened_size = drift_score.size(1)
            self._initialize_fc_layers(self.flattened_size)
            self.fc_layers = self.fc_layers.to(device)
        
        # Apply fully connected layers
        drift_score = self.fc_layers(drift_score)
        
        return drift_score

    def get_model_info(self):
        """Return model configuration information."""
        return {
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'sequence_length': self.sequence_length,
            'num_conv_layers': len(self.conv_layers),
            'flattened_size': self.flattened_size
        }


class MixMatchDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MixMatch training data.
    """
    
    def __init__(self, data_pairs):
        """
        Initialize dataset with data pairs.
        
        Args:
            data_pairs: List of tuples (proxy_seq, requester_seq, label, metadata)
        """
        self.data_pairs = data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        proxy_seq, requester_seq, label, _ = self.data_pairs[idx]
        
        # Convert to tensors
        proxy_tensor = torch.from_numpy(proxy_seq).float()
        requester_tensor = torch.from_numpy(requester_seq).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return proxy_tensor, requester_tensor, label_tensor


def create_model(sequence_length=500, kernel_size=8, pool_size=8):
    """
    Create a new MixMatch drift classifier model.
    
    Args:
        sequence_length: Length of input sequences
        kernel_size: Convolution kernel size
        pool_size: Pooling size
    
    Returns:
        SimplifiedDriftModel instance
    """
    model = SimplifiedDriftModel(
        kernel_size=kernel_size,
        pool_size=pool_size,
        sequence_length=sequence_length
    )
    return model


def save_model(model, filepath, metadata=None):
    """
    Save model state and metadata.
    
    Args:
        model: The trained model
        filepath: Path to save the model
        metadata: Additional metadata to save
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': model.get_model_info(),
        'metadata': metadata or {}
    }
    
    torch.save(save_dict, filepath)
    print(f"✅ Model saved to {filepath}")


def load_model(filepath, device='cpu'):
    """
    Load a saved model.
    
    Args:
        filepath: Path to the saved model
        device: Device to load the model on
    
    Returns:
        Loaded model and metadata
    """
    import warnings
    
    try:
        # Try with weights_only=False for compatibility
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except Exception as e:
        # Fallback for older PyTorch versions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(filepath, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    sequence_length = model_config.get('sequence_length', 500)
    kernel_size = model_config.get('kernel_size', 8)
    pool_size = model_config.get('pool_size', 8)
    
    # Create model with same configuration
    model = SimplifiedDriftModel(
        kernel_size=kernel_size,
        pool_size=pool_size,
        sequence_length=sequence_length
    )
    
    # Set flattened_size if available (for proper initialization)
    flattened_size = model_config.get('flattened_size', None)
    if flattened_size is not None:
        model.flattened_size = flattened_size
        # Initialize fc_layers with the correct size
        model._initialize_fc_layers(flattened_size)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    metadata = checkpoint.get('metadata', {})
    
    print(f"✅ Model loaded from {filepath}")
    return model, metadata


if __name__ == "__main__":
    # Test model creation
    print("🧪 Testing MixMatch model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model()
    model.to(device)
    
    # Test with dummy data
    batch_size = 4
    sequence_length = 500
    
    dummy_proxy = torch.randn(batch_size, sequence_length).to(device)
    dummy_requester = torch.randn(batch_size, sequence_length).to(device)
    
    # Forward pass
    output = model(dummy_proxy, dummy_requester)
    
    print(f"✅ Model test successful!")
    print(f"   Input shape: {dummy_proxy.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Model info: {model.get_model_info()}")
