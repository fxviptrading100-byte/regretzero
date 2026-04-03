import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class RegretZeroModel(nn.Module):
    """
    PyTorch model for RegretZero environment.
    
    This model takes decision context as input and outputs predicted regret score.
    Simple MLP architecture with 2 hidden layers, ReLU, and Dropout.
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super(RegretZeroModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple MLP with 2 hidden layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),  # Single output for regret score
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input decision context (batch_size, input_dim)
            
        Returns:
            regret_score: Predicted regret score (batch_size, 1)
        """
        return self.network(x)
    
    def predict_regret(self, decision_context: torch.Tensor) -> float:
        """
        Predict regret score for a single decision context.
        
        Args:
            decision_context: Single decision context (input_dim,)
            
        Returns:
            regret_score: Predicted regret score (float)
        """
        self.eval()
        with torch.no_grad():
            if decision_context.dim() == 1:
                decision_context = decision_context.unsqueeze(0)
            regret_score = self.forward(decision_context)
            return regret_score.item()
    
    def get_feature_importance(self, decision_context: torch.Tensor) -> torch.Tensor:
        """
        Get feature importance using gradient-based approach.
        
        Args:
            decision_context: Single decision context (input_dim,)
            
        Returns:
            importance: Feature importance scores (input_dim,)
        """
        self.eval()
        decision_context = decision_context.unsqueeze(0).requires_grad_(True)
        
        output = self.forward(decision_context)
        output.backward()
        
        importance = torch.abs(decision_context.grad).squeeze(0)
        return importance / importance.sum()  # Normalize to sum to 1


def create_model(input_dim: int = 32, hidden_dim: int = 64, dropout_rate: float = 0.2) -> RegretZeroModel:
    """Create and initialize a RegretZero model."""
    model = RegretZeroModel(input_dim, hidden_dim, dropout_rate)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    return model


def save_model(model: RegretZeroModel, filepath: str):
    """Save model to file."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim
        }
    }, filepath)


def load_model(filepath: str) -> RegretZeroModel:
    """Load model from file."""
    checkpoint = torch.load(filepath, map_location='cpu')
    config = checkpoint['model_config']
    
    model = RegretZeroModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_model()
    
    # Test forward pass
    batch_input = torch.randn(32, 32)  # batch of 32, input dim 32
    output = model(batch_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Model output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test single prediction
    single_input = torch.randn(32)
    regret_score = model.predict_regret(single_input)
    print(f"Single prediction regret score: {regret_score:.3f}")
    
    # Test feature importance
    importance = model.get_feature_importance(single_input)
    print(f"Feature importance shape: {importance.shape}")
    print(f"Top 5 important features: {importance.topk(5)}")
