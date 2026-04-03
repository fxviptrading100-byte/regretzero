#!/usr/bin/env python3
"""
Training script for RegretZero model.

This script trains the neural network model on simulated decision scenarios
to predict regret scores and learn optimal decision-making patterns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.regret_model import create_model, save_model
from env.regret_env import RegretZeroEnv


class RegretTrainer:
    """
    Trainer class for the RegretZero model.
    """
    
    def __init__(self, 
                 model_path: str = "model/regret_model.pt",
                 input_dim: int = 32,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100):
        """
        Initialize the trainer.
        
        Args:
            model_path: Path to save the trained model
            input_dim: Input dimension for the model
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.model_path = model_path
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Create model and optimizer
        self.model = create_model(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Training on device: {self.device}")
    
    def generate_training_data(self, num_episodes: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data using the RegretZero environment.
        
        Args:
            num_episodes: Number of episodes to simulate
            
        Returns:
            observations: Array of decision contexts
            targets: Array of regret scores
        """
        print(f"Generating training data from {num_episodes} episodes...")
        
        observations = []
        targets = []
        
        env = RegretZeroEnv(max_steps=30)
        
        for episode in tqdm(range(num_episodes), desc="Generating data"):
            obs, info = env.reset()
            episode_regret = 0.0
            
            for step in range(env.max_steps):
                # Store current observation
                observations.append(obs.copy())
                
                # Take random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Calculate cumulative regret for this step
                episode_regret += info.get('current_regret', 0)
                targets.append(episode_regret)
                
                if terminated or truncated:
                    break
        
        return np.array(observations, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def generate_validation_data(self, num_episodes: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate validation data.
        
        Args:
            num_episodes: Number of validation episodes
            
        Returns:
            observations: Array of decision contexts
            targets: Array of regret scores
        """
        print(f"Generating validation data from {num_episodes} episodes...")
        
        observations = []
        targets = []
        
        env = RegretZeroEnv(max_steps=30)
        
        for episode in tqdm(range(num_episodes), desc="Generating validation data"):
            obs, info = env.reset()
            episode_regret = 0.0
            
            for step in range(env.max_steps):
                observations.append(obs.copy())
                
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_regret += info.get('current_regret', 0)
                targets.append(episode_regret)
                
                if terminated or truncated:
                    break
        
        return np.array(observations, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def train_epoch(self, train_obs: np.ndarray, train_targets: np.ndarray) -> float:
        """
        Train for one epoch.
        
        Args:
            train_obs: Training observations
            train_targets: Training targets
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        
        # Shuffle data
        indices = np.random.permutation(len(train_obs))
        train_obs = train_obs[indices]
        train_targets = train_targets[indices]
        
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_obs), self.batch_size):
            batch_obs = train_obs[i:i+self.batch_size]
            batch_targets = train_targets[i:i+self.batch_size]
            
            # Convert to tensors
            batch_obs_tensor = torch.FloatTensor(batch_obs).to(self.device)
            batch_targets_tensor = torch.FloatTensor(batch_targets).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_obs_tensor).squeeze()
            loss = self.criterion(predictions, batch_targets_tensor)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_obs: np.ndarray, val_targets: np.ndarray) -> float:
        """
        Validate the model.
        
        Args:
            val_obs: Validation observations
            val_targets: Validation targets
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_obs), self.batch_size):
                batch_obs = val_obs[i:i+self.batch_size]
                batch_targets = val_targets[i:i+self.batch_size]
                
                batch_obs_tensor = torch.FloatTensor(batch_obs).to(self.device)
                batch_targets_tensor = torch.FloatTensor(batch_targets).to(self.device)
                
                predictions = self.model(batch_obs_tensor).squeeze()
                loss = self.criterion(predictions, batch_targets_tensor)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """
        Main training loop.
        """
        print("Starting training...")
        
        # Generate training and validation data
        train_obs, train_targets = self.generate_training_data(num_episodes=1000)
        val_obs, val_targets = self.generate_validation_data(num_episodes=200)
        
        # Normalize targets
        target_mean = np.mean(train_targets)
        target_std = np.std(train_targets)
        train_targets_normalized = (train_targets - target_mean) / (target_std + 1e-8)
        val_targets_normalized = (val_targets - target_mean) / (target_std + 1e-8)
        
        print(f"Training data: {len(train_obs)} samples")
        print(f"Validation data: {len(val_obs)} samples")
        print(f"Target stats: mean={target_mean:.3f}, std={target_std:.3f}")
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_obs, train_targets_normalized)
            
            # Validate
            val_loss = self.validate(val_obs, val_targets_normalized)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                print(f"New best model saved at epoch {epoch}")
        
        print("Training completed!")
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_model(self):
        """Save the trained model."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model with normalization parameters
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim
            },
            'training_info': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'epochs': len(self.train_losses)
            }
        }
        
        torch.save(checkpoint, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Training Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_path = self.model_path.replace('.pt', '_training_curves.png')
            plt.savefig(plot_path)
            print(f"Training curves saved to {plot_path}")
            plt.close()
            
        except ImportError:
            print("Matplotlib not available, skipping plot")
    
    def load_model(self) -> bool:
        """
        Load a trained model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            print(f"No trained model found at {self.model_path}")
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def main():
    """Main training function."""
    print("=== RegretZero Model Training ===")
    
    # Create trainer
    trainer = RegretTrainer(
        model_path="model/regret_model.pt",
        input_dim=32,
        hidden_dim=64,
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )
    
    # Check if model already exists
    if trainer.load_model():
        print("Model already exists. Skipping training.")
        print("To retrain, delete the existing model file.")
        return
    
    # Train the model
    trainer.train()
    
    # Test the trained model
    print("\n=== Testing Trained Model ===")
    
    # Create a test environment
    test_env = RegretZeroEnv(max_steps=10)
    obs, info = test_env.reset()
    
    # Make predictions
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
    
    with torch.no_grad():
        prediction = trainer.model(obs_tensor)
        print(f"Test prediction: {prediction.item():.4f}")
    
    print("Training and testing completed!")


if __name__ == "__main__":
    main()
