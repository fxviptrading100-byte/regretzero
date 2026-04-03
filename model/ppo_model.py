import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with shared backbone for PPO.
    
    This network takes 64-dimensional observations from RegretZero environment
    and outputs both action probabilities (actor) and state values (critic).
    """
    
    def __init__(
        self, 
        obs_dim: int = 64, 
        action_dim: int = 8, 
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        """
        Initialize Actor-Critic network.
        
        Args:
            obs_dim: Observation space dimension (64 for RegretZero)
            action_dim: Action space dimension (8 for RegretZero)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers in shared backbone
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared backbone layers
        backbone_layers = []
        input_dim = obs_dim
        
        for i in range(num_layers):
            backbone_layers.extend([
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = self.hidden_dim
            
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Actor head (policy network)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value network)
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs: Input observations [batch_size, obs_dim]
            
        Returns:
            action_logits: Action logits [batch_size, action_dim]
            value: State values [batch_size, 1]
        """
        # Shared backbone
        shared_features = self.backbone(obs)
        
        # Actor and critic heads
        action_logits = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        
        return action_logits, value
    
    def get_action_and_value(
        self, 
        obs: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, value, log probability, and entropy.
        
        Args:
            obs: Input observations [batch_size, obs_dim]
            action: Optional actions to evaluate [batch_size]
            
        Returns:
            action: Sampled actions [batch_size]
            value: State values [batch_size, 1]
            log_prob: Log probabilities of actions [batch_size]
            entropy: Distribution entropy [batch_size]
        """
        action_logits, value = self.forward(obs)
        
        # Create categorical distribution
        probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, value, log_prob, entropy
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value for given observation."""
        _, value = self.forward(obs)
        return value
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Actions to evaluate [batch_size]
            
        Returns:
            log_prob: Log probabilities of actions [batch_size]
            entropy: Distribution entropy [batch_size]
            value: State values [batch_size, 1]
        """
        action_logits, value = self.forward(obs)
        
        # Create categorical distribution
        probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, entropy, value


class PPOAgent:
    """
    Proximal Policy Optimization agent with Actor-Critic architecture.
    
    This class implements the PPO algorithm with clipped surrogate objective,
    Generalized Advantage Estimation (GAE), and value function clipping.
    """
    
    def __init__(
        self,
        obs_dim: int = 64,
        action_dim: int = 8,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize PPO agent.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value function coefficient
            entropy_coef: Entropy regularization coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use ("auto", "cpu", "cuda")
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize networks and optimizer
        self.network = ActorCriticNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # Training statistics
        self.update_count = 0
        
    def select_action(self, obs: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """
        Select action using current policy.
        
        Args:
            obs: Single observation [obs_dim]
            
        Returns:
            action: Selected action
            info: Dictionary with additional information
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value, log_prob, entropy = self.network.get_action_and_value(obs_tensor)
        
        action = action.cpu().numpy()[0]
        value = value.cpu().numpy()[0, 0]
        log_prob = log_prob.cpu().numpy()[0]
        entropy = entropy.cpu().numpy()[0]
        
        info = {
            "value": value,
            "log_prob": log_prob,
            "entropy": entropy
        }
        
        return action, info
    
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of episode termination flags
            next_value: Value of next state (0 for terminal states)
            
        Returns:
            advantages: List of advantage estimates
            returns: List of return estimates
        """
        advantages = []
        returns = []
        
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value_step * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return advantages, returns
    
    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Actions taken [batch_size]
            old_log_probs: Old log probabilities [batch_size]
            advantages: Advantage estimates [batch_size]
            returns: Return estimates [batch_size]
            epochs: Number of update epochs
            batch_size: Mini-batch size
            
        Returns:
            stats: Dictionary with training statistics
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # Mini-batch updates
        batch_size = min(batch_size, len(obs))
        for epoch in range(epochs):
            # Shuffle indices for each epoch
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Evaluate current policy
                new_log_probs, entropy, values = self.network.evaluate_actions(batch_obs, batch_actions)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        self.update_count += 1
        
        # Compute statistics
        num_updates = (len(obs) // batch_size) * epochs
        stats = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "update_count": self.update_count
        }
        
        return stats
    
    def save(self, filepath: str):
        """Save model state."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
                "update_count": self.update_count
            }
        }, filepath)
    
    def load(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        config = checkpoint["config"]
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_epsilon = config["clip_epsilon"]
        self.value_coef = config["value_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.update_count = config["update_count"]
