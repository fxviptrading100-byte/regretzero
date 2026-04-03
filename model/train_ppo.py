import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import argparse
import json
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.regret_env import RegretZeroEnv
from model.ppo_model import PPOAgent


class PPOTrainer:
    """
    PPO trainer for RegretZero environment.
    
    This class handles the complete training pipeline including data collection,
    advantage computation, and policy updates.
    """
    
    def __init__(
        self,
        env: RegretZeroEnv,
        agent: PPOAgent,
        save_dir: str = "model",
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 200
    ):
        """
        Initialize PPO trainer.
        
        Args:
            env: RegretZero environment
            agent: PPO agent
            save_dir: Directory to save models and logs
            log_interval: Interval for logging training progress
            eval_interval: Interval for evaluation
            save_interval: Interval for saving models
        """
        self.env = env
        self.agent = agent
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Training parameters
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Training statistics
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_regrets": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "eval_rewards": [],
            "eval_regrets": []
        }
        
        # Best model tracking
        self.best_eval_reward = float("-inf")
        self.best_model_path = os.path.join(save_dir, "regret_ppo_best.pt")
        self.latest_model_path = os.path.join(save_dir, "regret_ppo_latest.pt")
    
    def collect_rollouts(
        self, 
        num_episodes: int, 
        max_steps_per_episode: int = 50
    ) -> Tuple[Dict[str, List], Dict[str, float]]:
        """
        Collect rollouts for training.
        
        Args:
            num_episodes: Number of episodes to collect
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            rollout_data: Dictionary with collected data
            episode_stats: Dictionary with episode statistics
        """
        rollout_data = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": []
        }
        
        episode_rewards = []
        episode_lengths = []
        episode_regrets = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_regret = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action, info = self.agent.select_action(obs)
                value = info["value"]
                log_prob = info["log_prob"]
                
                # Store data
                rollout_data["obs"].append(obs.copy())
                rollout_data["actions"].append(action)
                rollout_data["values"].append(value)
                rollout_data["log_probs"].append(log_prob)
                
                # Take step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                rollout_data["rewards"].append(reward)
                rollout_data["dones"].append(done)
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                episode_regret += info.get("current_regret", 0)
                
                obs = next_obs
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_regrets.append(episode_regret)
        
        episode_stats = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_regret": np.mean(episode_regrets)
        }
        
        return rollout_data, episode_stats
    
    def compute_advantages_and_returns(
        self, 
        rollout_data: Dict[str, List]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute advantages and returns from rollout data.
        
        Args:
            rollout_data: Dictionary with collected rollout data
            
        Returns:
            obs: Observations array
            actions: Actions array
            old_log_probs: Old log probabilities array
            advantages: Advantages array
            returns: Returns array
        """
        obs = np.array(rollout_data["obs"])
        actions = np.array(rollout_data["actions"])
        old_log_probs = np.array(rollout_data["log_probs"])
        rewards = rollout_data["rewards"]
        values = rollout_data["values"]
        dones = rollout_data["dones"]
        
        # Compute next value for GAE (0 for terminal states)
        next_value = 0.0
        
        # Compute GAE advantages and returns
        advantages, returns = self.agent.compute_gae(rewards, values, dones, next_value)
        
        return obs, actions, old_log_probs, np.array(advantages), np.array(returns)
    
    def evaluate(self, num_episodes: int = 10, max_steps_per_episode: int = 50) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            eval_stats: Dictionary with evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []
        eval_regrets = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_regret = 0
            
            for step in range(max_steps_per_episode):
                # Select action (deterministic for evaluation)
                action, _ = self.agent.select_action(obs)
                
                # Take step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                episode_regret += info.get("current_regret", 0)
                
                obs = next_obs
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_regrets.append(episode_regret)
        
        eval_stats = {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "mean_regret": np.mean(eval_regrets)
        }
        
        return eval_stats
    
    def train(
        self,
        total_timesteps: int = 500000,
        episodes_per_rollout: int = 16,
        update_epochs: int = 10,
        batch_size: int = 64,
        max_steps_per_episode: int = 50
    ):
        """
        Main training loop with enhanced stability and progress tracking.
        
        Args:
            total_timesteps: Total number of environment timesteps
            episodes_per_rollout: Number of episodes per rollout collection
            update_epochs: Number of PPO update epochs per rollout
            batch_size: Mini-batch size for PPO updates
            max_steps_per_episode: Maximum steps per episode
        """
        print("\nStarting enhanced PPO training for RegretZero...")
        print(f"Target timesteps: {total_timesteps:,}")
        print(f"Episodes per rollout: {episodes_per_rollout}")
        print(f"Update epochs: {update_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.agent.device}")
        print(f"Save directory: {self.save_dir}")
        print("-" * 60)
        
        timesteps_collected = 0
        update_count = 0
        episode_count = 0
        
        # Training progress tracking
        best_regret_so_far = float('inf')
        convergence_patience = 0
        max_patience = 500  # Stop if no improvement for 500 updates
        
        with tqdm(total=total_timesteps, desc="Training PPO", unit="timesteps") as pbar:
            while timesteps_collected < total_timesteps:
                # Collect rollouts with progress tracking
                rollout_data, episode_stats = self.collect_rollouts(
                    episodes_per_rollout, max_steps_per_episode
                )
                
                # Update counters
                rollout_timesteps = sum(len(r) for r in rollout_data["rewards"])
                timesteps_collected += rollout_timesteps
                episode_count += episodes_per_rollout
                update_count += 1
                
                # Store episode statistics
                self.training_stats["episode_rewards"].extend(
                    [episode_stats["mean_reward"]] * episodes_per_rollout
                )
                self.training_stats["episode_lengths"].extend(
                    [episode_stats["mean_length"]] * episodes_per_rollout
                )
                self.training_stats["episode_regrets"].extend(
                    [episode_stats["mean_regret"]] * episodes_per_rollout
                )
                
                # Compute advantages and returns
                obs, actions, old_log_probs, advantages, returns = self.compute_advantages_and_returns(
                    rollout_data
                )
                
                # Update policy with stability checks
                try:
                    update_stats = self.agent.update(
                        obs, actions, old_log_probs, advantages, returns,
                        epochs=update_epochs, batch_size=batch_size
                    )
                    
                    # Store update statistics
                    self.training_stats["policy_losses"].append(update_stats["policy_loss"])
                    self.training_stats["value_losses"].append(update_stats["value_loss"])
                    self.training_stats["entropy_losses"].append(update_stats["entropy_loss"])
                    
                except Exception as e:
                    print(f"\nWarning: Update failed at update {update_count}: {e}")
                    continue
                
                # Enhanced logging with convergence tracking
                if update_count % self.log_interval == 0:
                    recent_rewards = self.training_stats["episode_rewards"][-100:]
                    recent_regrets = self.training_stats["episode_regrets"][-100:]
                    recent_lengths = self.training_stats["episode_lengths"][-100:]
                    
                    avg_reward = np.mean(recent_rewards)
                    avg_regret = np.mean(recent_regrets)
                    avg_length = np.mean(recent_lengths)
                    
                    # Check for improvement
                    if avg_regret < best_regret_so_far:
                        best_regret_so_far = avg_regret
                        convergence_patience = 0
                    else:
                        convergence_patience += 1
                    
                    pbar.set_postfix({
                        "Episodes": f"{episode_count:,}",
                        "Reward": f"{avg_reward:.3f}",
                        "Regret": f"{avg_regret:.3f}",
                        "Length": f"{avg_length:.1f}",
                        "Updates": update_count,
                        "BestRegret": f"{best_regret_so_far:.3f}"
                    })
                
                # Evaluation with enhanced reporting
                if update_count % self.eval_interval == 0:
                    eval_stats = self.evaluate(num_episodes=20, max_steps_per_episode=max_steps_per_episode)
                    self.training_stats["eval_rewards"].append(eval_stats["mean_reward"])
                    self.training_stats["eval_regrets"].append(eval_stats["mean_regret"])
                    
                    print(f"\n{'='*50}")
                    print(f"EVALUATION AT UPDATE {update_count} (Episodes: {episode_count:,})")
                    print(f"{'='*50}")
                    print(f"  Mean Reward: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
                    print(f"  Mean Regret: {eval_stats['mean_regret']:.4f}")
                    print(f"  Mean Length: {eval_stats['mean_length']:.2f}")
                    print(f"  Best Training Regret: {best_regret_so_far:.4f}")
                    
                    # Save best model based on regret (lower is better)
                    if eval_stats["mean_regret"] < best_regret_so_far * 0.95:  # 5% improvement threshold
                        self.best_eval_reward = eval_stats["mean_reward"]
                        self.agent.save(self.best_model_path)
                        print(f"  🎉 NEW BEST MODEL SAVED! (Regret: {eval_stats['mean_regret']:.4f})")
                    else:
                        print(f"  No improvement. Best regret: {best_regret_so_far:.4f}")
                    
                    # Early stopping check
                    if convergence_patience > max_patience:
                        print(f"\n⚠️  EARLY STOPPING: No improvement for {max_patience} updates")
                        print(f"   Best regret achieved: {best_regret_so_far:.4f}")
                        break
                
                # Save latest model
                if update_count % self.save_interval == 0:
                    self.agent.save(self.latest_model_path)
                    print(f"\n💾 Latest model saved at update {update_count} (Episodes: {episode_count:,})")
                
                pbar.update(rollout_timesteps)
        
        # Final save and summary
        self.agent.save(self.latest_model_path)
        print(f"\n🏁 Training completed! Episodes trained: {episode_count:,}")
        print(f"   Final model saved to: {self.latest_model_path}")
        print(f"   Best regret achieved: {best_regret_so_far:.4f}")
        
        # Save training statistics
        self.save_training_stats()
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_training_stats(self):
        """Save training statistics to JSON file."""
        stats_path = os.path.join(self.save_dir, "training_stats.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {}
        for key, values in self.training_stats.items():
            if isinstance(values, list) and len(values) > 0 and isinstance(values[0], np.ndarray):
                json_stats[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
            else:
                json_stats[key] = values
        
        with open(stats_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"Training statistics saved to {stats_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        if self.training_stats["episode_rewards"]:
            axes[0, 0].plot(self.training_stats["episode_rewards"])
            axes[0, 0].set_title("Episode Rewards")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].grid(True)
            
            # Moving average
            if len(self.training_stats["episode_rewards"]) > 100:
                moving_avg = np.convolve(
                    self.training_stats["episode_rewards"], 
                    np.ones(100)/100, 
                    mode='valid'
                )
                axes[0, 0].plot(range(99, len(self.training_stats["episode_rewards"])), moving_avg, 
                              'r-', linewidth=2, label='100-episode MA')
                axes[0, 0].legend()
        
        # Episode regrets
        if self.training_stats["episode_regrets"]:
            axes[0, 1].plot(self.training_stats["episode_regrets"])
            axes[0, 1].set_title("Episode Regrets")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Regret")
            axes[0, 1].grid(True)
        
        # Episode lengths
        if self.training_stats["episode_lengths"]:
            axes[0, 2].plot(self.training_stats["episode_lengths"])
            axes[0, 2].set_title("Episode Lengths")
            axes[0, 2].set_xlabel("Episode")
            axes[0, 2].set_ylabel("Length")
            axes[0, 2].grid(True)
        
        # Policy loss
        if self.training_stats["policy_losses"]:
            axes[1, 0].plot(self.training_stats["policy_losses"])
            axes[1, 0].set_title("Policy Loss")
            axes[1, 0].set_xlabel("Update")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].grid(True)
        
        # Value loss
        if self.training_stats["value_losses"]:
            axes[1, 1].plot(self.training_stats["value_losses"])
            axes[1, 1].set_title("Value Loss")
            axes[1, 1].set_xlabel("Update")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True)
        
        # Evaluation rewards
        if self.training_stats["eval_rewards"]:
            eval_x = np.arange(0, len(self.training_stats["eval_rewards"])) * self.eval_interval
            axes[1, 2].plot(eval_x, self.training_stats["eval_rewards"], 'o-')
            axes[1, 2].set_title("Evaluation Rewards")
            axes[1, 2].set_xlabel("Update")
            axes[1, 2].set_ylabel("Reward")
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO agent on RegretZero environment")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps (500K for ~8000 episodes)")
    parser.add_argument("--episodes_per_rollout", type=int, default=16, help="Episodes per rollout collection")
    parser.add_argument("--update_epochs", type=int, default=10, help="PPO update epochs per rollout")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for PPO updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for larger network")
    parser.add_argument("--save_dir", type=str, default="model", help="Save directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_episodes", type=int, default=8000, help="Maximum number of episodes to train")
    
    args = parser.parse_args()
    
    print("=== PPO TRAINING FOR REGRETZERO ===")
    print(f"Target timesteps: {args.timesteps:,}")
    print(f"Target episodes: {args.max_episodes:,}")
    print(f"Episodes per rollout: {args.episodes_per_rollout}")
    print(f"Update epochs: {args.update_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    print("="*50)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = RegretZeroEnv(max_steps=50, seed=args.seed)
    
    # Create PPO agent
    agent = PPOAgent(
        obs_dim=64,
        action_dim=8,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=args.device
    )
    
    # Create trainer with enhanced logging
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        save_dir=args.save_dir,
        log_interval=25,  # Log every 25 updates
        eval_interval=100,  # Evaluate every 100 updates
        save_interval=200   # Save every 200 updates
    )
    
    # Start training with episode limit
    trainer.train(
        total_timesteps=args.timesteps,
        episodes_per_rollout=args.episodes_per_rollout,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        max_steps_per_episode=50
    )
    
    # Final comprehensive evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model for final evaluation
    if os.path.exists(trainer.best_model_path):
        agent.load(trainer.best_model_path)
        print(f"Loaded best model from {trainer.best_model_path}")
    
    # Run comprehensive final evaluation
    final_eval_episodes = 100
    final_stats = trainer.evaluate(num_episodes=final_eval_episodes, max_steps_per_episode=50)
    
    print(f"\nFinal Results ({final_eval_episodes} episodes):")
    print(f"  Mean Reward: {final_stats['mean_reward']:.4f} ± {final_stats['std_reward']:.4f}")
    print(f"  Mean Regret: {final_stats['mean_regret']:.4f}")
    print(f"  Mean Episode Length: {final_stats['mean_length']:.2f}")
    print(f"  Best Training Reward: {trainer.best_eval_reward:.4f}")
    
    # Calculate additional metrics
    if len(trainer.training_stats["episode_regrets"]) > 0:
        final_avg_regret = np.mean(trainer.training_stats["episode_regrets"][-100:])
        improvement = ((trainer.training_stats["episode_regrets"][0] - final_avg_regret) / 
                      max(0.001, trainer.training_stats["episode_regrets"][0])) * 100
        print(f"  Regret Improvement: {improvement:.1f}%")
        print(f"  Final 100-episode Avg Regret: {final_avg_regret:.4f}")
    
    # Save final model as regret_ppo.pt (as requested)
    final_model_path = os.path.join(args.save_dir, "regret_ppo.pt")
    agent.save(final_model_path)
    print(f"\n🎯 Final model saved as: {final_model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Best model saved to: {trainer.best_model_path}")
    print(f"Latest model saved to: {trainer.latest_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training statistics saved to: {os.path.join(args.save_dir, 'training_stats.json')}")
    print(f"Training curves saved to: {os.path.join(args.save_dir, 'training_curves.png')}")
    print(f"\n🏆 FINAL AVERAGE REGRET SCORE: {final_stats['mean_regret']:.4f}")


if __name__ == "__main__":
    main()
