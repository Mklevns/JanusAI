# JanusAI/ml/training/advanced_ppo_trainer.py
"""
Advanced PPO trainer with clean separation of data collection and learning phases.
Supports distributed data collection, experience replay, and curriculum learning.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from dataclasses import dataclass
from collections import deque

from janus_ai.ml.base_trainer import BaseTrainer

from janus_ai.rollouts.rollout_buffer import RolloutBuffer

from janus_ai.utils.general_utils import safe_env_reset
from janus_ai.utils.io.checkpoint_manager import CheckpointManager


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    rollout_length: int = 2048
    use_gae: bool = True
    normalize_advantages: bool = True


class ExperienceBuffer:
    """Advanced experience buffer supporting replay and curriculum learning"""
    
    def __init__(self, max_size: int = 100000, prioritized: bool = False):
        self.max_size = max_size
        self.prioritized = prioritized
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size) if prioritized else None
        
    def add_trajectory(self, trajectory: Dict[str, Any], priority: float = 1.0):
        """Add a complete trajectory to the buffer"""
        self.buffer.append(trajectory)
        if self.prioritized:
            self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample trajectories from the buffer"""
        if self.prioritized:
            # Implement prioritized sampling
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), 
                                     p=probs, replace=True)
            return [self.buffer[i] for i in indices]
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), 
                                     replace=True)
            return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        if self.prioritized:
            self.priorities.clear()


class CurriculumManager:
    """Manages curriculum learning progression"""
    
    def __init__(self, difficulty_levels: List[Dict[str, Any]], 
                 progression_criterion: str = 'success_rate',
                 threshold: float = 0.8):
        self.difficulty_levels = difficulty_levels
        self.current_level = 0
        self.progression_criterion = progression_criterion
        self.threshold = threshold
        self.performance_history = deque(maxlen=100)
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current difficulty configuration"""
        return self.difficulty_levels[self.current_level]
    
    def update_performance(self, metrics: Dict[str, float]):
        """Update performance and check for progression"""
        self.performance_history.append(metrics.get(self.progression_criterion, 0.0))
        
        if len(self.performance_history) >= 50:  # Enough samples
            recent_performance = np.mean(list(self.performance_history)[-20:])
            if recent_performance >= self.threshold and self.current_level < len(self.difficulty_levels) - 1:
                self.current_level += 1
                logging.info(f"Curriculum progressed to level {self.current_level}")
                self.performance_history.clear()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get curriculum progress information"""
        return {
            'current_level': self.current_level,
            'total_levels': len(self.difficulty_levels),
            'recent_performance': np.mean(list(self.performance_history)[-10:]) if self.performance_history else 0.0
        }


class AdvancedPPOTrainer(BaseTrainer):
    """
    Advanced PPO trainer with clean separation of data collection and learning.
    Supports distributed collection, experience replay, and curriculum learning.
    """
    
    def __init__(self, 
                 policy: torch.nn.Module,
                 env: Any,
                 config: PPOConfig = None,
                 rollout_buffer: Optional[RolloutBuffer] = None,
                 experience_buffer: Optional[ExperienceBuffer] = None,
                 curriculum_manager: Optional[CurriculumManager] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None,
                 device: str = 'auto'):
        
        super().__init__()
        
        # Core components
        self.policy = policy
        self.env = env
        self.config = config or PPOConfig()
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.policy.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        
        # Buffers
        self.rollout_buffer = rollout_buffer or RolloutBuffer()
        self.experience_buffer = experience_buffer
        self.curriculum_manager = curriculum_manager
        
        # Checkpoint management
        self.checkpoint_manager = checkpoint_manager
        
        # Training state
        self.total_steps = 0
        self.episode_rewards = []
        self.training_metrics = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def collect_rollouts(self, 
                        n_steps: int,
                        env_config: Optional[Dict[str, Any]] = None,
                        parallel_envs: Optional[List[Any]] = None,
                        **kwargs) -> Dict[str, torch.Tensor]:
        """
        Collect experience rollouts with support for distributed collection.
        
        Args:
            n_steps: Number of steps to collect
            env_config: Environment configuration (for curriculum learning)
            parallel_envs: List of parallel environments for distributed collection
            **kwargs: Additional arguments (task_trajectories, ai_model_representation, etc.)
        """
        self.logger.debug(f"Collecting {n_steps} rollout steps")
        
        # Apply curriculum configuration if available
        if self.curriculum_manager and env_config:
            self._apply_env_config(env_config)
        
        # Decide whether to use distributed collection
        if parallel_envs:
            return self._collect_distributed_rollouts(n_steps, parallel_envs, **kwargs)
        else:
            return self._collect_single_env_rollouts(n_steps, **kwargs)
    
    def _collect_single_env_rollouts(self, n_steps: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Collect rollouts from a single environment"""
        
        self.rollout_buffer.reset()
        self.policy.eval()  # Set to eval mode for data collection
        
        # Reset environment
        obs, info = safe_env_reset(self.env)
        tree_structure = info.get('tree_structure') if isinstance(info, dict) else None
        
        episode_rewards = []
        current_episode_reward = 0
        
        for step in range(n_steps):
            # Prepare observation
            obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)
            
            # Get action mask
            action_mask_np = self.env.get_action_mask()
            action_mask_tensor = torch.BoolTensor(action_mask_np).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get action from policy
                action_val, log_prob_val, value_val = self.policy.get_action(
                    obs=obs_tensor,
                    action_mask=action_mask_tensor,
                    tree_structure=tree_structure,
                    **kwargs  # Pass any additional arguments (task_trajectories, etc.)
                )
            
            # Take environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action_val)
            done = terminated or truncated
            
            # Update episode reward
            current_episode_reward += reward
            
            # Get next tree structure
            next_tree_structure = info.get('tree_structure') if isinstance(info, dict) else None
            
            # Add experience to buffer
            self.rollout_buffer.add(
                obs, action_val, reward, value_val, log_prob_val, 
                done, action_mask_np, tree_structure
            )
            
            # Update state
            obs, tree_structure = next_obs, next_tree_structure
            self.total_steps += 1
            
            # Handle episode termination
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                
                # Reset environment
                obs, info = safe_env_reset(self.env)
                tree_structure = info.get('tree_structure') if isinstance(info, dict) else None
        
        # Store episode rewards
        self.episode_rewards.extend(episode_rewards)
        
        # Compute returns and advantages
        task_trajectories = kwargs.get('task_trajectories')
        self.rollout_buffer.compute_returns_and_advantages(
            self.policy, self.config.gamma, self.config.gae_lambda, task_trajectories
        )
        
        rollout_data = self.rollout_buffer.get()
        
        # Add to experience buffer if available
        if self.experience_buffer:
            trajectory_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            self.experience_buffer.add_trajectory(rollout_data, priority=trajectory_reward)
        
        self.logger.debug(f"Collected rollout with {len(rollout_data['observations'])} samples")
        return rollout_data
    
    def _collect_distributed_rollouts(self, n_steps: int, parallel_envs: List[Any], 
                                    **kwargs) -> Dict[str, torch.Tensor]:
        """Collect rollouts from multiple parallel environments"""
        
        # This is a simplified version - in practice, you'd use multiprocessing or async
        all_rollouts = []
        steps_per_env = n_steps // len(parallel_envs)
        
        for env in parallel_envs:
            # Temporarily switch environment
            original_env = self.env
            self.env = env
            
            rollout = self._collect_single_env_rollouts(steps_per_env, **kwargs)
            all_rollouts.append(rollout)
            
            # Restore original environment
            self.env = original_env
        
        # Combine rollouts
        combined_rollout = self._combine_rollouts(all_rollouts)
        return combined_rollout
    
    def _combine_rollouts(self, rollouts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combine multiple rollouts into a single batch"""
        
        combined = {}
        for key in rollouts[0].keys():
            if isinstance(rollouts[0][key], torch.Tensor):
                combined[key] = torch.cat([r[key] for r in rollouts], dim=0)
            else:
                # Handle non-tensor data
                combined[key] = np.concatenate([r[key] for r in rollouts], axis=0)
        
        return combined
    
    def learn(self, 
             rollout_data: Optional[Dict[str, torch.Tensor]] = None,
             use_experience_replay: bool = False,
             replay_ratio: float = 0.25) -> Dict[str, float]:
        """
        Learn from collected data with support for experience replay.
        
        Args:
            rollout_data: Fresh rollout data (if None, uses experience buffer)
            use_experience_replay: Whether to mix in experience replay
            replay_ratio: Fraction of batch to fill with replay data
        """
        self.policy.train()  # Set to training mode
        
        # Prepare training data
        training_batches = self._prepare_training_data(
            rollout_data, use_experience_replay, replay_ratio
        )
        
        # Training loop
        total_metrics = {}
        
        for epoch in range(self.config.n_epochs):
            epoch_metrics = []
            
            for batch in training_batches:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Train step
                metrics = self._train_step(batch)
                epoch_metrics.append(metrics)
            
            # Average metrics for this epoch
            for key in epoch_metrics[0].keys():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(np.mean([m[key] for m in epoch_metrics]))
        
        # Average across epochs
        final_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
        
        # Update curriculum if available
        if self.curriculum_manager:
            self.curriculum_manager.update_performance(final_metrics)
        
        self.training_metrics.update(final_metrics)
        return final_metrics
    
    def _prepare_training_data(self, 
                              rollout_data: Optional[Dict[str, torch.Tensor]],
                              use_experience_replay: bool,
                              replay_ratio: float) -> List[Dict[str, torch.Tensor]]:
        """Prepare training data with optional experience replay"""
        
        batches = []
        
        if rollout_data:
            # Create batches from fresh rollout data
            n_samples = len(rollout_data['observations'])
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch = {k: v[batch_indices] for k, v in rollout_data.items()}
                batches.append(batch)
        
        # Add experience replay if requested
        if use_experience_replay and self.experience_buffer and self.experience_buffer.size() > 0:
            n_replay_batches = max(1, int(len(batches) * replay_ratio))
            
            for _ in range(n_replay_batches):
                replay_trajectories = self.experience_buffer.sample(1)
                if replay_trajectories:
                    # Create batch from replay trajectory
                    replay_data = replay_trajectories[0]
                    n_replay_samples = len(replay_data['observations'])
                    
                    # Sample from replay trajectory
                    replay_indices = torch.randperm(n_replay_samples)[:self.config.batch_size]
                    replay_batch = {k: v[replay_indices] for k, v in replay_data.items()}
                    batches.append(replay_batch)
        
        return batches
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step"""
        
        # Unpack batch
        observations = batch['observations']
        actions = batch['actions'] 
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        action_masks = batch['action_masks']
        
        # Normalize advantages if configured
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        outputs = self.policy.forward(
            obs=observations,
            action_mask=action_masks,
            tree_structure=batch.get('tree_structures')
        )
        
        # Calculate losses
        dist = Categorical(logits=outputs['action_logits'])
        log_probs = dist.log_prob(actions)
        value_pred = outputs['value'].squeeze(-1)
        
        # PPO policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(value_pred, returns)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'explained_variance': self._explained_variance(value_pred, returns).item()
        }
    
    def _explained_variance(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate explained variance"""
        var_y = torch.var(y_true)
        return torch.max(torch.tensor(0.0), 1 - torch.var(y_true - y_pred) / var_y)
    
    def train(self, 
             total_timesteps: int,
             rollout_length: Optional[int] = None,
             log_interval: int = 10,
             save_interval: int = 100,
             use_curriculum: bool = True,
             use_experience_replay: bool = False,
             **kwargs) -> Dict[str, Any]:
        """
        Main training loop with advanced features.
        
        Args:
            total_timesteps: Total environment steps to train for
            rollout_length: Steps per rollout (defaults to config)
            log_interval: How often to log progress
            save_interval: How often to save checkpoints
            use_curriculum: Whether to use curriculum learning
            use_experience_replay: Whether to use experience replay
            **kwargs: Additional arguments passed to collect_rollouts
        """
        
        rollout_length = rollout_length or self.config.rollout_length
        n_updates = total_timesteps // rollout_length
        
        # Load from checkpoint if available
        start_update = 0
        if self.checkpoint_manager:
            checkpoint_data = self.checkpoint_manager.load_latest()
            if checkpoint_data:
                self.policy.load_state_dict(checkpoint_data['model_state'])
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
                start_update = checkpoint_data.get('update', 0)
                self.total_steps = checkpoint_data.get('total_steps', 0)
                self.logger.info(f"Resumed from update {start_update}, step {self.total_steps}")
        
        self.logger.info(f"Starting training: {total_timesteps} steps, {n_updates} updates")
        self.logger.info(f"Config: lr={self.config.learning_rate}, γ={self.config.gamma}, "
                        f"ε={self.config.clip_epsilon}")
        
        start_time = time.time()
        
        for update in range(start_update + 1, n_updates + 1):
            update_start = time.time()
            
            # Get curriculum configuration if using c