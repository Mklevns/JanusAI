# janus/ml/training/ppo_trainer.py
"""
Refactored PPO Trainer with Decoupled Training Loop
===================================================

This module contains a refactored PPO trainer that separates data collection
and learning phases for better modularity and flexibility.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple, Deque, Any
from collections import deque
from dataclasses import dataclass, field

# Import necessary modules (adjust paths as needed)
from janus_ai.ml.networks.hypothesis_net import HypothesisNet
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus_ai.utils.io.checkpoint_manager import CheckpointManager
from janus_ai.utils.general_utils import safe_env_reset


@dataclass
class RolloutBuffer:
    """
    Storage for rollout data collected during environment interaction.

    This buffer stores trajectories and computes advantages using GAE.
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)
    tree_structures: List[Optional[Dict]] = field(default_factory=list)

    def reset(self):
        """Clear all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()
        self.action_masks.clear()
        self.tree_structures.clear()

    def add(self, obs: np.ndarray, action: int, reward: float, value: float,
            log_prob: float, done: bool, action_mask: np.ndarray,
            tree_structure: Optional[Dict] = None):
        """Add a single transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.tree_structures.append(tree_structure)

    def compute_returns_and_advantages(self, policy: nn.Module, gamma: float,
                                      gae_lambda: float,
                                      task_trajectories: Optional[torch.Tensor] = None):
        """Compute returns and advantages using GAE."""
        # Get final value estimate
        if len(self.observations) > 0:
            with torch.no_grad():
                last_obs = torch.FloatTensor(self.observations[-1]).unsqueeze(0)
                last_action_mask = torch.BoolTensor(self.action_masks[-1]).unsqueeze(0)
                outputs = policy(last_obs, last_action_mask, task_trajectories)
                last_value = outputs['value'].item()
        else:
            last_value = 0.0

        # Compute GAE
        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
                next_done = False
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        self.advantages = advantages
        self.returns = returns

        # Normalize advantages
        adv_array = np.array(self.advantages)
        self.advantages = ((adv_array - adv_array.mean()) /
                          (adv_array.std() + 1e-8)).tolist()

    def get(self) -> Dict[str, Any]:
        """Get all data as tensors."""
        obs_tensor = torch.FloatTensor(np.array(self.observations))
        action_masks_tensor = torch.BoolTensor(np.array(self.action_masks))

        return {
            'observations': obs_tensor,
            'actions': torch.LongTensor(self.actions),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'log_probs': torch.FloatTensor(self.log_probs),
            'advantages': torch.FloatTensor(self.advantages),
            'returns': torch.FloatTensor(self.returns),
            'action_masks': action_masks_tensor,
            'tree_structures': self.tree_structures
        }

    def __len__(self) -> int:
        """Return the number of transitions stored."""
        return len(self.observations)


class PPOTrainer:
    """
    Refactored Proximal Policy Optimization (PPO) trainer.

    This version separates data collection and learning phases for better
    modularity and follows modern RL library patterns.
    """

    def __init__(
        self,
        policy: HypothesisNet,
        env: SymbolicDiscoveryEnv,
        learning_rate: float = 3e-4,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize PPO trainer.

        Args:
            policy: The policy network (HypothesisNet)
            env: The environment for training
            learning_rate: Learning rate for optimizer
            n_epochs: Number of epochs to train on each rollout
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Mini-batch size for training
            checkpoint_dir: Directory for saving checkpoints
            device: Device to run training on
        """
        self.policy = policy
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move policy to device
        self.policy = self.policy.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

        # PPO hyperparameters
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()

        # Tracking
        self.episode_rewards: Deque[float] = deque(maxlen=100)
        self.total_timesteps = 0
        self.n_updates = 0

        # Checkpoint manager
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if checkpoint_dir and CheckpointManager:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)

    def collect_rollouts(
        self,
        n_steps: int,
        task_trajectories: Optional[torch.Tensor] = None,
        ai_model_representation: Optional[torch.Tensor] = None,
        ai_model_type_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Collect experience from the environment for n_steps.

        This method handles environment interaction and fills the rollout buffer.

        Args:
            n_steps: Number of environment steps to collect
            task_trajectories: Optional task trajectories for meta-learning
            ai_model_representation: Optional AI model representation
            ai_model_type_idx: Optional AI model type index

        Returns:
            Dictionary containing collected rollout data
        """
        # Reset buffer
        self.rollout_buffer.reset()

        # Get initial observation if needed
        if not hasattr(self, '_current_obs'):
            self._current_obs, self._current_info = safe_env_reset(self.env)
            self._episode_reward = 0.0

        # Collect rollouts
        for step in range(n_steps):
            obs_tensor = torch.FloatTensor(self._current_obs).unsqueeze(0).to(self.device)

            # Get action mask
            if hasattr(self.env, 'get_action_mask'):
                action_mask = self.env.get_action_mask()
            else:
                action_mask = np.ones(self.env.action_space.n, dtype=bool)
            action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

            # Get tree structure if available
            tree_structure = self._current_info.get('tree_structure') if isinstance(self._current_info, dict) else None

            # Get action from policy
            with torch.no_grad():
                outputs = self.policy(
                    obs=obs_tensor,
                    action_mask=action_mask_tensor,
                    task_trajectories=task_trajectories,
                    tree_structure=tree_structure,
                    ai_model_representation=ai_model_representation,
                    ai_model_type_idx=ai_model_type_idx
                )

                # Sample action
                if 'action_logits' in outputs:
                    dist = Categorical(logits=outputs['action_logits'])
                    action = dist.sample()
                    log_prob = dist.log_prob(action).item()
                else:
                    # Fallback for policies that return actions directly
                    action = outputs.get('action', 0)
                    log_prob = 0.0

                value = outputs['value'].item()
                action = action.item()

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self._episode_reward += reward

            # Store transition
            self.rollout_buffer.add(
                obs=self._current_obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                action_mask=action_mask,
                tree_structure=tree_structure
            )

            # Update current state
            self._current_obs = next_obs
            self._current_info = info
            self.total_timesteps += 1

            # Handle episode end
            if done:
                self.episode_rewards.append(self._episode_reward)
                self._current_obs, self._current_info = safe_env_reset(self.env)
                self._episode_reward = 0.0

        # Compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(
            self.policy, self.gamma, self.gae_lambda, task_trajectories
        )

        return self.rollout_buffer.get()

    def learn(
        self,
        rollout_data: Dict[str, torch.Tensor],
        task_trajectories: Optional[torch.Tensor] = None,
        ai_model_representation: Optional[torch.Tensor] = None,
        ai_model_type_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train the policy on collected rollout data.

        This method handles the learning phase with multiple epochs and mini-batches.

        Args:
            rollout_data: Dictionary containing rollout data from collect_rollouts
            task_trajectories: Optional task trajectories for meta-learning
            ai_model_representation: Optional AI model representation
            ai_model_type_idx: Optional AI model type index

        Returns:
            Dictionary of training metrics averaged over all epochs and batches
        """
        # Move data to device
        for key, value in rollout_data.items():
            if isinstance(value, torch.Tensor):
                rollout_data[key] = value.to(self.device)

        # Get number of samples
        n_samples = len(rollout_data['observations'])
        if n_samples == 0:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        # Training metrics
        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []

        # Prepare indices for shuffling
        indices = np.arange(n_samples)

        # Train for n_epochs
        for epoch in range(self.n_epochs):
            # Shuffle data
            np.random.shuffle(indices)

            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]

                if len(batch_indices) == 0:
                    continue

                # Create mini-batch
                batch = self._create_minibatch(rollout_data, batch_indices)

                # Prepare additional inputs if provided
                batch_task_traj = None
                batch_ai_repr = None
                batch_ai_type = None

                if task_trajectories is not None:
                    batch_task_traj = task_trajectories.expand(len(batch_indices), -1, -1, -1)
                if ai_model_representation is not None:
                    batch_ai_repr = ai_model_representation.expand(len(batch_indices), -1)
                if ai_model_type_idx is not None:
                    batch_ai_type = ai_model_type_idx.expand(len(batch_indices))

                # Perform training step
                metrics = self._train_step(
                    batch,
                    batch_task_traj,
                    batch_ai_repr,
                    batch_ai_type
                )

                # Accumulate metrics
                epoch_losses.append(metrics['loss'])
                epoch_policy_losses.append(metrics['policy_loss'])
                epoch_value_losses.append(metrics['value_loss'])
                epoch_entropies.append(metrics['entropy'])

        # Update counter
        self.n_updates += 1

        # Return averaged metrics
        return {
            'loss': np.mean(epoch_losses),
            'policy_loss': np.mean(epoch_policy_losses),
            'value_loss': np.mean(epoch_value_losses),
            'entropy': np.mean(epoch_entropies),
            'n_updates': self.n_updates
        }

    def _create_minibatch(
        self,
        rollout_data: Dict[str, Any],
        indices: np.ndarray
    ) -> Dict[str, Any]:
        """Create a mini-batch from rollout data."""
        batch = {}

        for key, value in rollout_data.items():
            if key == 'tree_structures':
                # Handle list of tree structures
                batch[key] = [value[i] for i in indices.tolist()]
            elif isinstance(value, torch.Tensor):
                batch[key] = value[indices]
            else:
                # Try to handle other types
                try:
                    batch[key] = value[indices]
                except (TypeError, IndexError):
                    batch[key] = [value[i] for i in indices.tolist()]

        return batch

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        task_trajectories_batch: Optional[torch.Tensor] = None,
        ai_model_representation_batch: Optional[torch.Tensor] = None,
        ai_model_type_idx_batch: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform a single PPO training step on a mini-batch.

        Args:
            batch: Mini-batch data
            task_trajectories_batch: Optional task trajectories
            ai_model_representation_batch: Optional AI model representations
            ai_model_type_idx_batch: Optional AI model type indices

        Returns:
            Dictionary of training metrics
        """
        # Unpack batch
        observations = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        action_masks = batch['action_masks']
        tree_structures = batch.get('tree_structures')

        # Forward pass
        outputs = self.policy(
            obs=observations,
            action_mask=action_masks,
            task_trajectories=task_trajectories_batch,
            tree_structure=None,  # Tree structures handled internally if needed
            ai_model_representation=ai_model_representation_batch,
            ai_model_type_idx=ai_model_type_idx_batch
        )

        # Calculate losses
        dist = Categorical(logits=outputs['action_logits'])
        log_probs = dist.log_prob(actions)
        values = outputs['value'].squeeze(-1)

        # PPO policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def train(
        self,
        total_timesteps: int,
        rollout_length: int = 2048,
        log_interval: int = 10,
        save_interval: int = 100,
        task_trajectories: Optional[torch.Tensor] = None,
        ai_model_representation: Optional[torch.Tensor] = None,
        ai_model_type_idx: Optional[torch.Tensor] = None,
        callback: Optional[callable] = None
    ) -> Dict[str, List[float]]:
        """
        Main training loop orchestrating collection and learning.

        This is the high-level training method that alternates between
        collecting rollouts and learning from them.

        Args:
            total_timesteps: Total number of environment steps to train for
            rollout_length: Number of steps per rollout collection phase
            log_interval: Frequency of logging (in updates)
            save_interval: Frequency of checkpointing (in updates)
            task_trajectories: Optional fixed task trajectories
            ai_model_representation: Optional fixed AI model representation
            ai_model_type_idx: Optional fixed AI model type index
            callback: Optional callback function called after each update

        Returns:
            Dictionary of training history
        """
        # Training history
        history = {
            'timesteps': [],
            'mean_reward': [],
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }

        # Calculate number of updates
        n_updates = total_timesteps // rollout_length

        # Load from checkpoint if available
        start_timestep = 0
        if self.checkpoint_manager:
            loaded_state = self.load_from_checkpoint()
            if loaded_state:
                start_timestep = loaded_state.get('total_timesteps', 0)
                print(f"Resumed training from timestep {start_timestep}")

        print(f"Starting PPO training for {total_timesteps} timesteps")
        print(f"Rollout length: {rollout_length}, Batch size: {self.batch_size}")
        print(f"Number of epochs: {self.n_epochs}, Number of updates: {n_updates}")
        print("-" * 50)

        # Training loop
        start_time = time.time()

        while self.total_timesteps < total_timesteps:
            # Collect rollouts
            rollout_data = self.collect_rollouts(
                n_steps=rollout_length,
                task_trajectories=task_trajectories,
                ai_model_representation=ai_model_representation,
                ai_model_type_idx=ai_model_type_idx
            )

            # Learn from rollouts
            metrics = self.learn(
                rollout_data=rollout_data,
                task_trajectories=task_trajectories,
                ai_model_representation=ai_model_representation,
                ai_model_type_idx=ai_model_type_idx
            )

            # Update history
            history['timesteps'].append(self.total_timesteps)
            history['mean_reward'].append(np.mean(self.episode_rewards) if self.episode_rewards else 0.0)
            history['loss'].append(metrics['loss'])
            history['policy_loss'].append(metrics['policy_loss'])
            history['value_loss'].append(metrics['value_loss'])
            history['entropy'].append(metrics['entropy'])

            # Logging
            if self.n_updates % log_interval == 0:
                elapsed_time = time.time() - start_time
                fps = (self.total_timesteps - start_timestep) / elapsed_time

                print(f"\nUpdate {self.n_updates}/{n_updates}")
                print(f"Timesteps: {self.total_timesteps}/{total_timesteps}")
                print(f"FPS: {fps:.1f}")
                print(f"Mean reward: {history['mean_reward'][-1]:.4f}")
                print(f"Loss: {metrics['loss']:.4f}")
                print(f"Policy loss: {metrics['policy_loss']:.4f}")
                print(f"Value loss: {metrics['value_loss']:.4f}")
                print(f"Entropy: {metrics['entropy']:.4f}")
                print("-" * 50)

            # Checkpointing
            if self.checkpoint_manager and self.n_updates % save_interval == 0:
                self.save_checkpoint()

            # Callback
            if callback is not None:
                callback(self, metrics)

        print(f"\nTraining completed in {time.time() - start_time:.1f} seconds")

        return history

    def save_checkpoint(self) -> bool:
        """Save training checkpoint."""
        if not self.checkpoint_manager:
            return False

        checkpoint_data = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'n_updates': self.n_updates,
            'episode_rewards': list(self.episode_rewards),
            'hyperparameters': {
                'n_epochs': self.n_epochs,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'batch_size': self.batch_size
            }
        }

        return self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            self.total_timesteps,
            {'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0}
        )

    def load_from_checkpoint(self) -> Optional[Dict]:
        """Load from checkpoint."""
        if not self.checkpoint_manager:
            return None

        checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint_data:
            self.policy.load_state_dict(checkpoint_data['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            self.total_timesteps = checkpoint_data.get('total_timesteps', 0)
            self.n_updates = checkpoint_data.get('n_updates', 0)

            if 'episode_rewards' in checkpoint_data:
                self.episode_rewards = deque(checkpoint_data['episode_rewards'], maxlen=100)

            return checkpoint_data

        return None