"""
PPOTrainer
==========

Implements the Proximal Policy Optimization (PPO) algorithm for training
HypothesisNet policies in symbolic discovery environments.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Deque
import os # For checkpoint_dir path management
import time # For timing training

# Internal project imports based on new structure
from janus.ml.networks.hypothesis_net import HypothesisNet # The policy definition
from janus.environments.base.symbolic_env import SymbolicDiscoveryEnv # The environment type
from janus.environments.base.symbolic_env import safe_env_reset # Utility for env reset

# CheckpointManager will be in src/janus/utils/io/checkpoint_manager.py
# Assuming a CheckpointManager class is available from this path
try:
    from janus.utils.io.checkpoint_manager import CheckpointManager
except ImportError:
    print("Warning: CheckpointManager not found. Checkpointing will be disabled.")
    CheckpointManager = None


class RolloutBuffer:
    """
    Stores experiences (observations, actions, rewards, etc.) collected during agent rollouts.
    Provides methods to compute returns and advantages.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Clears the buffer."""
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.action_masks: List[np.ndarray] = []
        # Store tree structures if TreeEncoder is used and env provides them
        self.tree_structures: List[Optional[Dict[int, List[int]]]] = []
        
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(self, 
            obs: np.ndarray, 
            action: int, 
            reward: float, 
            value: float, 
            log_prob: float, 
            done: bool, 
            action_mask: np.ndarray, 
            tree_structure: Optional[Dict[int, List[int]]]):
        """Adds a single step's experience to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.tree_structures.append(tree_structure)

    def compute_returns_and_advantages(self, 
                                       policy: HypothesisNet, 
                                       gamma: float, 
                                       gae_lambda: float,
                                       task_trajectories: Optional[torch.Tensor] = None):
        """
        Computes Generalized Advantage Estimation (GAE) advantages and discounted returns.
        
        Args:
            policy: The current policy network to get the value of the last observation.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            task_trajectories: Optional task-specific trajectories for meta-learning context.
        """
        # Convert lists to numpy arrays for efficient computation
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32) # 1.0 if terminal, 0.0 otherwise

        # Get value estimate for the last observed state to bootstrap returns/advantages
        # If the last state in the buffer is a terminal state, its next value is 0.
        # Otherwise, query the policy for its value.
        with torch.no_grad():
            last_obs_np = self.observations[-1]
            last_obs = torch.FloatTensor(last_obs_np).unsqueeze(0) # Ensure (1, obs_dim)
            
            last_action_mask_np = self.action_masks[-1]
            last_action_mask = torch.BoolTensor(last_action_mask_np).unsqueeze(0) # Ensure (1, action_dim)
            
            last_tree_structure = self.tree_structures[-1] # Can be None

            # Get the value from the policy's forward pass
            policy_outputs = policy(
                obs=last_obs, 
                action_mask=last_action_mask, 
                task_trajectories=task_trajectories, 
                tree_structure=last_tree_structure
            )
            last_value = policy_outputs['value'].squeeze().item()

        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0 # Initialize last GAE-Lambda value
        for t in reversed(range(len(rewards))):
            # next_non_terminal is 0 if `dones[t]` is True (terminal state), 1 otherwise
            next_non_terminal = 1.0 - dones[t] 
            
            # The value of the next state (V(S_t+1))
            # If it's the very last step in the rollout, use `last_value` (bootstrapped value)
            # Otherwise, use the value stored for the next state from the buffer
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t+1]

            # TD-error (delta_t)
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            # GAE formula (cumulative sum of weighted deltas)
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        # Compute discounted returns (targets for the value function)
        self.returns = advantages + values
        
        # Normalize advantages for more stable training
        self.advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    def get(self) -> Dict[str, Any]:
        """
        Retrieves all collected data as PyTorch tensors.

        Returns:
            A dictionary containing tensors for observations, actions, rewards, values,
            log probabilities, advantages, returns, action masks, and tree structures.
        """
        # Convert observations and action masks to tensors (observations can be complex, use np.array then FloatTensor)
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
            'tree_structures': self.tree_structures # Keep as list of dicts/None for now
        }


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer.

    Manages the training loop for a HypothesisNet policy using collected rollouts.
    """
    def __init__(
        self,
        policy: HypothesisNet,
        env: SymbolicDiscoveryEnv,
        learning_rate: float = 3e-4,
        n_epochs: int = 10, # Number of PPO epochs to run on collected data per update
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2, # PPO clipping parameter (epsilon)
        value_coef: float = 0.5, # Coefficient for the value loss in the total loss
        entropy_coef: float = 0.01, # Coefficient for the entropy bonus in the total loss
        max_grad_norm: float = 0.5, # Maximum gradient norm for clipping
        checkpoint_dir: Optional[str] = None
    ) -> None:
        self.policy = policy
        self.env = env
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.n_epochs = n_epochs # Stored for use in the train method
        
        self.rollout_buffer = RolloutBuffer()
        self.episode_rewards: Deque[float] = deque(maxlen=100) # Tracks recent episode rewards

        self.checkpoint_manager: Optional[CheckpointManager] = None
        if checkpoint_dir and CheckpointManager:
            # Ensure checkpoint directory exists
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        elif checkpoint_dir:
            print(f"Checkpoint directory specified ({checkpoint_dir}), but CheckpointManager is not available.")

    def collect_rollouts(self, 
                         n_steps: int, 
                         task_trajectories: Optional[torch.Tensor] = None,
                         ai_model_representation: Optional[torch.Tensor] = None, # New
                         ai_model_type_idx: Optional[torch.Tensor] = None # New
                        ) -> Dict[str, torch.Tensor]:
        """
        Collects experience rollouts from the environment using the current policy.

        Args:
            n_steps: Number of steps to collect for the rollout.
            task_trajectories: Optional meta-learning context for the policy.
            ai_model_representation: Optional AI model representation for AIHypothesisNet.
            ai_model_type_idx: Optional AI model type index for AIHypothesisNet.

        Returns:
            A dictionary containing the collected rollout data as tensors.
        """
        self.rollout_buffer.reset()
        
        # Reset environment and get initial observation and info
        obs, info = safe_env_reset(self.env)
        # Extract tree structure if provided by the environment for TreeEncoder
        tree_structure = info.get('tree_structure') if isinstance(info, dict) else None

        for _ in range(n_steps):
            # Prepare observation for the policy (unsqueeze for batch dimension)
            obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0) # (1, obs_dim)
            
            # Get action mask from the environment
            action_mask_np = self.env.get_action_mask() # (action_dim,)
            action_mask_tensor = torch.BoolTensor(action_mask_np).unsqueeze(0) # (1, action_dim)

            with torch.no_grad():
                # Get action, log_prob, and value from the policy
                # Pass all relevant arguments to policy.get_action, including new AI-specific ones
                action_val, log_prob_val, value_val = self.policy.get_action(
                    obs=obs_tensor, 
                    action_mask=action_mask_tensor, 
                    task_trajectories=task_trajectories,
                    tree_structure=tree_structure
                )
            
            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_val)
            done = terminated or truncated # Combined done flag
            
            # Get next tree structure if env supports it
            next_tree_structure = info.get('tree_structure') if isinstance(info, dict) else None

            # Add the experience to the rollout buffer
            self.rollout_buffer.add(obs, action_val, reward, value_val, log_prob_val, done, action_mask_np, tree_structure)
            
            obs, tree_structure = next_obs, next_tree_structure

            # If an episode terminates, reset the environment
            if done:
                ep_rew = info.get('episode_reward', reward) # Get full episode reward if available in info
                self.episode_rewards.append(ep_rew) # Store episode reward
                
                # Reset env for the next episode in the rollout
                obs, info = safe_env_reset(self.env)
                tree_structure = info.get('tree_structure') if isinstance(info, dict) else None

        # After collecting all steps, compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(self.policy, self.gamma, self.gae_lambda, task_trajectories)
        return self.rollout_buffer.get()

    def train_step(self, 
                   batch: Dict[str, torch.Tensor], 
                   task_trajectories_batch: Optional[torch.Tensor] = None,
                   ai_model_representation_batch: Optional[torch.Tensor] = None, # New
                   ai_model_type_idx_batch: Optional[torch.Tensor] = None # New
                  ) -> Dict[str, float]:
        """
        Performs a single PPO training step using a mini-batch of collected data.

        Args:
            batch: A dictionary containing mini-batch data (observations, actions, etc.).
            task_trajectories_batch: Task-specific trajectories for meta-learning context.
            ai_model_representation_batch: AI model representation for AIHypothesisNet.
            ai_model_type_idx_batch: AI model type index for AIHypothesisNet.

        Returns:
            A dictionary of training metrics (loss, policy_loss, value_loss, entropy).
        """
        # Unpack batch data
        observations = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        action_masks = batch['action_masks']
        tree_structures_batch = batch.get('tree_structures') # Get tree structures if available

        # Perform forward pass through the policy
        # Pass all relevant inputs to the policy's forward method
        outputs = self.policy.forward(
            obs=observations, 
            action_mask=action_masks, 
            task_trajectories=task_trajectories_batch, 
            tree_structure=None # Assuming batching tree_structures for TreeEncoder is handled internally or passed as None
        )

        # Calculate new log probabilities and value predictions
        dist = Categorical(logits=outputs['action_logits'])
        log_probs = dist.log_prob(actions)
        value_pred = outputs['value'].squeeze(-1) # Ensure value_pred is 1D

        # PPO Policy Loss (Clipped Surrogate Objective)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value Loss (Mean Squared Error)
        value_loss = F.mse_loss(value_pred, returns)

        # Entropy Bonus (for exploration)
        entropy = dist.entropy().mean()

        # Total PPO Loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Optimize the policy
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'loss': loss.item(), 
            'policy_loss': policy_loss.item(), 
            'value_loss': value_loss.item(), 
            'entropy': entropy.item()
        }

    def train(self, 
              total_timesteps: int, 
              rollout_length: int = 2048, 
              batch_size: int = 64, 
              log_interval: int = 10,
              # task_trajectories_for_training and ai_model_info_for_training
              # are for if these are fixed across all rollouts/epochs,
              # otherwise they need to be generated/sampled within collect_rollouts or train_step.
              # For now, keeping them as Optional[None] and assuming they are handled by env or policy.
              task_trajectories_for_training: Optional[torch.Tensor] = None,
              ai_model_representation_for_training: Optional[torch.Tensor] = None,
              ai_model_type_idx_for_training: Optional[torch.Tensor] = None
             ):
        """
        Main PPO training loop.

        Args:
            total_timesteps: Total number of environment steps to train for.
            rollout_length: Number of steps to collect in each rollout phase.
            batch_size: Mini-batch size for PPO updates.
            log_interval: How often (in updates) to print progress.
            task_trajectories_for_training: Optional task trajectories if fixed for all training.
            ai_model_representation_for_training: Optional fixed AI model representation.
            ai_model_type_idx_for_training: Optional fixed AI model type index.
        """
        n_updates = total_timesteps // rollout_length
        current_timesteps = 0
        
        # Attempt to load from checkpoint if a manager is configured
        if self.checkpoint_manager:
            loaded_timesteps = self.load_from_checkpoint()
            if loaded_timesteps > 0:
                current_timesteps = loaded_timesteps
                print(f"Resumed training from timestep {current_timesteps}")

        print(f"Starting PPO training for {total_timesteps} timesteps ({n_updates} updates)")
        print(f"Rollout length: {rollout_length}, Batch size: {batch_size}, Epochs per update: {self.n_epochs}")
        print("-" * 50)

        for update in range(1, n_updates + 1):
            start_time = time.time()

            # Collect rollouts
            # Pass AI-specific and meta-learning context to rollout collection
            rollout_data = self.collect_rollouts(
                rollout_length, 
                task_trajectories=task_trajectories_for_training,
                ai_model_representation=ai_model_representation_for_training,
                ai_model_type_idx=ai_model_type_idx_for_training
            )

            num_samples_in_rollout = len(rollout_data['observations'])
            if num_samples_in_rollout == 0:
                print(f"Update {update}/{n_updates}: No samples in rollout. Skipping training for this update.")
                continue # Skip to next update if no samples collected

            # Convert rollout data to tensors for batching
            # Ensure all tensors are on the correct device if using GPU
            for k, v in rollout_data.items():
                if isinstance(v, torch.Tensor):
                    rollout_data[k] = v.to(self.policy.parameters().__next__().device) # Move to policy's device
            
            data_indices = np.arange(num_samples_in_rollout)
            last_loss = 0.0 # Initialize for checkpointing metrics

            # Perform PPO epochs on the collected rollout data
            for epoch_num in range(self.n_epochs):
                np.random.shuffle(data_indices) # Shuffle indices for mini-batching

                for start_idx in range(0, num_samples_in_rollout, batch_size):
                    mini_batch_indices_np = data_indices[start_idx : start_idx + batch_size]

                    if len(mini_batch_indices_np) == 0:
                        continue

                    # Create mini-batch
                    mini_batch = {}
                    for key, value_from_rollout in rollout_data.items():
                        if key == 'tree_structures':
                            # tree_structures cannot be directly indexed by np array if it's a list of dicts.
                            # It needs to be handled as a list of individual elements.
                            mini_batch[key] = [value_from_rollout[i] for i in mini_batch_indices_np.tolist()]
                        elif isinstance(value_from_rollout, torch.Tensor):
                            mini_batch[key] = value_from_rollout[mini_batch_indices_np]
                        else:
                            try:
                                mini_batch[key] = value_from_rollout[mini_batch_indices_np]
                            except (TypeError, IndexError):
                                mini_batch[key] = [value_from_rollout[i] for i in mini_batch_indices_np.tolist()]
                            except Exception as e_inner:
                                print(f"Warning: Could not create batch for key '{key}' (type: {type(value_from_rollout)}). Error: {e_inner}. Skipping this key for the batch.")

                    if 'observations' in mini_batch and len(mini_batch['observations']) > 0:
                        # Perform a single training step on the mini-batch
                        metrics = self.train_step(
                            mini_batch, 
                            task_trajectories_batch=task_trajectories_for_training, # If fixed per training run
                            ai_model_representation_batch=ai_model_representation_for_training,
                            ai_model_type_idx_batch=ai_model_type_idx_for_training
                        )
                        last_loss = metrics.get('loss', 0.0) # Store last loss for checkpointing
                    else:
                        print(f"Skipping train_step due to empty or invalid mini-batch for update {update}, epoch {epoch_num}, start_idx {start_idx}")

            current_timesteps += num_samples_in_rollout
            step_time = time.time() - start_time

            # Logging progress
            if update % log_interval == 0:
                avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else float('nan')
                print(f"Update {update}/{n_updates}, Timesteps: {current_timesteps}/{total_timesteps}, Avg Reward: {avg_reward:.3f}, Loss: {last_loss:.4f}, Step Time: {step_time:.2f}s")
            
            # Checkpointing
            # Checkpoint at intervals or at the very end of training
            if self.checkpoint_manager and current_timesteps > 0 and (update % (10000 // rollout_length) == 0 or update == n_updates):
                # Optionally save environment state if it has a `get_state` method
                env_state = self.env.get_state() if hasattr(self.env, 'get_state') else None

                state_to_save = {
                    'policy_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'timesteps': current_timesteps,
                    'env_state': env_state,
                    'episode_rewards_deque': list(self.episode_rewards) # Save deque as a list
                }
                metrics_to_save = {
                    'mean_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
                    'loss': last_loss
                }
                self.checkpoint_manager.save_checkpoint(state_to_save, current_timesteps, metrics_to_save)
                print(f"Saved checkpoint at timestep {current_timesteps}")

        print("\nPPO Training complete!")
        if self.checkpoint_manager:
            # Save final checkpoint explicitly
            self.checkpoint_manager.save_checkpoint(
                {
                    'policy_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'timesteps': current_timesteps,
                    'env_state': self.env.get_state() if hasattr(self.env, 'get_state') else None,
                    'episode_rewards_deque': list(self.episode_rewards)
                },
                current_timesteps,
                {
                    'mean_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
                    'loss': last_loss
                },
                is_final=True # Indicate this is the final checkpoint
            )
            print("Final checkpoint saved.")


    def load_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> int:
        """
        Loads the trainer's state (policy, optimizer, episode rewards, environment state)
        from a checkpoint.

        Args:
            checkpoint_path: Path to a specific checkpoint file. If None, loads the latest.

        Returns:
            The timestep from which training is resumed, or 0 if no checkpoint was loaded.
        """
        if not self.checkpoint_manager:
            print("Checkpoint manager not configured. Cannot load checkpoint.")
            return 0

        checkpoint_data = None
        if checkpoint_path:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        else:
            checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()

        if checkpoint_data:
            try:
                # Load policy and optimizer states
                self.policy.load_state_dict(checkpoint_data['state']['policy_state_dict'])
                self.optimizer.load_state_dict(checkpoint_data['state']['optimizer_state_dict'])

                # Restore episode_rewards deque
                if 'episode_rewards_deque' in checkpoint_data['state']:
                    self.episode_rewards = deque(checkpoint_data['state']['episode_rewards_deque'], maxlen=self.episode_rewards.maxlen)

                # Restore environment state if available and the environment supports it
                if 'env_state' in checkpoint_data['state'] and checkpoint_data['state']['env_state'] is not None:
                    if hasattr(self.env, 'set_state'):
                        self.env.set_state(checkpoint_data['state']['env_state'])
                    else:
                        print("Warning: Environment has no set_state method. Cannot restore env_state from checkpoint.")

                print(f"Successfully loaded checkpoint from step {checkpoint_data.get('timestep', 0)}")
                return checkpoint_data['state'].get('timesteps', 0)
            except Exception as e:
                print(f"Error loading state from checkpoint: {e}. Starting training from scratch.")
                return 0
        else:
            print("No checkpoint found to load.")
            return 0


if __name__ == "__main__":
    # This __main__ block is for testing the PPOTrainer specifically.
    # It requires a mock environment and a HypothesisNet policy.

    # --- Dummy Environment and Policy Setup for Testing ---
    # Assume SymbolicDiscoveryEnv, ProgressiveGrammar, Variable are importable from their new paths
    # (as defined in the imports at the top of this file)
    from janus.core.grammar.base_grammar import ProgressiveGrammar
    from janus.core.expressions.expression import Variable

    print("Setting up dummy environment and policy for PPOTrainer test...")
    # Initialize a simple grammar and variables
    grammar = ProgressiveGrammar()
    variables = [Variable(), Variable()]
    
    # Create some dummy data for the environment
    dummy_data = np.column_stack([np.random.rand(100), np.random.rand(100) * 2, np.random.rand(100) + 1])

    # Instantiate a SymbolicDiscoveryEnv. Ensure it has `get_action_mask` and optionally `get_state`/`set_state`
    # and provides `tree_structure` in its info dict if TreeEncoder is used.
    # For testing, we might need to mock or ensure the env has these methods.
    # Here, assuming SymbolicDiscoveryEnv from janus.environments.base.symbolic_env provides them.
    try:
        env = SymbolicDiscoveryEnv(
            grammar=grammar, 
            target_data=dummy_data, 
            variables=variables, 
            max_depth=5, 
            max_complexity=10, 
            reward_config={'mse_weight': -1.0},
            provide_tree_structure=True # Hypothetical flag to ensure info contains tree_structure
        )
    except TypeError: # Fallback if provide_tree_structure is not an arg
        env = SymbolicDiscoveryEnv(
            grammar=grammar, 
            target_data=dummy_data, 
            variables=variables, 
            max_depth=5, 
            max_complexity=10, 
            reward_config={'mse_weight': -1.0}
        )
        print("Warning: SymbolicDiscoveryEnv does not support 'provide_tree_structure'. TreeEncoder might not function as expected.")


    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Instantiate HypothesisNet (or AIHypothesisNet) as the policy
    # Using 'transformer' encoder for simplicity in testing as it doesn't strictly need `tree_structure`
    policy = HypothesisNet(
        obs_dim=obs_dim, 
        act_dim=action_dim, 
        hidden_dim=128, # Smaller hidden_dim for faster testing
        encoder_type='transformer',
        grammar=grammar,
        use_meta_learning=True # Enable meta-learning path for testing
    )
    
    print(f"Policy has {sum(p.numel() for p in policy.parameters())} parameters.")

    # --- PPO Trainer Initialization and Training ---
    print("\n--- Testing PPOTrainer with HypothesisNet ---")
    
    # Define a temporary checkpoint directory for testing
    test_checkpoint_dir = "./ppo_test_checkpoints"
    # Clean up previous test checkpoints if any
    if os.path.exists(test_checkpoint_dir):
        import shutil
        shutil.rmtree(test_checkpoint_dir)
    os.makedirs(test_checkpoint_dir, exist_ok=True)

    ppo_trainer = PPOTrainer(
        policy=policy,
        env=env,
        learning_rate=1e-4,
        n_epochs=2, # Small number of epochs for quick test
        checkpoint_dir=test_checkpoint_dir # Pass the checkpoint directory
    )

    # Dummy task trajectories for policy's meta-learning component if needed
    # Make sure this matches the expected input shape for policy's task_encoder
    # (batch_size, num_trajectories, trajectory_length, obs_feature_dim)
    dummy_task_trajectories = torch.randn(
        1, # Batch size for single-task training
        3, # Number of dummy trajectories
        10, # Length of each trajectory
        policy.node_feature_dim # Feature dimension of each step in trajectory
    )

    # Run a short training loop
    try:
        ppo_trainer.train(
            total_timesteps=100, # Very short total timesteps for quick test
            rollout_length=32, 
            batch_size=16, 
            log_interval=1,
            task_trajectories_for_training=dummy_task_trajectories.to(policy.parameters().__next__().device) # Pass to device
        )
    except Exception as e:
        print(f"\nError during PPO training test: {e}")
        import traceback
        traceback.print_exc()

    print("\nPPOTrainer test finished.")
    
    # Optional: Clean up test checkpoint directory
    if os.path.exists(test_checkpoint_dir):
        import shutil
        print(f"Cleaning up test checkpoint directory: {test_checkpoint_dir}")
        shutil.rmtree(test_checkpoint_dir)

