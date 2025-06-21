# JanusAI/ml/training/advanced_ppo_examples.py
"""
Advanced Examples Using the Refactored PPO Trainer
==================================================

This module demonstrates advanced use cases enabled by the decoupled
collect_rollouts and learn methods.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

from janus_ai.ml.training.ppo_trainer import PPOTrainer
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus_ai.ml.networks.hypothesis_net import HypothesisNet


class DistributedPPOTrainer:
    """
    Example: Distributed data collection with centralized learning.
    
    This pattern is useful for:
    - Scaling data collection across multiple machines
    - Utilizing multiple CPUs for environment simulation
    - Separating compute-intensive simulation from GPU training
    """
    
    def __init__(
        self,
        policy: HypothesisNet,
        env_factory: callable,
        n_collectors: int = 4,
        device: Optional[torch.device] = None
    ):
        self.policy = policy
        self.env_factory = env_factory
        self.n_collectors = n_collectors
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create collector trainers (one per worker)
        self.collectors = []
        for i in range(n_collectors):
            env = env_factory()
            # Each collector has its own environment but shares the policy
            collector = PPOTrainer(
                policy=policy,
                env=env,
                device=torch.device("cpu")  # Collectors run on CPU
            )
            self.collectors.append(collector)
        
        # Create learner (runs on GPU)
        self.learner = PPOTrainer(
            policy=policy,
            env=env_factory(),  # Dummy env for initialization
            device=self.device
        )
    
    def distributed_train(
        self,
        total_timesteps: int,
        rollout_length: int = 512,
        log_interval: int = 10
    ):
        """Train using distributed collection and centralized learning."""
        n_updates = total_timesteps // (rollout_length * self.n_collectors)
        
        print(f"Starting distributed training with {self.n_collectors} collectors")
        print(f"Total updates: {n_updates}")
        
        with ThreadPoolExecutor(max_workers=self.n_collectors) as executor:
            for update in range(n_updates):
                # Parallel data collection
                futures = []
                for collector in self.collectors:
                    # Each collector gathers rollouts in parallel
                    future = executor.submit(
                        collector.collect_rollouts,
                        n_steps=rollout_length
                    )
                    futures.append(future)
                
                # Wait for all collectors and combine data
                all_rollouts = []
                for future in futures:
                    rollout_data = future.result()
                    all_rollouts.append(rollout_data)
                
                # Combine rollout data
                combined_data = self._combine_rollouts(all_rollouts)
                
                # Centralized learning on GPU
                metrics = self.learner.learn(combined_data)
                
                # Sync policy weights back to collectors
                self._sync_policies()
                
                # Logging
                if update % log_interval == 0:
                    total_steps = update * rollout_length * self.n_collectors
                    mean_reward = np.mean([
                        np.mean(c.episode_rewards) if c.episode_rewards else 0
                        for c in self.collectors
                    ])
                    print(f"Update {update}/{n_updates}, "
                          f"Steps: {total_steps}/{total_timesteps}, "
                          f"Mean Reward: {mean_reward:.4f}, "
                          f"Loss: {metrics['loss']:.4f}")
    
    def _combine_rollouts(self, rollouts: List[Dict]) -> Dict:
        """Combine multiple rollout dictionaries into one."""
        combined = {}
        
        for key in rollouts[0].keys():
            if key == 'tree_structures':
                # Special handling for tree structures
                combined[key] = []
                for rollout in rollouts:
                    combined[key].extend(rollout[key])
            else:
                # Concatenate tensors
                tensors = [rollout[key] for rollout in rollouts]
                combined[key] = torch.cat(tensors, dim=0)
        
        return combined
    
    def _sync_policies(self):
        """Sync learner's policy weights to all collectors."""
        state_dict = self.learner.policy.state_dict()
        for collector in self.collectors:
            collector.policy.load_state_dict(state_dict)


class ExperienceReplayPPOTrainer(PPOTrainer):
    """
    Example: PPO with experience replay buffer.
    
    This demonstrates:
    - Saving rollouts to disk for later use
    - Training from a mixture of fresh and replayed data
    - Implementing simple prioritized replay
    """
    
    def __init__(self, *args, replay_dir: str = "./replay_buffer", **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(exist_ok=True)
        self.replay_files = []
        self.max_replay_files = 50
    
    def collect_and_save_rollouts(self, n_steps: int, save: bool = True) -> Dict:
        """Collect rollouts and optionally save to disk."""
        rollout_data = self.collect_rollouts(n_steps)
        
        if save:
            # Save rollout to disk
            filename = self.replay_dir / f"rollout_{self.total_timesteps}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(rollout_data, f)
            
            self.replay_files.append(filename)
            
            # Maintain buffer size
            if len(self.replay_files) > self.max_replay_files:
                oldest = self.replay_files.pop(0)
                oldest.unlink()  # Delete oldest file
        
        return rollout_data
    
    def learn_with_replay(
        self,
        fresh_data: Dict,
        replay_ratio: float = 0.5,
        prioritized: bool = True
    ) -> Dict:
        """Learn from mixture of fresh data and replayed experience."""
        # First, learn from fresh data
        fresh_metrics = self.learn(fresh_data)
        
        if not self.replay_files:
            return fresh_metrics
        
        # Determine how many replay batches to use
        n_replay_batches = int(self.n_epochs * replay_ratio)
        
        all_metrics = [fresh_metrics]
        
        for _ in range(n_replay_batches):
            # Select replay file (prioritized or random)
            if prioritized:
                # Simple priority: prefer recent experience
                weights = np.exp(np.linspace(-2, 0, len(self.replay_files)))
                weights /= weights.sum()
                replay_file = np.random.choice(self.replay_files, p=weights)
            else:
                replay_file = np.random.choice(self.replay_files)
            
            # Load and learn from replay
            with open(replay_file, 'rb') as f:
                replay_data = pickle.load(f)
            
            replay_metrics = self.learn(replay_data)
            all_metrics.append(replay_metrics)
        
        # Average metrics
        averaged_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            averaged_metrics[key] = np.mean(values)
        
        return averaged_metrics
    
    def train_with_replay(self, total_timesteps: int, **kwargs):
        """Training loop with experience replay."""
        rollout_length = kwargs.get('rollout_length', 2048)
        
        while self.total_timesteps < total_timesteps:
            # Collect fresh data
            fresh_data = self.collect_and_save_rollouts(rollout_length)
            
            # Learn from fresh + replayed data
            metrics = self.learn_with_replay(fresh_data)
            
            print(f"Timesteps: {self.total_timesteps}/{total_timesteps}, "
                  f"Loss: {metrics['loss']:.4f}, "
                  f"Replay Buffer Size: {len(self.replay_files)}")


class CurriculumPPOTrainer(PPOTrainer):
    """
    Example: Automatic curriculum learning with PPO.
    
    This demonstrates:
    - Adaptive difficulty based on performance
    - Separate data collection for different difficulty levels
    - Performance-based progression
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_difficulty = 1
        self.max_difficulty = 5
        self.performance_history = []
        self.promotion_threshold = 0.8  # 80% success rate
        self.demotion_threshold = 0.3   # 30% success rate
    
    def collect_rollouts_with_curriculum(self, n_steps: int) -> Dict:
        """Collect rollouts with current difficulty level."""
        # Set environment difficulty
        if hasattr(self.env, 'set_difficulty'):
            self.env.set_difficulty(self.current_difficulty)
        
        # Collect data
        rollout_data = self.collect_rollouts(n_steps)
        
        # Track performance
        if self.episode_rewards:
            recent_performance = np.mean(list(self.episode_rewards)[-10:])
            self.performance_history.append(recent_performance)
        
        return rollout_data
    
    def adjust_difficulty(self):
        """Adjust difficulty based on recent performance."""
        if len(self.performance_history) < 10:
            return
        
        recent_avg = np.mean(self.performance_history[-10:])
        
        # Normalize performance to [0, 1] if needed
        max_possible_reward = getattr(self.env, 'max_reward', 1.0)
        success_rate = recent_avg / max_possible_reward
        
        # Adjust difficulty
        if success_rate > self.promotion_threshold and self.current_difficulty < self.max_difficulty:
            self.current_difficulty += 1
            print(f"Promoted to difficulty {self.current_difficulty}")
            self.performance_history = []  # Reset after promotion
            
        elif success_rate < self.demotion_threshold and self.current_difficulty > 1:
            self.current_difficulty -= 1
            print(f"Demoted to difficulty {self.current_difficulty}")
            self.performance_history = []  # Reset after demotion
    
    def train_with_curriculum(self, total_timesteps: int, **kwargs):
        """Training loop with automatic curriculum."""
        rollout_length = kwargs.get('rollout_length', 2048)
        
        print(f"Starting curriculum training at difficulty {self.current_difficulty}")
        
        while self.total_timesteps < total_timesteps:
            # Collect with current difficulty
            rollout_data = self.collect_rollouts_with_curriculum(rollout_length)
            
            # Learn from collected data
            metrics = self.learn(rollout_data)
            
            # Adjust difficulty based on performance
            self.adjust_difficulty()
            
            # Logging
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards)
                print(f"Steps: {self.total_timesteps}/{total_timesteps}, "
                      f"Difficulty: {self.current_difficulty}, "
                      f"Mean Reward: {mean_reward:.4f}, "
                      f"Loss: {metrics['loss']:.4f}")


class AdaptiveBatchSizePPOTrainer(PPOTrainer):
    """
    Example: PPO with adaptive batch size based on gradient variance.
    
    This demonstrates:
    - Monitoring gradient statistics during learning
    - Dynamically adjusting batch size for stability
    - Separate collection and learning with different configurations
    """
    
    def __init__(self, *args, min_batch_size: int = 32, max_batch_size: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.gradient_variance_history = []
    
    def learn_with_adaptive_batch(self, rollout_data: Dict) -> Dict:
        """Learn with dynamically adjusted batch size."""
        # Estimate gradient variance with small batch
        test_batch_size = self.min_batch_size
        gradient_vars = []
        
        # Sample a few mini-batches to estimate variance
        n_samples = len(rollout_data['observations'])
        for _ in range(5):
            indices = np.random.choice(n_samples, test_batch_size)
            batch = self._create_minibatch(rollout_data, indices)
            
            # Compute gradients
            self.optimizer.zero_grad()
            metrics = self._train_step(batch)
            
            # Store gradient norms
            grad_norm = 0.0
            for p in self.policy.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            gradient_vars.append(np.sqrt(grad_norm))
        
        # Adjust batch size based on gradient variance
        grad_var = np.var(gradient_vars)
        self.gradient_variance_history.append(grad_var)
        
        if grad_var > 0.1:  # High variance - increase batch size
            self.batch_size = min(int(self.batch_size * 1.2), self.max_batch_size)
        elif grad_var < 0.01:  # Low variance - can use smaller batch
            self.batch_size = max(int(self.batch_size * 0.8), self.min_batch_size)
        
        print(f"Gradient variance: {grad_var:.4f}, Adjusted batch size: {self.batch_size}")
        
        # Now do actual learning with adjusted batch size
        return self.learn(rollout_data)


def demonstrate_advanced_features():
    """Demonstrate the advanced features enabled by the refactored design."""
    
    # Create mock environment and policy for demonstration
    from janus_ai.core.grammar.base_grammar import ProgressiveGrammar
    from janus_ai.core.expressions.expression import Variable
    
    # Setup
    grammar = ProgressiveGrammar()
    variables = [Variable(f'x{i}', i) for i in range(3)]
    
    def create_env():
        """Factory function for creating environments."""
        # Create simple test data
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1] - X[:, 2]
        
        return SymbolicDiscoveryEnv(
            grammar=grammar,
            X_data=X,
            y_data=y,
            variables=variables
        )
    
    env = create_env()
    policy = HypothesisNet(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n,
        hidden_dim=128,
        encoder_type='transformer',
        grammar=grammar
    )
    
    print("=== Advanced PPO Training Examples ===\n")
    
    # Example 1: Distributed Training
    print("1. Distributed Collection with Centralized Learning")
    print("-" * 50)
    distributed_trainer = DistributedPPOTrainer(
        policy=policy,
        env_factory=create_env,
        n_collectors=4
    )
    distributed_trainer.distributed_train(
        total_timesteps=10000,
        rollout_length=256
    )
    
    print("\n2. Experience Replay PPO")
    print("-" * 50)
    replay_trainer = ExperienceReplayPPOTrainer(
        policy=policy,
        env=create_env(),
        replay_dir="./ppo_replay_buffer"
    )
    replay_trainer.train_with_replay(
        total_timesteps=10000
    )
    
    print("\n3. Curriculum Learning PPO")
    print("-" * 50)
    curriculum_trainer = CurriculumPPOTrainer(
        policy=policy,
        env=create_env()
    )
    curriculum_trainer.train_with_curriculum(
        total_timesteps=10000
    )
    
    print("\n4. Adaptive Batch Size PPO")
    print("-" * 50)
    adaptive_trainer = AdaptiveBatchSizePPOTrainer(
        policy=policy,
        env=create_env(),
        min_batch_size=16,
        max_batch_size=256
    )
    # Use the adaptive learning
    rollout_data = adaptive_trainer.collect_rollouts(1000)
    metrics = adaptive_trainer.learn_with_adaptive_batch(rollout_data)
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    demonstrate_advanced_features()