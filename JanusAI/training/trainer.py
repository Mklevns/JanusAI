"""
janus/training/trainer.py

Unified trainer that orchestrates multiple training modes for the Janus project.
Supports physics discovery, AI interpretability, and hybrid approaches.
"""

import os
import json
import pickle
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# For progress tracking
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class TrainerConfig:
    """Configuration for UnifiedTrainer."""
    
    # General settings
    mode: str = 'symbolic'  # 'symbolic', 'rl', 'hybrid'
    max_episodes: int = 1000
    max_steps_per_episode: int = 100
    eval_frequency: int = 100
    save_frequency: int = 500
    
    # Device and parallelization
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    n_jobs: int = 1  # For symbolic/genetic algorithms
    
    # Logging and checkpointing
    log_level: str = 'INFO'
    log_to_console: bool = True
    log_to_file: bool = True
    log_dir: str = './logs'
    checkpoint_dir: str = './checkpoints'
    save_best_only: bool = True
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 32
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 100
    min_improvement: float = 1e-4
    
    # Mode-specific settings
    symbolic_config: Optional[Dict[str, Any]] = None
    rl_config: Optional[Dict[str, Any]] = None
    hybrid_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default configurations for each mode."""
        if self.symbolic_config is None:
            self.symbolic_config = {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'tournament_size': 3,
                'elitism': True
            }
        
        if self.rl_config is None:
            self.rl_config = {
                'algorithm': 'SAC',  # or 'PPO', 'TD3'
                'buffer_size': 10000,
                'warmup_steps': 1000,
                'gradient_steps': 1,
                'target_update_interval': 1,
                'policy_lr': 3e-4,
                'value_lr': 3e-4,
                'gamma': 0.99,
                'tau': 0.005
            }
        
        if self.hybrid_config is None:
            self.hybrid_config = {
                'symbolic_episodes': 500,
                'rl_episodes': 500,
                'alternation_frequency': 50,
                'knowledge_transfer': True
            }
        
        # Auto-detect device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrainingMetrics:
    """Tracks and manages training metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        self.current_episode = 0
        
    def update(self, **kwargs):
        """Update metrics for current episode."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            
            # Track best values
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value
        
        self.current_episode += 1
    
    def get_recent_avg(self, key: str, n: int = 100) -> float:
        """Get average of last n values for a metric."""
        if key not in self.metrics:
            return 0.0
        values = self.metrics[key][-n:]
        return sum(values) / len(values) if values else 0.0
    
    def get_best(self, key: str) -> float:
        """Get best value for a metric."""
        return self.best_metrics.get(key, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for saving."""
        return {
            'metrics': dict(self.metrics),
            'best_metrics': self.best_metrics,
            'current_episode': self.current_episode
        }


class UnifiedTrainer:
    """
    Unified trainer for multiple training paradigms in Janus.
    
    Supports:
    - Symbolic regression (genetic algorithms, etc.)
    - Reinforcement learning (SAC, PPO, TD3)
    - Hybrid approaches combining both
    """
    
    def __init__(self, 
                 config: TrainerConfig,
                 env: Any,
                 **components):
        """
        Initialize the unified trainer.
        
        Args:
            config: Training configuration
            env: Environment (SymbolicDiscoveryEnv, gym.Env, etc.)
            **components: Algorithm-specific components (policy, regressor, etc.)
        """
        self.config = config
        self.env = env
        self.components = components
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize metrics tracking
        self.metrics = TrainingMetrics()
        
        # Create directories
        self._create_directories()
        
        # Device setup
        self.device = torch.device(config.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Move components to device if they're PyTorch models
        self._move_to_device()
        
        # Training state
        self.best_solution = None
        self.best_reward = float('-inf')
        self.episodes_since_improvement = 0
        
        self.logger.info(f"UnifiedTrainer initialized for mode: {config.mode}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f'janus_trainer_{id(self)}')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.log_level))
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            os.makedirs(self.config.log_dir, exist_ok=True)
            log_file = os.path.join(self.config.log_dir, f'training_{int(time.time())}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.log_level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _create_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def _move_to_device(self):
        """Move PyTorch components to the specified device."""
        for name, component in self.components.items():
            if isinstance(component, nn.Module):
                component.to(self.device)
                self.logger.info(f"Moved {name} to {self.device}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop based on the configured mode.
        
        Returns:
            Dict containing training results and best solution
        """
        self.logger.info(f"Starting {self.config.mode} training for {self.config.max_episodes} episodes")
        
        start_time = time.time()
        
        try:
            if self.config.mode == 'symbolic':
                results = self._train_symbolic()
            elif self.config.mode == 'rl':
                results = self._train_rl()
            elif self.config.mode == 'hybrid':
                results = self._train_hybrid()
            else:
                raise ValueError(f"Unknown training mode: {self.config.mode}")
            
            training_time = time.time() - start_time
            
            # Add timing and final metrics
            results.update({
                'training_time': training_time,
                'total_episodes': self.metrics.current_episode,
                'best_reward': self.best_reward,
                'final_metrics': self.metrics.to_dict()
            })
            
            self.logger.info(f"Training completed in {training_time:.2f}s")
            self.logger.info(f"Best reward achieved: {self.best_reward:.4f}")
            
            return results
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            return self._get_current_results()
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def _train_symbolic(self) -> Dict[str, Any]:
        """Train using symbolic regression/genetic algorithms."""
        self.logger.info("Starting symbolic regression training")
        
        if 'regressor' not in self.components:
            raise ValueError("Symbolic training requires 'regressor' component")
        
        regressor = self.components['regressor']
        
        # Initialize population if needed
        if hasattr(regressor, 'initialize_population'):
            regressor.initialize_population()
        
        best_expression = None
        
        for episode in tqdm(range(self.config.max_episodes), desc="Symbolic Training"):
            # Run one generation/iteration
            if hasattr(regressor, 'evolve'):
                # Genetic algorithm style
                population, fitness_scores = regressor.evolve()
                best_idx = np.argmax(fitness_scores)
                episode_reward = fitness_scores[best_idx]
                current_expression = population[best_idx]
            else:
                # Direct optimization style
                expression, episode_reward = regressor.step()
                current_expression = expression
            
            # Update metrics
            self.metrics.update(
                episode=episode,
                reward=episode_reward,
                expression_complexity=len(str(current_expression)) if current_expression else 0
            )
            
            # Check for improvement
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_solution = current_expression
                best_expression = current_expression
                self.episodes_since_improvement = 0
                
                self.logger.info(f"Episode {episode}: New best reward {episode_reward:.4f}")
                self.logger.info(f"Best expression: {best_expression}")
            else:
                self.episodes_since_improvement += 1
            
            # Early stopping check
            if self._should_early_stop():
                self.logger.info(f"Early stopping at episode {episode}")
                break
            
            # Periodic evaluation and saving
            if episode % self.config.eval_frequency == 0:
                self._log_progress(episode)
            
            if episode % self.config.save_frequency == 0:
                self._save_checkpoint(episode)
        
        return {
            'mode': 'symbolic',
            'best_expression': str(best_expression) if best_expression else None,
            'best_solution': best_expression
        }
    
    def _train_rl(self) -> Dict[str, Any]:
        """Train using reinforcement learning."""
        self.logger.info("Starting RL training")
        
        if 'policy' not in self.components:
            raise ValueError("RL training requires 'policy' component")
        
        policy = self.components['policy']
        
        for episode in tqdm(range(self.config.max_episodes), desc="RL Training"):
            # Reset environment
            obs, info = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(self.config.max_steps_per_episode):
                # Select action
                if hasattr(policy, 'act'):
                    action = policy.act(obs)
                else:
                    # Fallback for different policy interfaces
                    action = policy(torch.tensor(obs, device=self.device))
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                
                # Environment step
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Store experience if agent supports it
                if hasattr(policy, 'store_experience'):
                    policy.store_experience(obs, action, reward, terminated)
                
                if terminated or truncated:
                    break
            
            # Train policy
            if hasattr(policy, 'train_step'):
                policy_loss = policy.train_step()
            else:
                policy_loss = 0.0
            
            # Update metrics
            self.metrics.update(
                episode=episode,
                reward=episode_reward,
                episode_length=episode_steps,
                policy_loss=policy_loss
            )
            
            # Check for improvement
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_solution = info.get('expression', f'episode_{episode}_solution')
                self.episodes_since_improvement = 0
                
                self.logger.info(f"Episode {episode}: New best reward {episode_reward:.4f}")
            else:
                self.episodes_since_improvement += 1
            
            # Early stopping check
            if self._should_early_stop():
                self.logger.info(f"Early stopping at episode {episode}")
                break
            
            # Periodic evaluation and saving
            if episode % self.config.eval_frequency == 0:
                self._log_progress(episode)
            
            if episode % self.config.save_frequency == 0:
                self._save_checkpoint(episode)
        
        return {
            'mode': 'rl',
            'best_solution': self.best_solution,
            'final_policy': policy
        }
    
    def _train_hybrid(self) -> Dict[str, Any]:
        """Train using hybrid symbolic-RL approach."""
        self.logger.info("Starting hybrid training")
        
        if 'regressor' not in self.components or 'policy' not in self.components:
            raise ValueError("Hybrid training requires both 'regressor' and 'policy' components")
        
        hybrid_config = self.config.hybrid_config
        alternation_freq = hybrid_config['alternation_frequency']
        
        symbolic_results = None
        rl_results = None
        
        for phase in range(0, self.config.max_episodes, alternation_freq):
            phase_episodes = min(alternation_freq, self.config.max_episodes - phase)
            
            # Alternate between symbolic and RL phases
            if (phase // alternation_freq) % 2 == 0:
                self.logger.info(f"Hybrid phase {phase}: Symbolic training")
                # Temporarily adjust config for this phase
                old_max = self.config.max_episodes
                self.config.max_episodes = phase_episodes
                symbolic_results = self._train_symbolic()
                self.config.max_episodes = old_max
                
                # Transfer knowledge to RL if configured
                if hybrid_config['knowledge_transfer'] and 'regressor' in self.components:
                    self._transfer_symbolic_to_rl()
            else:
                self.logger.info(f"Hybrid phase {phase}: RL training")
                old_max = self.config.max_episodes
                self.config.max_episodes = phase_episodes
                rl_results = self._train_rl()
                self.config.max_episodes = old_max
                
                # Transfer knowledge to symbolic if configured
                if hybrid_config['knowledge_transfer'] and 'policy' in self.components:
                    self._transfer_rl_to_symbolic()
        
        return {
            'mode': 'hybrid',
            'symbolic_results': symbolic_results,
            'rl_results': rl_results,
            'best_solution': self.best_solution
        }
    
    def _transfer_symbolic_to_rl(self):
        """Transfer knowledge from symbolic regressor to RL policy."""
        # This is a placeholder for knowledge transfer mechanisms
        # In practice, this might involve:
        # - Using symbolic expressions as policy initialization
        # - Providing symbolic solutions as expert demonstrations
        # - Using symbolic fitness as auxiliary rewards
        self.logger.info("Transferring symbolic knowledge to RL policy")
    
    def _transfer_rl_to_symbolic(self):
        """Transfer knowledge from RL policy to symbolic regressor."""
        # This might involve:
        # - Using RL-discovered expressions as genetic algorithm seeds
        # - Extracting symbolic patterns from learned policies
        # - Using RL evaluation as fitness function
        self.logger.info("Transferring RL knowledge to symbolic regressor")
    
    def _should_early_stop(self) -> bool:
        """Check if training should stop early."""
        if not self.config.early_stopping:
            return False
        
        return self.episodes_since_improvement >= self.config.patience
    
    def _log_progress(self, episode: int):
        """Log training progress."""
        recent_reward = self.metrics.get_recent_avg('reward', 100)
        best_reward = self.metrics.get_best('reward')
        
        self.logger.info(
            f"Episode {episode}: "
            f"Recent avg reward: {recent_reward:.4f}, "
            f"Best reward: {best_reward:.4f}, "
            f"Episodes since improvement: {self.episodes_since_improvement}"
        )
    
    def _get_current_results(self) -> Dict[str, Any]:
        """Get current training results (for interruption handling)."""
        return {
            'mode': self.config.mode,
            'best_solution': self.best_solution,
            'best_reward': self.best_reward,
            'current_episode': self.metrics.current_episode,
            'metrics': self.metrics.to_dict(),
            'interrupted': True
        }
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save a training checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir, 
                f'checkpoint_episode_{self.metrics.current_episode}.pkl'
            )
        
        checkpoint = {
            'config': asdict(self.config),
            'metrics': self.metrics.to_dict(),
            'best_solution': self.best_solution,
            'best_reward': self.best_reward,
            'episodes_since_improvement': self.episodes_since_improvement,
            'components': {}
        }
        
        # Save component states
        for name, component in self.components.items():
            if isinstance(component, nn.Module):
                checkpoint['components'][name] = component.state_dict()
            elif hasattr(component, 'get_state'):
                checkpoint['components'][name] = component.get_state()
            else:
                # Try to pickle the component
                try:
                    checkpoint['components'][name] = component
                except Exception as e:
                    self.logger.warning(f"Could not save component {name}: {e}")
        
        # Save checkpoint
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.logger.info(f"Checkpoint saved to {path}")
    
    def _save_checkpoint(self, episode: int):
        """Save checkpoint during training."""
        if not self.config.save_best_only or self.episodes_since_improvement == 0:
            self.save_checkpoint()
    
    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore metrics
        self.metrics.metrics = defaultdict(list, checkpoint['metrics']['metrics'])
        self.metrics.best_metrics = checkpoint['metrics']['best_metrics']
        self.metrics.current_episode = checkpoint['metrics']['current_episode']
        
        # Restore training state
        self.best_solution = checkpoint['best_solution']
        self.best_reward = checkpoint['best_reward']
        self.episodes_since_improvement = checkpoint['episodes_since_improvement']
        
        # Restore component states
        for name, state in checkpoint['components'].items():
            if name in self.components:
                component = self.components[name]
                if isinstance(component, nn.Module):
                    component.load_state_dict(state)
                elif hasattr(component, 'set_state'):
                    component.set_state(state)
                else:
                    self.components[name] = state
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current best solution."""
        self.logger.info(f"Evaluating best solution over {num_episodes} episodes")
        
        eval_rewards = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            
            # Use best solution if available
            if self.best_solution is not None:
                # Apply best solution to environment
                if hasattr(self.env, 'set_expression'):
                    self.env.set_expression(self.best_solution)
                
                # Evaluate
                obs, reward, terminated, truncated, info = self.env.step(0)  # Dummy action
                episode_reward = reward
            
            eval_rewards.append(episode_reward)
        
        eval_results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards)
        }
        
        self.logger.info(f"Evaluation results: {eval_results}")
        return eval_results


# Convenience functions for quick training setup
def train_symbolic_regressor(env, regressor, config: Optional[TrainerConfig] = None) -> Dict[str, Any]:
    """Convenience function for symbolic regression training."""
    if config is None:
        config = TrainerConfig(mode='symbolic')
    
    trainer = UnifiedTrainer(config, env, regressor=regressor)
    return trainer.train()


def train_rl_policy(env, policy, config: Optional[TrainerConfig] = None) -> Dict[str, Any]:
    """Convenience function for RL policy training."""
    if config is None:
        config = TrainerConfig(mode='rl')
    
    trainer = UnifiedTrainer(config, env, policy=policy)
    return trainer.train()


def train_hybrid(env, regressor, policy, config: Optional[TrainerConfig] = None) -> Dict[str, Any]:
    """Convenience function for hybrid training."""
    if config is None:
        config = TrainerConfig(mode='hybrid')
    
    trainer = UnifiedTrainer(config, env, regressor=regressor, policy=policy)
    return trainer.train()


# Example usage
if __name__ == "__main__":
    # This would be used with actual Janus components
    
    # Example config
    config = TrainerConfig(
        mode='symbolic',
        max_episodes=500,
        log_level='INFO',
        early_stopping=True,
        patience=50
    )
    
    print(f"Example config created for {config.mode} training")
    print(f"Max episodes: {config.max_episodes}")
    print(f"Device: {config.device}")
    print(f"Symbolic config: {config.symbolic_config}")