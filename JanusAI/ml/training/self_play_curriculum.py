# File: JanusAI/ml/training/self_play_curriculum.py
"""
Self-Play Curriculum Learning Framework

Implements the adversarial self-play loop between TaskSetter and LawDiscoverer agents,
creating an open-ended curriculum for physics discovery.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import wandb
import json
import time
from pathlib import Path
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from janus_ai.ml.agents.task_setter import TaskSetterAgent, TaskSetterConfig
from janus_ai.ml.training.meta_trainer import MAMLTrainer, MetaLearningConfig
from janus_ai.physics.data.dynamic_task_distribution import DynamicPhysicsTaskDistribution
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv


@dataclass
class SelfPlayConfig:
    """Configuration for self-play curriculum learning"""
    
    # Training schedule
    total_iterations: int = 1000
    setter_train_interval: int = 10  # Train setter every N iterations
    discoverer_train_interval: int = 1  # Train discoverer every iteration
    
    # Episode configuration
    tasks_per_iteration: int = 5
    episodes_per_task: int = 10
    
    # Performance tracking
    performance_window: int = 50
    min_setter_performance: float = 0.2  # Min performance to start setter training
    
    # Checkpointing
    checkpoint_interval: int = 50
    checkpoint_dir: str = "checkpoints/self_play"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "janus-self-play"
    log_interval: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PerformanceTracker:
    """Tracks performance metrics for both agents"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Discoverer metrics
        self.discoverer_success_rates = deque(maxlen=window_size)
        self.discoverer_rewards = deque(maxlen=window_size)
        self.discoverer_complexities = deque(maxlen=window_size)
        
        # Setter metrics
        self.setter_rewards = deque(maxlen=window_size)
        self.task_difficulties = deque(maxlen=window_size)
        self.task_diversity_scores = deque(maxlen=window_size)
        
        # Task history
        self.task_history = []
        self.performance_by_task_type = {}
        
    def update_discoverer_metrics(self, success_rate: float, reward: float, 
                                 complexity: float, task_type: str):
        """Update discoverer performance metrics"""
        self.discoverer_success_rates.append(success_rate)
        self.discoverer_rewards.append(reward)
        self.discoverer_complexities.append(complexity)
        
        # Track performance by task type
        if task_type not in self.performance_by_task_type:
            self.performance_by_task_type[task_type] = []
        self.performance_by_task_type[task_type].append(success_rate)
    
    def update_setter_metrics(self, reward: float, task_difficulty: float, 
                             diversity_score: float):
        """Update setter performance metrics"""
        self.setter_rewards.append(reward)
        self.task_difficulties.append(task_difficulty)
        self.task_diversity_scores.append(diversity_score)
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for logging"""
        stats = {}
        
        # Discoverer stats
        if self.discoverer_success_rates:
            stats['discoverer/mean_success_rate'] = np.mean(self.discoverer_success_rates)
            stats['discoverer/mean_reward'] = np.mean(self.discoverer_rewards)
            stats['discoverer/mean_complexity'] = np.mean(self.discoverer_complexities)
            stats['discoverer/success_rate_std'] = np.std(self.discoverer_success_rates)
        
        # Setter stats
        if self.setter_rewards:
            stats['setter/mean_reward'] = np.mean(self.setter_rewards)
            stats['setter/mean_task_difficulty'] = np.mean(self.task_difficulties)
            stats['setter/mean_diversity_score'] = np.mean(self.task_diversity_scores)
        
        # Learning progress
        if len(self.discoverer_success_rates) >= 20:
            recent = list(self.discoverer_success_rates)[-20:]
            older = list(self.discoverer_success_rates)[-40:-20]
            if older:
                stats['discoverer/learning_progress'] = np.mean(recent) - np.mean(older)
        
        return stats


class SelfPlayCallback(BaseCallback):
    """Custom callback for monitoring self-play training"""
    
    def __init__(self, tracker: PerformanceTracker, verbose: int = 0):
        super().__init__(verbose)
        self.tracker = tracker
        
    def _on_step(self) -> bool:
        # Log current metrics if available
        if self.n_calls % 1000 == 0:
            stats = self.tracker.get_summary_stats()
            for key, value in stats.items():
                self.logger.record(key, value)
        return True


class SelfPlayCurriculumTrainer:
    """
    Main trainer that orchestrates self-play between TaskSetter and LawDiscoverer
    """
    
    def __init__(self, config: SelfPlayConfig):
        self.config = config
        
        # Initialize components
        self.task_distribution = DynamicPhysicsTaskDistribution(include_noise=True)
        
        # Initialize discoverer (using MAML)
        self.meta_config = MetaLearningConfig(
            meta_lr=0.0003,
            adaptation_lr=0.01,
            adaptation_steps=5,
            tasks_per_batch=config.tasks_per_iteration,
            device=config.device
        )
        
        # We'll initialize MAML trainer after determining dimensions
        self.discoverer_trainer = None
        self.discoverer_env = None
        
        # Initialize setter
        self.setter_config = TaskSetterConfig()
        self.setter_agent = None
        
        # Performance tracking
        self.tracker = PerformanceTracker(config.performance_window)
        
        # Iteration counter
        self.iteration = 0
        
        # Setup logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config={
                    "self_play_config": config.__dict__,
                    "meta_config": self.meta_config.__dict__,
                    "setter_config": self.setter_config.__dict__
                }
            )
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _initialize_agents(self):
        """Initialize both agents after determining environment dimensions"""
        # Sample a task to determine dimensions
        sample_task = self.task_distribution.sample_task()
        
        # Create discoverer environment
        self.discoverer_env = SymbolicDiscoveryEnv(
            data=sample_task.generate_data(100),
            target_expr="unknown",  # Will be discovered
            max_depth=10,
            max_complexity=30
        )
        
        # Get dimensions
        obs_dim = self.discoverer_env.observation_space.shape[0]
        action_dim = self.discoverer_env.action_space.n
        
        print(f"Environment dimensions - Obs: {obs_dim}, Actions: {action_dim}")
        
        # Initialize MAML trainer for discoverer
        from janus_ai.ml.training.meta_trainer import MetaLearningPolicy
        
        self.discoverer_policy = MetaLearningPolicy(
            observation_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256
        )
        
        self.discoverer_trainer = MAMLTrainer(
            policy=self.discoverer_policy,
            config=self.meta_config,
            task_distribution=self.task_distribution
        )
        
        # Initialize setter agent
        self.setter_agent = TaskSetterAgent(
            config=self.setter_config,
            discoverer_env=self.discoverer_env
        )
    
    def train(self):
        """Main self-play training loop"""
        print("Starting Self-Play Curriculum Training")
        print("=" * 50)
        
        # Initialize agents
        self._initialize_agents()
        
        for iteration in range(self.config.total_iterations):
            self.iteration = iteration
            
            # Phase 1: Generate tasks using TaskSetter (or random initially)
            if self._should_use_setter():
                tasks = self._generate_tasks_with_setter()
            else:
                tasks = self._generate_random_tasks()
            
            # Phase 2: Train LawDiscoverer on generated tasks
            discoverer_metrics = self._train_discoverer_on_tasks(tasks)
            
            # Phase 3: Update performance tracking
            self._update_performance_tracking(tasks, discoverer_metrics)
            
            # Phase 4: Train TaskSetter based on discoverer performance
            if self._should_train_setter():
                setter_metrics = self._train_setter()
            
            # Logging and checkpointing
            if iteration % self.config.log_interval == 0:
                self._log_metrics()
            
            if iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint()
            
            # Print progress
            if iteration % 10 == 0:
                stats = self.tracker.get_summary_stats()
                print(f"\nIteration {iteration}/{self.config.total_iterations}")
                print(f"Discoverer Success Rate: {stats.get('discoverer/mean_success_rate', 0):.3f}")
                print(f"Mean Task Difficulty: {stats.get('setter/mean_task_difficulty', 0.5):.3f}")
    
    def _should_use_setter(self) -> bool:
        """Determine if we should use setter or random tasks"""
        # Use setter after warmup period and if performance is sufficient
        if self.iteration < 50:
            return False
        
        stats = self.tracker.get_summary_stats()
        mean_success = stats.get('discoverer/mean_success_rate', 0)
        
        return mean_success > self.config.min_setter_performance
    
    def _should_train_setter(self) -> bool:
        """Determine if setter should be trained this iteration"""
        return (self.iteration > 0 and 
                self.iteration % self.config.setter_train_interval == 0 and
                self._should_use_setter())
    
    def _generate_tasks_with_setter(self) -> List[Any]:
        """Generate tasks using the TaskSetter agent"""
        tasks = []
        
        for _ in range(self.config.tasks_per_iteration):
            # Create observation from performance history
            performance_history = [
                {
                    'success_rate': self.tracker.discoverer_success_rates[-1] 
                                   if self.tracker.discoverer_success_rates else 0.5,
                    'final_reward': self.tracker.discoverer_rewards[-1]
                                   if self.tracker.discoverer_rewards else 0.0,
                    'expression_complexity': self.tracker.discoverer_complexities[-1]
                                           if self.tracker.discoverer_complexities else 10.0
                }
            ]
            
            # Get task from setter
            task = self.task_distribution.sample_task_batch_dynamic(
                n_tasks=1,
                setter_agent=self.setter_agent,
                performance_history=performance_history
            )[0]
            
            tasks.append(task)
        
        return tasks
    
    def _generate_random_tasks(self) -> List[Any]:
        """Generate random tasks for initial warmup"""
        return self.task_distribution.sample_task_batch(
            self.config.tasks_per_iteration,
            curriculum=True
        )
    
    def _train_discoverer_on_tasks(self, tasks: List[Any]) -> Dict[str, float]:
        """Train the discoverer on a batch of tasks"""
        all_metrics = []
        
        for task in tasks:
            # Train on this task
            task_metrics = self.discoverer_trainer.train_on_task(
                task=task,
                n_episodes=self.config.episodes_per_task
            )
            
            # Extract key metrics
            metrics = {
                'success_rate': task_metrics.get('success_rate', 0.0),
                'final_reward': task_metrics.get('mean_reward', 0.0),
                'expression_complexity': task_metrics.get('expression_complexity', 10.0),
                'task_type': task.name.split('_')[0]
            }
            
            all_metrics.append(metrics)
            
            # Update tracker
            self.tracker.update_discoverer_metrics(
                success_rate=metrics['success_rate'],
                reward=metrics['final_reward'],
                complexity=metrics['expression_complexity'],
                task_type=metrics['task_type']
            )
        
        # Return average metrics
        return {
            'mean_success_rate': np.mean([m['success_rate'] for m in all_metrics]),
            'mean_reward': np.mean([m['final_reward'] for m in all_metrics]),
            'mean_complexity': np.mean([m['expression_complexity'] for m in all_metrics])
        }
    
    def _train_setter(self) -> Dict[str, float]:
        """Train the TaskSetter agent"""
        # Train for a fixed number of steps
        train_steps = 2048
        
        print(f"\nTraining TaskSetter for {train_steps} steps...")
        
        # Use callback to track metrics
        callback = SelfPlayCallback(self.tracker)
        
        # Train
        self.setter_agent.train(total_timesteps=train_steps)
        
        return {
            'setter_train_steps': train_steps
        }
    
    def _update_performance_tracking(self, tasks: List[Any], metrics: Dict[str, float]):
        """Update performance tracking with latest results"""
        # Update task distribution with performance data
        for task in tasks:
            if hasattr(task, 'task_id'):
                performance = {
                    'success_rate': metrics['mean_success_rate'],
                    'timestamp': time.time()
                }
                self.task_distribution.update_task_performance(
                    task.task_id, 
                    performance
                )
        
        # Calculate task diversity
        if len(self.tracker.task_history) > 1:
            # Simple diversity metric based on parameter variance
            recent_tasks = tasks[-10:]
            param_variance = self._calculate_task_diversity(recent_tasks)
            self.tracker.task_diversity_scores.append(param_variance)
    
    def _calculate_task_diversity(self, tasks: List[Any]) -> float:
        """Calculate diversity score for a set of tasks"""
        if len(tasks) < 2:
            return 0.0
        
        # Extract parameter vectors
        param_vectors = []
        for task in tasks:
            if hasattr(task, 'parameters'):
                params = task.parameters
                vector = [
                    params.get('m1', 1.0),
                    params.get('k', 10.0),
                    params.get('L1', 1.0),
                    params.get('g', 9.81)
                ]
                param_vectors.append(vector)
        
        if not param_vectors:
            return 0.0
        
        # Calculate variance across parameters
        param_array = np.array(param_vectors)
        diversity = np.mean(np.std(param_array, axis=0))
        
        return diversity
    
    def _log_metrics(self):
        """Log metrics to wandb and console"""
        stats = self.tracker.get_summary_stats()
        
        if self.config.use_wandb:
            wandb.log(stats, step=self.iteration)
        
        # Also log task type distribution
        if self.tracker.performance_by_task_type:
            task_type_stats = {}
            for task_type, performances in self.tracker.performance_by_task_type.items():
                if performances:
                    task_type_stats[f'task_type/{task_type}_success_rate'] = np.mean(performances[-10:])
            
            if self.config.use_wandb:
                wandb.log(task_type_stats, step=self.iteration)
    
    def _save_checkpoint(self):
        """Save checkpoint of both agents"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"iteration_{self.iteration}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save discoverer
        discoverer_path = checkpoint_path / "discoverer.pt"
        torch.save({
            'policy_state_dict': self.discoverer_policy.state_dict(),
            'meta_config': self.meta_config.__dict__,
            'iteration': self.iteration
        }, discoverer_path)
        
        # Save setter
        setter_path = checkpoint_path / "setter"
        self.setter_agent.save(str(setter_path))
        
        # Save performance history
        history_path = checkpoint_path / "performance_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'discoverer_success_rates': list(self.tracker.discoverer_success_rates),
                'setter_rewards': list(self.tracker.setter_rewards),
                'task_difficulties': list(self.tracker.task_difficulties),
                'iteration': self.iteration
            }, f)
        
        print(f"Saved checkpoint at iteration {self.iteration}")
    
    def load_checkpoint(self, iteration: int):
        """Load a checkpoint from a specific iteration"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"iteration_{iteration}"
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load discoverer
        discoverer_path = checkpoint_path / "discoverer.pt"
        checkpoint = torch.load(discoverer_path)
        self.discoverer_policy.load_state_dict(checkpoint['policy_state_dict'])
        
        # Load setter
        setter_path = checkpoint_path / "setter"
        self.setter_agent.load(str(setter_path))
        
        # Load performance history
        history_path = checkpoint_path / "performance_history.json"
        with open(history_path, 'r') as f:
            history = json.load(f)
            # Restore tracking state
            self.tracker.discoverer_success_rates = deque(
                history['discoverer_success_rates'], 
                maxlen=self.config.performance_window
            )
            self.tracker.setter_rewards = deque(
                history['setter_rewards'],
                maxlen=self.config.performance_window
            )
        
        self.iteration = history['iteration']
        print(f"Loaded checkpoint from iteration {iteration}")


# Example usage
if __name__ == "__main__":
    config = SelfPlayConfig(
        total_iterations=1000,
        tasks_per_iteration=5,
        episodes_per_task=10,
        setter_train_interval=10,
        checkpoint_interval=50,
        use_wandb=True
    )
    
    trainer = SelfPlayCurriculumTrainer(config)
    trainer.train()