# File: JanusAI/ml/agents/task_setter.py
"""
TaskSetter Agent for Self-Play Curriculum Generation

This agent learns to generate appropriate physics tasks for the LawDiscoverer agent,
creating an open-ended curriculum through adversarial self-play.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

from janus.physics.data.generators import PhysicsTask


@dataclass
class TaskSetterConfig:
    """Configuration for the TaskSetter agent"""
    # Parameter ranges for physics tasks
    mass_range: Tuple[float, float] = (0.1, 10.0)
    length_range: Tuple[float, float] = (0.1, 5.0)
    spring_constant_range: Tuple[float, float] = (0.5, 50.0)
    gravity_range: Tuple[float, float] = (1.0, 20.0)
    damping_range: Tuple[float, float] = (0.0, 2.0)
    
    # Curriculum difficulty targets
    target_success_rate: float = 0.3  # Sweet spot for learning
    success_rate_tolerance: float = 0.1
    
    # Reward shaping parameters
    progress_reward_weight: float = 1.0
    diversity_reward_weight: float = 0.2
    difficulty_reward_weight: float = 0.5
    
    # History tracking
    history_window: int = 20
    
    # Neural network architecture
    hidden_dim: int = 256
    num_layers: int = 3


class TaskParameterNetwork(nn.Module):
    """Neural network that outputs task parameters"""
    
    def __init__(self, input_dim: int, output_dim: int, config: TaskSetterConfig):
        super().__init__()
        self.config = config
        
        layers = []
        prev_dim = input_dim
        
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(prev_dim, config.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(config.hidden_dim)
            ])
            prev_dim = config.hidden_dim
        
        # Output heads for different parameter types
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate heads for different physics parameters
        self.mass_head = nn.Linear(config.hidden_dim, 2)  # m1, m2
        self.length_head = nn.Linear(config.hidden_dim, 2)  # L1, L2
        self.spring_head = nn.Linear(config.hidden_dim, 1)  # k
        self.gravity_head = nn.Linear(config.hidden_dim, 1)  # g
        self.damping_head = nn.Linear(config.hidden_dim, 1)  # b
        
        # Task type selection head
        self.task_type_head = nn.Linear(config.hidden_dim, 5)  # 5 task types
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.shared_layers(x)
        
        # Get bounded parameters using sigmoid activation
        masses = torch.sigmoid(self.mass_head(features))
        lengths = torch.sigmoid(self.length_head(features))
        spring_k = torch.sigmoid(self.spring_head(features))
        gravity = torch.sigmoid(self.gravity_head(features))
        damping = torch.sigmoid(self.damping_head(features))
        
        # Task type as categorical distribution
        task_type_logits = self.task_type_head(features)
        
        return {
            'masses': masses,
            'lengths': lengths,
            'spring_constant': spring_k,
            'gravity': gravity,
            'damping': damping,
            'task_type_logits': task_type_logits
        }


class TaskSetterEnv(gym.Env):
    """
    Environment for the TaskSetter agent.
    
    The agent observes the LawDiscoverer's performance history and
    outputs task parameters. Rewards are based on maintaining optimal
    difficulty and promoting diversity.
    """
    
    def __init__(self, config: TaskSetterConfig, discoverer_env):
        super().__init__()
        self.config = config
        self.discoverer_env = discoverer_env
        
        # Define observation space (performance history + current state)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(config.history_window * 4 + 10,),  # History stats + current state
            dtype=np.float32
        )
        
        # Define action space (continuous parameters)
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(10,),  # All normalized parameters
            dtype=np.float32
        )
        
        # Performance tracking
        self.performance_history = []
        self.task_history = []
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.performance_history = []
        self.task_history = []
        self.current_step = 0
        return self._get_observation(), {}
    
    def step(self, action):
        # Convert normalized action to task parameters
        task_params = self._action_to_task_params(action)
        
        # Create and evaluate task with LawDiscoverer
        task = self._create_physics_task(task_params)
        discoverer_performance = self._evaluate_discoverer_on_task(task)
        
        # Update history
        self.performance_history.append(discoverer_performance)
        self.task_history.append(task_params)
        
        # Calculate reward
        reward = self._calculate_reward(discoverer_performance, task_params)
        
        # Check termination
        done = self.current_step >= 100  # Episode length
        
        self.current_step += 1
        
        return self._get_observation(), reward, done, False, {
            'task_params': task_params,
            'discoverer_performance': discoverer_performance
        }
    
    def _action_to_task_params(self, action: np.ndarray) -> Dict[str, float]:
        """Convert normalized action to actual task parameters"""
        config = self.config
        
        return {
            'm1': action[0] * (config.mass_range[1] - config.mass_range[0]) + config.mass_range[0],
            'm2': action[1] * (config.mass_range[1] - config.mass_range[0]) + config.mass_range[0],
            'L1': action[2] * (config.length_range[1] - config.length_range[0]) + config.length_range[0],
            'L2': action[3] * (config.length_range[1] - config.length_range[0]) + config.length_range[0],
            'k': action[4] * (config.spring_constant_range[1] - config.spring_constant_range[0]) + config.spring_constant_range[0],
            'g': action[5] * (config.gravity_range[1] - config.gravity_range[0]) + config.gravity_range[0],
            'b': action[6] * (config.damping_range[1] - config.damping_range[0]) + config.damping_range[0],
            'task_type': int(action[7] * 5)  # 5 task types
        }
    
    def _create_physics_task(self, params: Dict[str, float]) -> PhysicsTask:
        """Create a PhysicsTask instance from parameters"""
        task_types = [
            "harmonic_oscillator",
            "pendulum",
            "double_pendulum",
            "spring_pendulum",
            "coupled_oscillators"
        ]
        
        task_type = task_types[params['task_type']]
        
        # Create custom task with specified parameters
        # This would interface with the PhysicsTaskDistribution
        # For now, return a placeholder
        return PhysicsTask(
            name=f"custom_{task_type}",
            domain="mechanics",
            difficulty=0.5,  # Will be learned
            variables=["x", "v", "a", "t"],
            parameters=params,
            conserved_quantities=["energy"],
            symmetries=["time_translation"]
        )
    
    def _evaluate_discoverer_on_task(self, task: PhysicsTask) -> Dict[str, float]:
        """Run the LawDiscoverer on the task and return performance metrics"""
        # This would actually run the discoverer agent
        # For now, return mock performance
        return {
            'success_rate': np.random.uniform(0, 1),
            'final_reward': np.random.uniform(-1, 1),
            'expression_complexity': np.random.uniform(0, 20),
            'convergence_time': np.random.uniform(10, 100)
        }
    
    def _calculate_reward(self, performance: Dict[str, float], task_params: Dict[str, float]) -> float:
        """
        Calculate reward for the TaskSetter based on:
        1. Maintaining optimal difficulty (not too easy, not too hard)
        2. Promoting task diversity
        3. Encouraging learning progress
        """
        config = self.config
        
        # 1. Difficulty reward: peaked around target success rate
        success_rate = performance['success_rate']
        difficulty_reward = np.exp(
            -((success_rate - config.target_success_rate) ** 2) / 
            (2 * config.success_rate_tolerance ** 2)
        )
        
        # 2. Diversity reward: penalize similar consecutive tasks
        diversity_reward = 0.0
        if len(self.task_history) > 1:
            prev_params = self.task_history[-2]
            param_diff = sum(
                abs(task_params[k] - prev_params[k]) 
                for k in task_params if k in prev_params
            )
            diversity_reward = np.tanh(param_diff / 10.0)
        
        # 3. Progress reward: reward if discoverer is learning
        progress_reward = 0.0
        if len(self.performance_history) > 5:
            recent_performance = [p['success_rate'] for p in self.performance_history[-5:]]
            progress_reward = np.mean(np.diff(recent_performance))
        
        # Combine rewards
        total_reward = (
            config.difficulty_reward_weight * difficulty_reward +
            config.diversity_reward_weight * diversity_reward +
            config.progress_reward_weight * progress_reward
        )
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation from performance history and current state
        """
        obs = []
        
        # Recent performance statistics
        if self.performance_history:
            recent = self.performance_history[-self.config.history_window:]
            
            # Extract statistics
            success_rates = [p['success_rate'] for p in recent]
            rewards = [p['final_reward'] for p in recent]
            
            obs.extend([
                np.mean(success_rates),
                np.std(success_rates),
                np.mean(rewards),
                np.std(rewards),
                len(recent) / self.config.history_window  # Fill ratio
            ])
        else:
            obs.extend([0.5, 0.0, 0.0, 0.0, 0.0])  # Default values
        
        # Pad observation to fixed size
        while len(obs) < self.observation_space.shape[0]:
            obs.append(0.0)
        
        return np.array(obs[:self.observation_space.shape[0]], dtype=np.float32)


class TaskSetterAgent:
    """
    High-level agent that manages the TaskSetter's learning process
    """
    
    def __init__(self, config: TaskSetterConfig, discoverer_env):
        self.config = config
        self.env = TaskSetterEnv(config, discoverer_env)
        
        # Initialize PPO agent
        self.agent = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
    def train(self, total_timesteps: int):
        """Train the TaskSetter agent"""
        self.agent.learn(total_timesteps=total_timesteps)
        
    def propose_task(self, observation: np.ndarray) -> Dict[str, float]:
        """Propose a new task given current state"""
        action, _ = self.agent.predict(observation, deterministic=False)
        return self.env._action_to_task_params(action)
    
    def save(self, path: str):
        """Save the trained agent"""
        self.agent.save(path)
        
    def load(self, path: str):
        """Load a trained agent"""
        self.agent = PPO.load(path, env=self.env)