# File: JanusAI/ml/training/enhanced_ppo_trainer.py
"""
Enhanced PPO Trainer with Advanced Exploration Strategies

Integrates:
1. MaxInfoRL (Information Gain)
2. PreND (Pre-trained Network Distillation)
3. LLM-Driven Exploration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from janus.ml.networks.dynamics_ensemble import DynamicsEnsemble
from janus.ml.rewards.intrinsic_rewards import (
    InformationGainReward,
    PreNDIntrinsicReward,
    GoalMatchingReward,
    CombinedIntrinsicReward
)
from janus.utils.ai.llm_exploration import (
    LLMGoalGenerator,
    ExplorationContext,
    AdaptiveLLMExploration
)
from janus.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus.physics.data.generators import PhysicsTask


@dataclass
class EnhancedPPOConfig:
    """Configuration for enhanced PPO training."""
    
    # Base PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Exploration parameters
    use_information_gain: bool = True
    use_prend: bool = True
    use_llm_goals: bool = True
    
    # Information gain parameters
    ensemble_size: int = 5
    ensemble_hidden_dim: int = 128
    info_gain_scale: float = 0.5
    
    # PreND parameters
    prend_model_name: str = "clip-vit-base-patch32"
    prend_scale: float = 0.3
    
    # LLM parameters
    llm_model: str = "gpt-4"
    llm_exploration_rate: float = 0.2
    llm_goal_duration: int = 100
    
    # Combined reward weights
    extrinsic_weight: float = 1.0
    intrinsic_weight: float = 0.5
    adaptive_weights: bool = True
    
    # Training parameters
    total_timesteps: int = 1000000
    checkpoint_interval: int = 10000
    log_interval: int = 10
    
    # Environment parameters
    max_episode_steps: int = 200
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ExpressionEmbedder(nn.Module):
    """
    Embeds symbolic expressions into vector representations.
    
    Used for dynamics ensemble and other exploration mechanisms.
    """
    
    def __init__(self, vocab_size: int = 100, embedding_dim: int = 128, 
                 hidden_dim: int = 256):
        super().__init__()
        
        # Token embeddings for expression elements
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 200, embedding_dim)  # Max sequence length 200
        )
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, hidden_dim)
        
    def forward(self, expression_tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed expression tokens into a fixed-size vector.
        
        Args:
            expression_tokens: Tensor of token indices (batch_size, seq_len)
            
        Returns:
            Expression embedding (batch_size, hidden_dim)
        """
        # Get token embeddings
        embeddings = self.token_embedding(expression_tokens)
        
        # Add positional encoding
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.positional_encoding[:, :seq_len, :]
        
        # Apply transformer
        encoded = self.transformer(embeddings)
        
        # Pool over sequence dimension (mean pooling)
        pooled = encoded.mean(dim=1)
        
        # Project to output dimension
        return self.output_proj(pooled)


class EnhancedPPOTrainer:
    """
    PPO Trainer enhanced with advanced exploration strategies.
    """
    
    def __init__(self, 
                 env: SymbolicDiscoveryEnv,
                 config: EnhancedPPOConfig):
        self.env = env
        self.config = config
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Initialize expression embedder
        self.expression_embedder = ExpressionEmbedder().to(config.device)
        
        # Initialize exploration components
        self._init_exploration_components()
        
        # Create PPO agent
        self.agent = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            verbose=1,
            tensorboard_log="./ppo_enhanced_tensorboard/",
            device=config.device
        )
        
        # Custom callback for intrinsic rewards
        self.intrinsic_callback = IntrinsicRewardCallback(
            trainer=self,
            verbose=1
        )
        
        # Statistics tracking
        self.episode_rewards = []
        self.intrinsic_rewards = []
        self.discovered_expressions = []
        self.training_start_time = time.time()
        
    def _init_exploration_components(self):
        """Initialize all exploration components."""
        components = {}
        
        # 1. Information Gain (MaxInfoRL)
        if self.config.use_information_gain:
            # Get embedding dimension from env observation space
            obs_dim = self.env.observation_space.shape[0]
            
            # Create dynamics ensemble
            self.dynamics_ensemble = DynamicsEnsemble(
                num_models=self.config.ensemble_size,
                input_dim=256,  # Output dim of expression embedder
                hidden_dim=self.config.ensemble_hidden_dim,
                device=self.config.device
            )
            
            # Create information gain reward
            components['information_gain'] = InformationGainReward(
                ensemble=self.dynamics_ensemble,
                scale_factor=self.config.info_gain_scale
            )
        
        # 2. Pre-trained Network Distillation (PreND)
        if self.config.use_prend:
            # Load pre-trained model
            if "clip" in self.config.prend_model_name:
                from transformers import CLIPModel
                target_model = CLIPModel.from_pretrained(self.config.prend_model_name)
            else:
                # Default to a simple pre-trained model
                target_model = self._create_default_target_model()
            
            components['prend'] = PreNDIntrinsicReward(
                target_net=target_model,
                reward_scale=self.config.prend_scale,
                device=self.config.device
            )
        
        # 3. LLM-Driven Goal Matching
        if self.config.use_llm_goals:
            # Initialize LLM generator
            self.llm_generator = LLMGoalGenerator(
                model_name=self.config.llm_model,
                api_type="openai"  # Adjust based on your setup
            )
            
            # Initialize goal matching reward
            components['goal_matching'] = GoalMatchingReward(
                grammar=self.env.grammar,
                max_reward=10.0
            )
            
            # Initialize adaptive exploration
            self.adaptive_exploration = AdaptiveLLMExploration(
                goal_generator=self.llm_generator,
                grammar=self.env.grammar,
                initial_exploration_rate=self.config.llm_exploration_rate
            )
        
        # Create combined intrinsic reward
        self.intrinsic_reward = CombinedIntrinsicReward(
            reward_components=components,
            adaptive_weights=self.config.adaptive_weights
        )
    
    def _create_default_target_model(self) -> nn.Module:
        """Create a default target model for PreND."""
        class SimpleTargetModel(nn.Module):
            def __init__(self, input_dim: int = 512, output_dim: int = 256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.LayerNorm(512),
                    nn.Linear(512, output_dim),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.net(x)
        
        return SimpleTargetModel()
    
    def calculate_total_reward(self, 
                             extrinsic_reward: float,
                             obs: np.ndarray,
                             info: Dict[str, Any]) -> float:
        """
        Calculate total reward combining extrinsic and intrinsic components.
        
        Args:
            extrinsic_reward: Environment reward
            obs: Current observation
            info: Additional info from environment
            
        Returns:
            Total reward
        """
        # Prepare inputs for intrinsic reward calculation
        intrinsic_kwargs = {}
        
        # Get expression embedding if needed
        if self.config.use_information_gain:
            # Convert observation to expression tokens (simplified)
            expression_tokens = self._obs_to_expression_tokens(obs)
            expression_tokens = torch.tensor(expression_tokens).unsqueeze(0).to(self.config.device)
            
            # Get embedding
            with torch.no_grad():
                expression_embedding = self.expression_embedder(expression_tokens)
            
            intrinsic_kwargs['expression_embedding'] = expression_embedding
        
        # Add state representation for PreND
        if self.config.use_prend:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            intrinsic_kwargs['state_representation'] = state_tensor
        
        # Add generated expression for goal matching
        if self.config.use_llm_goals and 'expression' in info:
            intrinsic_kwargs['generated_expression'] = info['expression']
        
        # Calculate intrinsic reward
        intrinsic_result = self.intrinsic_reward.calculate_reward(
            return_components=True,
            **intrinsic_kwargs
        )
        
        intrinsic_value = intrinsic_result['total_reward']
        
        # Combine rewards
        total_reward = (self.config.extrinsic_weight * extrinsic_reward + 
                       self.config.intrinsic_weight * intrinsic_value)
        
        # Track statistics
        self.intrinsic_rewards.append(intrinsic_value)
        
        return total_reward
    
    def _obs_to_expression_tokens(self, obs: np.ndarray) -> List[int]:
        """
        Convert observation to expression tokens.
        
        This is a simplified version - in practice, you'd parse the
        actual expression tree from the observation.
        """
        # Simple tokenization based on observation values
        # This should be replaced with actual expression parsing
        tokens = []
        for i, val in enumerate(obs[:20]):  # Use first 20 values
            if val > 0.5:
                tokens.append(int(val * 10) % 50)  # Map to token space
        
        # Pad or truncate to fixed length
        max_len = 50
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))  # Padding token is 0
        else:
            tokens = tokens[:max_len]
        
        return tokens
    
    def update_exploration_components(self, batch_data: Dict[str, Any]):
        """Update exploration components with collected data."""
        # Update dynamics ensemble
        if self.config.use_information_gain and 'expression_embeddings' in batch_data:
            self.intrinsic_reward.components['information_gain'].update(
                expression_embeddings=batch_data['expression_embeddings'],
                true_rewards=batch_data['rewards']
            )
        
        # Update PreND predictor
        if self.config.use_prend and 'observations' in batch_data:
            self.intrinsic_reward.components['prend'].update(
                state_representations=batch_data['observations']
            )
        
        # Update LLM goal if needed
        if self.config.use_llm_goals and self.adaptive_exploration.goal_steps_remaining <= 0:
            # Check if we should set a new goal
            if self.adaptive_exploration.should_use_llm_goal():
                context = self._build_exploration_context()
                self.adaptive_exploration.set_new_goal(context, self.config.llm_goal_duration)
                
                # Update goal matching reward
                if self.adaptive_exploration.current_goal:
                    goal_expr_str = str(self.adaptive_exploration.current_goal)
                    self.intrinsic_reward.components['goal_matching'].set_goal(goal_expr_str)
    
    def _build_exploration_context(self) -> ExplorationContext:
        """Build context for LLM exploration."""
        # Get recent discoveries
        recent_discoveries = []
        for disc in self.discovered_expressions[-10:]:
            recent_discoveries.append({
                'expression': str(disc.get('expression', '')),
                'reward': disc.get('reward', 0.0)
            })
        
        # Get task info from environment
        task_info = getattr(self.env, 'task_info', {})
        
        return ExplorationContext(
            domain=task_info.get('domain', 'mechanics'),
            variables=self.env.variable_names if hasattr(self.env, 'variable_names') else ['x', 'y'],
            variable_descriptions=task_info.get('variable_descriptions', {}),
            discovered_expressions=recent_discoveries,
            failed_attempts=[],  # Could track low-reward expressions
            performance_history=[r for r, _ in self.episode_rewards[-20:]],
            metadata={'system_description': task_info.get('name', 'Unknown system')}
        )
    
    def train(self):
        """Main training loop with enhanced exploration."""
        print("Starting Enhanced PPO Training")
        print(f"Exploration strategies enabled:")
        print(f"  - Information Gain: {self.config.use_information_gain}")
        print(f"  - PreND: {self.config.use_prend}")
        print(f"  - LLM Goals: {self.config.use_llm_goals}")
        print("=" * 50)
        
        # Train the agent
        self.agent.learn(
            total_timesteps=self.config.total_timesteps,
            callback=self.intrinsic_callback,
            log_interval=self.config.log_interval,
            tb_log_name="enhanced_ppo"
        )
        
        # Save final statistics
        self._save_training_statistics()
    
    def _save_training_statistics(self):
        """Save training statistics and discovered expressions."""
        stats = {
            'training_time': time.time() - self.training_start_time,
            'total_episodes': len(self.episode_rewards),
            'discovered_expressions': self.discovered_expressions,
            'final_episode_rewards': self.episode_rewards[-100:],
            'intrinsic_reward_stats': self.intrinsic_reward.get_statistics(),
            'config': self.config.__dict__
        }
        
        # Save to file
        save_path = Path("training_results") / f"enhanced_ppo_{int(time.time())}.json"
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\nTraining complete! Statistics saved to {save_path}")
        print(f"Total discovered expressions: {len(self.discovered_expressions)}")
        print(f"Best discovery reward: {max([d['reward'] for d in self.discovered_expressions]) if self.discovered_expressions else 0}")


class IntrinsicRewardCallback(BaseCallback):
    """
    Custom callback for handling intrinsic rewards and exploration updates.
    """
    
    def __init__(self, trainer: EnhancedPPOTrainer, verbose: int = 0):
        super().__init__(verbose)
        self.trainer = trainer
        self.episode_rewards = []
        self.episode_intrinsic_rewards = []
        
    def _on_step(self) -> bool:
        # Get current info
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0])
        observations = self.locals.get('obs_tensor', None)
        
        # Calculate intrinsic rewards
        for i, (info, reward) in enumerate(zip(infos, rewards)):
            if observations is not None:
                obs = observations[i].cpu().numpy()
                total_reward = self.trainer.calculate_total_reward(
                    extrinsic_reward=reward,
                    obs=obs,
                    info=info
                )
                
                # Override the reward
                self.locals['rewards'][i] = total_reward
        
        # Check for discovered expressions
        for info in infos:
            if 'expression' in info and info.get('discovery_complete', False):
                self.trainer.discovered_expressions.append({
                    'expression': info['expression'],
                    'reward': info.get('reward', 0),
                    'mse': info.get('mse', float('inf')),
                    'timestep': self.num_timesteps
                })
        
        # Update exploration components periodically
        if self.n_calls % 1000 == 0:
            # Prepare batch data (simplified)
            batch_data = {
                'observations': observations,
                'rewards': torch.tensor(rewards)
            }
            self.trainer.update_exploration_components(batch_data)
        
        return True
    
    def _on_rollout_end(self):
        """Called at the end of a rollout."""
        # Log statistics
        if self.trainer.intrinsic_rewards:
            recent_intrinsic = np.mean(self.trainer.intrinsic_rewards[-100:])
            self.logger.record('exploration/mean_intrinsic_reward', recent_intrinsic)
        
        # Log exploration component statistics
        stats = self.trainer.intrinsic_reward.get_statistics()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.logger.record(f'exploration/{key}', value)


# Example usage
if __name__ == "__main__":
    # Create environment
    from janus.physics.data.generators import PhysicsTaskDistribution
    
    task_dist = PhysicsTaskDistribution()
    task = task_dist.sample_task()
    data = task.generate_data(200)
    
    env = SymbolicDiscoveryEnv(
        data=data,
        target_expr="unknown",
        max_depth=10,
        max_complexity=30
    )
    
    # Configure enhanced training
    config = EnhancedPPOConfig(
        total_timesteps=100000,
        use_information_gain=True,
        use_prend=True,
        use_llm_goals=True,
        llm_model="gpt-3.5-turbo",  # Use a faster model for demo
        ensemble_size=3,
        info_gain_scale=0.5,
        prend_scale=0.3
    )
    
    # Create and run trainer
    trainer = EnhancedPPOTrainer(env, config)
    trainer.train()