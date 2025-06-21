"""
Integrated Training Pipeline for Janus
=====================================

Integrates various components into cohesive, end-to-end discovery 
and interpretability pipelines.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
import logging

from janus_ai.core.grammar import ProgressiveGrammar
from janus_ai.core.expressions import Variable
from janus_ai.environments.base import SymbolicDiscoveryEnv
from janus_ai.ml.networks import HypothesisNet
from janus_ai.ml.training import PPOTrainer
from janus_ai.config.models import JanusConfig
from janus_ai.utils.logging import ExperimentLogger

# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

logger = logging.getLogger(__name__)


class JanusTrainer:
    """
    Base trainer that orchestrates the training pipeline for Janus.
    """
    
    def __init__(self, config: JanusConfig):
        self.config = config
        
        # Setup directories
        self._setup_directories()
        
        # Initialize components
        self.grammar = ProgressiveGrammar()
        self.variables = []
        self.env = None
        self.trainer = None
        
        # Initialize logging
        self.logger = ExperimentLogger(
            experiment_name=f"janus_{config.training_mode}_{int(time.time())}",
            log_dir=config.results_dir
        )
        
        # Initialize W&B if configured
        if config.wandb_project and HAS_WANDB:
            wandb.init(
                project=config.wandb_project,
                config=config.model_dump(),
                name=f"janus_{config.training_mode}_{int(time.time())}"
            )
    
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.data_dir, 
                        self.config.checkpoint_dir, 
                        self.config.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, 
                    data_path: Optional[str] = None,
                    generate_synthetic: bool = True) -> np.ndarray:
        """Prepare training data."""
        
        if data_path and Path(data_path).exists():
            # Load real data
            data = np.load(data_path)
            logger.info(f"Loaded data from {data_path}: shape {data.shape}")
        
        elif generate_synthetic:
            # Generate synthetic physics data
            data = self._generate_synthetic_data()
            logger.info(f"Generated synthetic {self.config.target_phenomena} data")
        
        else:
            raise ValueError("No data provided")
        
        # Discover variables from data
        logger.info("Discovering variables...")
        self.variables = self.grammar.discover_variables(data)
        logger.info(f"Discovered {len(self.variables)} variables")
        
        return data
    
    def _generate_synthetic_data(self) -> np.ndarray:
        """Generate synthetic physics data based on target phenomena."""
        
        n_samples = 2000
        
        if self.config.target_phenomena == "harmonic_oscillator":
            t = np.linspace(0, 20, n_samples)
            x = np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(n_samples)
            v = 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(n_samples)
            energy = 0.5 * (x**2 + v**2)
            data = np.column_stack([x, v, energy])
            
        elif self.config.target_phenomena == "pendulum":
            t = np.linspace(0, 20, n_samples)
            theta = 0.2 * np.sin(np.sqrt(9.81) * t) + 0.02 * np.random.randn(n_samples)
            omega = 0.2 * np.sqrt(9.81) * np.cos(np.sqrt(9.81) * t) + 0.02 * np.random.randn(n_samples)
            energy = 0.5 * omega**2 + 9.81 * (1 - np.cos(theta))
            data = np.column_stack([theta, omega, energy])
            
        elif self.config.target_phenomena == "kepler":
            # Circular orbit
            t = np.linspace(0, 10, n_samples)
            r = 1.0 + 0.01 * np.random.randn(n_samples)
            theta = t + 0.01 * np.random.randn(n_samples)
            vr = 0.01 * np.random.randn(n_samples)
            vtheta = 1.0 / r + 0.01 * np.random.randn(n_samples)
            energy = 0.5 * (vr**2 + r**2 * vtheta**2) - 1.0 / r
            angular_momentum = r**2 * vtheta
            data = np.column_stack([r, theta, vr, vtheta, energy, angular_momentum])
            
        else:
            # Default: simple polynomial data
            x = np.linspace(-2, 2, n_samples)
            y = 2 * x**2 + 3 * x + 1 + 0.1 * np.random.randn(n_samples)
            data = np.column_stack([x, y])
        
        # Save generated data
        save_path = Path(self.config.data_dir) / f"{self.config.target_phenomena}_synthetic.npy"
        np.save(save_path, data)
        
        return data
    
    def create_environment(self, data: np.ndarray) -> SymbolicDiscoveryEnv:
        """Create the discovery environment."""
        
        reward_config = self.config.reward_config or {
            'completion_bonus': 0.1,
            'mse_weight': -1.0,
            'complexity_penalty': -0.01,
            'depth_penalty': -0.001
        }
        
        env_config = {
            'grammar': self.grammar,
            'target_data': data,
            'variables': self.variables,
            'max_depth': self.config.max_depth,
            'max_complexity': self.config.max_complexity,
            'reward_config': reward_config.model_dump() if hasattr(reward_config, 'model_dump') else reward_config
        }
        
        # Create environment
        env = SymbolicDiscoveryEnv(**env_config)
        
        return env
    
    def create_trainer(self):
        """Create the trainer."""
        
        # Determine observation dimension
        obs_dim = self.env.observation_space.shape[0]
        
        # Create policy network
        policy = HypothesisNet(
            observation_dim=obs_dim,
            action_dim=self.env.action_space.n,
            hidden_dim=256,
            encoder_type='transformer',
            grammar=self.grammar
        )
        
        # Create trainer
        trainer = PPOTrainer(policy, self.env)
        
        return trainer
    
    def train(self):
        """Main training loop."""
        
        logger.info(f"Starting {self.config.training_mode} training...")
        logger.info(f"Total timesteps: {self.config.total_timesteps}")
        
        # Run training
        self.trainer.train(
            total_timesteps=self.config.total_timesteps,
            rollout_length=self.config.ppo_rollout_length,
            n_epochs=self.config.ppo_n_epochs,
            log_interval=self.config.log_interval
        )
        
        # Save final model
        self._save_checkpoint("final")
    
    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{name}_model.pt"
        
        checkpoint = {
            'policy_state_dict': self.trainer.policy.state_dict(),
            'grammar_state': self.grammar.export_grammar_state(),
            'config': self.config.model_dump(),
            'variables': [(v.name, v.index, v.properties) for v in self.variables]
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        if self.trainer and self.trainer.policy:
            self.trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        if 'grammar_state' in checkpoint:
            self.grammar.import_grammar_state(checkpoint['grammar_state'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


class AdvancedJanusTrainer(JanusTrainer):
    """
    Advanced trainer with enhanced features for complex training scenarios.
    """
    
    def __init__(self, config: JanusConfig):
        super().__init__(config)
        
        # Advanced components placeholders
        self.curriculum_manager = None
        self.distributed_trainer = None
        
        # Initialize based on mode
        self._initialize_advanced_features()
    
    def _initialize_advanced_features(self):
        """Initialize advanced training features."""
        
        # Setup distributed training if available
        if self.config.training_mode == "distributed" and HAS_RAY:
            if not ray.is_initialized():
                ray.init(num_cpus=self.config.num_workers * 2, 
                        num_gpus=self.config.num_gpus)
        
        # Setup curriculum learning if enabled
        if self.config.use_curriculum:
            try:
                from janus_ai.environments.enhanced import CurriculumManager
                self.curriculum_manager = CurriculumManager(
                    stages=self.config.curriculum_stages
                )
            except ImportError:
                logger.warning("CurriculumManager not available")
    
    def create_environment(self, data: np.ndarray) -> SymbolicDiscoveryEnv:
        """Create enhanced environment for advanced training."""
        
        # Try to use enhanced environment
        try:
            from janus_ai.environments.enhanced import EnhancedSymbolicDiscoveryEnv
            
            reward_config = self.config.reward_config or {
                'completion_bonus': 0.1,
                'mse_weight': -1.0,
                'complexity_penalty': -0.01,
                'depth_penalty': -0.001
            }
            
            env_config = {
                'grammar': self.grammar,
                'target_data': data,
                'variables': self.variables,
                'max_depth': self.config.max_depth,
                'max_complexity': self.config.max_complexity,
                'reward_config': reward_config.model_dump() if hasattr(reward_config, 'model_dump') else reward_config
            }
            
            env = EnhancedSymbolicDiscoveryEnv(**env_config)
            logger.info("Using EnhancedSymbolicDiscoveryEnv")
            
        except ImportError:
            logger.warning("EnhancedSymbolicDiscoveryEnv not available, using standard environment")
            env = super().create_environment(data)
        
        # Wrap with curriculum if enabled
        if self.curriculum_manager:
            env = self.curriculum_manager.wrap_environment(env)
            
        return env
    
    def train(self):
        """Advanced training loop with phases."""
        
        if self.config.training_mode == "advanced":
            self._train_advanced()
        else:
            super().train()
    
    def _train_advanced(self):
        """Advanced training with multiple phases."""
        
        timesteps_per_phase = self.config.total_timesteps // 4
        
        # Phase 1: Initial exploration
        logger.info("Phase 1: Initial Exploration")
        self.trainer.train(
            total_timesteps=timesteps_per_phase,
            rollout_length=1024,
            n_epochs=5,
            log_interval=self.config.log_interval
        )
        
        # Phase 2: Focused training
        logger.info("Phase 2: Focused Training")
        self.trainer.train(
            total_timesteps=timesteps_per_phase,
            rollout_length=2048,
            n_epochs=10,
            log_interval=self.config.log_interval
        )
        
        # Phase 3: Fine-tuning
        logger.info("Phase 3: Fine-tuning")
        self.trainer.train(
            total_timesteps=timesteps_per_phase,
            rollout_length=4096,
            n_epochs=15,
            log_interval=self.config.log_interval
        )
        
        # Phase 4: Final optimization
        logger.info("Phase 4: Final Optimization")
        self.trainer.train(
            total_timesteps=timesteps_per_phase,
            rollout_length=4096,
            n_epochs=20,
            log_interval=self.config.log_interval
        )
        
        # Save final model
        self._save_checkpoint("final_advanced")
        
        # Generate final report
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate a final training report."""
        
        report = {
            'training_mode': self.config.training_mode,
            'total_timesteps': self.config.total_timesteps,
            'target_phenomena': self.config.target_phenomena,
            'final_reward': self.trainer.episode_rewards[-1] if self.trainer.episode_rewards else 0,
            'num_variables': len(self.variables),
            'max_expression_depth': self.config.max_depth,
            'max_expression_complexity': self.config.max_complexity
        }
        
        # Save report
        report_path = Path(self.config.results_dir) / "final_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved to {report_path}")
        
        # Log to W&B if available
        if self.config.wandb_project and HAS_WANDB:
            wandb.log(report)
            wandb.finish()
    
    def run_experiment_suite(self):
        """Run validation experiments."""
        
        logger.info("Running experiment suite...")
        
        # Import experiment runner
        try:
            from janus_ai.experiments.runner import ExperimentRunner
            
            runner = ExperimentRunner(
                base_dir=self.config.results_dir,
                use_wandb=bool(self.config.wandb_project),
                strict_mode=self.config.strict_mode
            )
            
            # Run basic validation
            from janus_ai.experiments.configs import HarmonicOscillatorConfig
            config = HarmonicOscillatorConfig()
            
            results = runner.run_single_experiment(config)
            logger.info(f"Validation results: {results}")
            
        except ImportError as e:
            logger.warning(f"Could not run experiment suite: {e}")


def create_trainer(config: JanusConfig) -> JanusTrainer:
    """Factory function to create appropriate trainer based on config."""
    
    if config.training_mode == "advanced":
        return AdvancedJanusTrainer(config)
    else:
        return JanusTrainer(config)


def run_training_pipeline(config: JanusConfig):
    """Run the complete training pipeline."""
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Prepare data
    data = trainer.prepare_data(generate_synthetic=True)
    
    # Create environment
    trainer.env = trainer.create_environment(data)
    
    # Create trainer
    trainer.trainer = trainer.create_trainer()
    
    # Run training
    trainer.train()
    
    # Run validation if configured
    if config.run_validation_suite and hasattr(trainer, 'run_experiment_suite'):
        trainer.run_experiment_suite()
    
    logger.info("Training pipeline complete!")
    
    # Cleanup
    if HAS_RAY and ray.is_initialized():
        ray.shutdown()
    
    return trainer