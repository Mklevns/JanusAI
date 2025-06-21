"""
Experiment Configuration Models
===============================

Defines data models for structuring and validating experiment configurations.
This ensures consistency and ease of use for setting up various symbolic
discovery experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ExperimentConfig:
    """
    A comprehensive data class for structuring experiment configurations.
    This provides type-hinted and validated settings for training, environment,
    policy, and logging.
    """
    # General experiment settings
    experiment_name: str = "default_janus_experiment"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    total_timesteps: int = 1_000_000
    max_episode_steps: int = 100
    log_interval: int = 100 # Steps per log update
    eval_interval: int = 10000 # Timesteps per evaluation run
    checkpoint_interval: int = 100000 # Timesteps per checkpoint save

    # Environment parameters
    env_type: str = "SymbolicDiscoveryEnv" # E.g., "SymbolicDiscoveryEnv", "PhysicsEnvironment"
    env_config: Dict[str, Any] = field(default_factory=dict) # Specific config for the chosen environment

    # Policy (Model) parameters
    policy_type: str = "HypothesisNet" # E.g., "HypothesisNet", "AIHypothesisNet"
    policy_config: Dict[str, Any] = field(default_factory=dict) # Specific config for the policy model

    # Trainer parameters (e.g., PPO, MAML)
    trainer_type: str = "PPOTrainer"
    trainer_config: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 3e-4,
        "ppo_epochs": 10,
        "rollout_length": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
    })

    # Intrinsic Reward parameters (if applicable)
    intrinsic_rewards_enabled: bool = False
    intrinsic_reward_config: Dict[str, Any] = field(default_factory=lambda: {
        "novelty_weight": 0.3,
        "diversity_weight": 0.2,
        "complexity_growth_weight": 0.1,
        "conservation_weight": 0.4,
        "conservation_types": ["energy", "momentum"],
        "apply_entropy_penalty": False,
        "entropy_penalty_weight": 0.1,
        "history_size": 1000
    })

    # Adaptive Training / Curriculum parameters
    adaptive_training_enabled: bool = False
    adaptive_training_config: Dict[str, Any] = field(default_factory=lambda: {
        "stagnation_threshold": 0.01,
        "breakthrough_threshold": 0.05,
        "history_length": 50,
        "curriculum_manager_config": {
            "initial_difficulty": 0.0,
            "max_difficulty": 1.0,
            "difficulty_increment": 0.05,
            "steps_per_increment": 100,
            "adaptive": False
        }
    })

    # Distributed Training parameters (if applicable)
    distributed_training_enabled: bool = False
    distributed_config: Dict[str, Any] = field(default_factory=lambda: {
        "num_ray_workers": 1,
        "ray_init_kwargs": {},
        # Additional Ray/RLlib specific configs can go here
    })

    def __post_init__(self):
        """Perform validation and setup after initialization."""
        # Ensure that device is a valid PyTorch device
        if not torch.cuda.is_available() and self.device == "cuda":
            self.device = "cpu"
            print("Warning: CUDA not available, switching device to 'cpu'.")
        
        # Ensure log/checkpoint intervals are sensible
        if self.log_interval <= 0: self.log_interval = 1
        if self.eval_interval <= 0: self.eval_interval = self.log_interval
        if self.checkpoint_interval <= 0: self.checkpoint_interval = self.eval_interval
        if self.eval_interval < self.log_interval: self.eval_interval = self.log_interval
        if self.checkpoint_interval < self.eval_interval: self.checkpoint_interval = self.eval_interval

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Creates an instance of ExperimentConfig from a dictionary."""
        # This requires careful handling if nested dataclasses are present,
        # but for simple dicts in fields, direct passing works.
        # For robustness, could recursively call from_dict for known nested dataclasses.
        # For now, assumes top-level conversion.
        instance = cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        # Manually update dict fields that have default_factory or complex nested structure if needed
        # For example, if policy_config was a nested dataclass, you'd do:
        # if 'policy_config' in data and isinstance(data['policy_config'], dict):
        #    instance.policy_config = PolicyConfig.from_dict(data['policy_config'])
        return instance


if __name__ == "__main__":
    # Test ExperimentConfig creation and usage

    print("--- Testing ExperimentConfig ---")

    # Default configuration
    default_config = ExperimentConfig()
    print("\nDefault Config:")
    print(json.dumps(default_config.to_dict(), indent=2))
    assert default_config.total_timesteps == 1_000_000
    assert default_config.device == ("cuda" if torch.cuda.is_available() else "cpu")

    # Custom configuration
    custom_env_config = {
        "physics_task_name": "harmonic_oscillator_energy",
        "noise_level": 0.05
    }
    custom_policy_config = {
        "encoder_type": "transformer",
        "hidden_dim": 512
    }
    custom_trainer_config = {
        "learning_rate": 1e-3,
        "ppo_epochs": 15,
        "entropy_coef": 0.005
    }

    my_exp_config = ExperimentConfig(
        experiment_name="my_custom_experiment",
        total_timesteps=500_000,
        env_type="PhysicsEnvironment",
        env_config=custom_env_config,
        policy_config=custom_policy_config,
        trainer_config=custom_trainer_config,
        intrinsic_rewards_enabled=True,
        intrinsic_reward_config={"novelty_weight": 0.5, "conservation_weight": 0.1},
        adaptive_training_enabled=True,
        distributed_training_enabled=True,
        distributed_config={"num_ray_workers": 4}
    )

    print("\nCustom Config:")
    print(json.dumps(my_exp_config.to_dict(), indent=2))

    assert my_exp_config.experiment_name == "my_custom_experiment"
    assert my_exp_config.env_config["noise_level"] == 0.05
    assert my_exp_config.trainer_config["learning_rate"] == 1e-3
    assert my_exp_config.intrinsic_rewards_enabled == True
    assert my_exp_config.distributed_training_enabled == True

    # Test from_dict
    config_dict = {
        "experiment_name": "loaded_config",
        "total_timesteps": 1000,
        "device": "cpu",
        "policy_type": "AIHypothesisNet",
        "trainer_config": {"learning_rate": 0.0001, "ppo_epochs": 5}
    }
    loaded_config = ExperimentConfig.from_dict(config_dict)
    print("\nLoaded Config from Dict:")
    print(json.dumps(loaded_config.to_dict(), indent=2))
    assert loaded_config.experiment_name == "loaded_config"
    assert loaded_config.total_timesteps == 1000
    assert loaded_config.device == "cpu"
    assert loaded_config.policy_type == "AIHypothesisNet"
    assert loaded_config.trainer_config["learning_rate"] == 0.0001 # Trainer config should merge/overwrite defaults

    print("\nAll ExperimentConfig tests completed.")

