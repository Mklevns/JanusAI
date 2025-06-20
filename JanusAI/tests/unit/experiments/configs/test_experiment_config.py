"""
Tests for experiments/configs/experiment_config.py
"""
import pytest
import torch # Required for ExperimentConfig's device logic
from unittest.mock import patch
import json # For comparing dicts if necessary, or just direct comparison

from JanusAI.experiments.configs.experiment_config import ExperimentConfig

class TestExperimentConfig:

    def test_default_initialization(self):
        config = ExperimentConfig()
        assert config.experiment_name == "default_janus_experiment"
        assert config.seed == 42
        assert config.device == ("cuda" if torch.cuda.is_available() else "cpu")
        assert config.total_timesteps == 1_000_000
        assert config.max_episode_steps == 100
        assert config.log_interval == 100
        assert config.eval_interval == 10000 # Default logic: eval_interval = 10000
        assert config.checkpoint_interval == 100000 # Default logic: checkpoint_interval = 100000

        # Check default dicts (ensure they are new instances, not shared)
        assert isinstance(config.env_config, dict) and not config.env_config
        assert isinstance(config.policy_config, dict) and not config.policy_config
        assert isinstance(config.trainer_config, dict)
        assert config.trainer_config['learning_rate'] == 3e-4 # Check a default trainer value

        assert not config.intrinsic_rewards_enabled
        assert config.intrinsic_reward_config['novelty_weight'] == 0.3

        assert not config.adaptive_training_enabled
        assert config.adaptive_training_config['stagnation_threshold'] == 0.01

        assert not config.distributed_training_enabled
        assert config.distributed_config['num_ray_workers'] == 1

    def test_custom_initialization(self):
        env_cfg = {"task": "custom_task"}
        policy_cfg = {"layers": [64, 64]}
        trainer_cfg = {"learning_rate": 1e-5, "batch_size": 128} # Overrides some defaults, keeps others
        intrinsic_cfg = {"novelty_weight": 0.9}
        adaptive_cfg = {"history_length": 20}
        dist_cfg = {"num_ray_workers": 8}

        config = ExperimentConfig(
            experiment_name="custom_exp",
            seed=123,
            device="cpu",
            total_timesteps=500,
            max_episode_steps=50,
            log_interval=10,
            eval_interval=100,
            checkpoint_interval=200,
            env_type="CustomEnv",
            env_config=env_cfg,
            policy_type="CustomPolicy",
            policy_config=policy_cfg,
            trainer_type="CustomTrainer",
            trainer_config=trainer_cfg,
            intrinsic_rewards_enabled=True,
            intrinsic_reward_config=intrinsic_cfg,
            adaptive_training_enabled=True,
            adaptive_training_config=adaptive_cfg,
            distributed_training_enabled=True,
            distributed_config=dist_cfg
        )

        assert config.experiment_name == "custom_exp"
        assert config.seed == 123
        assert config.device == "cpu" # Custom override
        assert config.total_timesteps == 500
        assert config.max_episode_steps == 50
        assert config.log_interval == 10
        assert config.eval_interval == 100
        assert config.checkpoint_interval == 200

        assert config.env_type == "CustomEnv"
        assert config.env_config == env_cfg
        assert config.policy_type == "CustomPolicy"
        assert config.policy_config == policy_cfg
        assert config.trainer_type == "CustomTrainer"
        # Trainer config should merge: custom values override, others remain default
        expected_trainer_cfg = ExperimentConfig().trainer_config.copy() # Start with defaults
        expected_trainer_cfg.update(trainer_cfg) # Apply custom values
        assert config.trainer_config == expected_trainer_cfg

        assert config.intrinsic_rewards_enabled
        expected_intrinsic_cfg = ExperimentConfig().intrinsic_reward_config.copy()
        expected_intrinsic_cfg.update(intrinsic_cfg)
        assert config.intrinsic_reward_config == expected_intrinsic_cfg

        assert config.adaptive_training_enabled
        expected_adaptive_cfg = ExperimentConfig().adaptive_training_config.copy()
        expected_adaptive_cfg.update(adaptive_cfg)
        assert config.adaptive_training_config == expected_adaptive_cfg

        assert config.distributed_training_enabled
        expected_dist_cfg = ExperimentConfig().distributed_config.copy()
        expected_dist_cfg.update(dist_cfg)
        assert config.distributed_config == expected_dist_cfg


    @patch('torch.cuda.is_available', return_value=False)
    def test_post_init_device_fallback(self, mock_cuda_not_available, capsys):
        config = ExperimentConfig(device="cuda") # Try to set cuda
        assert config.device == "cpu" # Should fall back
        captured = capsys.readouterr()
        assert "Warning: CUDA not available, switching device to 'cpu'." in captured.out

    @patch('torch.cuda.is_available', return_value=True)
    def test_post_init_device_cuda_success(self, mock_cuda_available):
        config = ExperimentConfig(device="cuda")
        assert config.device == "cuda" # Should stay cuda

    def test_post_init_interval_adjustments(self):
        # log_interval <= 0
        config = ExperimentConfig(log_interval=0)
        assert config.log_interval == 1
        assert config.eval_interval == 1 # eval_interval becomes log_interval
        assert config.checkpoint_interval == 1 # checkpoint_interval becomes eval_interval

        # eval_interval <= 0
        config = ExperimentConfig(log_interval=50, eval_interval=-10)
        assert config.eval_interval == 50 # Becomes log_interval
        assert config.checkpoint_interval == 50 # Becomes eval_interval

        # checkpoint_interval <= 0
        config = ExperimentConfig(eval_interval=200, checkpoint_interval=0)
        assert config.checkpoint_interval == 200 # Becomes eval_interval

        # eval_interval < log_interval
        config = ExperimentConfig(log_interval=100, eval_interval=50)
        assert config.eval_interval == 100 # Becomes log_interval

        # checkpoint_interval < eval_interval
        config = ExperimentConfig(eval_interval=200, checkpoint_interval=100)
        assert config.checkpoint_interval == 200 # Becomes eval_interval

        # All positive and correctly ordered
        config = ExperimentConfig(log_interval=10, eval_interval=20, checkpoint_interval=30)
        assert config.log_interval == 10
        assert config.eval_interval == 20
        assert config.checkpoint_interval == 30


    def test_to_dict(self):
        config = ExperimentConfig(experiment_name="to_dict_test", seed=777)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['experiment_name'] == "to_dict_test"
        assert config_dict['seed'] == 777
        assert config_dict['trainer_config']['gamma'] == 0.99 # Check a nested default

    def test_from_dict_basic(self):
        data = {
            "experiment_name": "from_dict_exp",
            "total_timesteps": 2000,
            "device": "mps", # Example non-default device string (post_init won't change this if not 'cuda')
            "policy_type": "AnotherPolicy",
            "trainer_config": {"learning_rate": 5e-5, "clip_range": 0.1} # Partial override
        }
        config = ExperimentConfig.from_dict(data)

        assert config.experiment_name == "from_dict_exp"
        assert config.total_timesteps == 2000
        assert config.device == "mps"
        assert config.policy_type == "AnotherPolicy"

        # Check trainer_config merge: 5e-5 and 0.1 should be set, others default
        default_trainer_cfg = ExperimentConfig().trainer_config
        assert config.trainer_config['learning_rate'] == 5e-5
        assert config.trainer_config['clip_range'] == 0.1
        assert config.trainer_config['ppo_epochs'] == default_trainer_cfg['ppo_epochs'] # Check a non-overridden default

        # Check a field not in `data` uses its default
        assert config.seed == 42
        assert config.intrinsic_rewards_enabled is False


    def test_from_dict_ignores_extra_keys(self):
        data = {
            "experiment_name": "extra_keys_test",
            "extra_field_not_in_dataclass": "should_be_ignored"
        }
        config = ExperimentConfig.from_dict(data)
        assert config.experiment_name == "extra_keys_test"
        assert not hasattr(config, "extra_field_not_in_dataclass")

    def test_from_dict_with_all_defaults(self):
        # Test creating from an empty dict, should get all defaults
        config = ExperimentConfig.from_dict({})
        default_config = ExperimentConfig() # Create a default instance for comparison

        # Compare all fields (or a representative subset)
        assert config.experiment_name == default_config.experiment_name
        assert config.seed == default_config.seed
        assert config.trainer_config == default_config.trainer_config
        assert config.intrinsic_reward_config == default_config.intrinsic_reward_config
