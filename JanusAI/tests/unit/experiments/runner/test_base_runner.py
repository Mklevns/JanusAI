"""
Tests for experiments/runner/base_runner.py
"""
import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import torch

from janus_ai.experiments.runner.base_runner import BaseExperimentRunner

# --- Mocks for dependencies ---
MockSymbolicDiscoveryEnv = MagicMock(name="MockSymbolicDiscoveryEnv")
MockHypothesisNet = MagicMock(name="MockHypothesisNet") # Policy
MockPPOTrainer = MagicMock(name="MockPPOTrainer")
MockTrainingLogger = MagicMock(name="MockTrainingLogger")
MockCheckpointManager = MagicMock(name="MockCheckpointManager")
MockExperimentConfig = MagicMock(name="MockExperimentConfig") # If using a dataclass for config

# Patch imported modules for all tests in this class
@patch('JanusAI.experiments.runner.base_runner.SymbolicDiscoveryEnv', MockSymbolicDiscoveryEnv)
@patch('JanusAI.experiments.runner.base_runner.HypothesisNet', MockHypothesisNet)
@patch('JanusAI.experiments.runner.base_runner.PPOTrainer', MockPPOTrainer)
@patch('JanusAI.experiments.runner.base_runner.TrainingLogger', MockTrainingLogger)
@patch('JanusAI.experiments.runner.base_runner.CheckpointManager', MockCheckpointManager)
class TestBaseExperimentRunner:

    @pytest.fixture(autouse=True)
    def reset_global_mocks(self):
        MockSymbolicDiscoveryEnv.reset_mock()
        MockHypothesisNet.reset_mock()
        MockPPOTrainer.reset_mock()
        MockTrainingLogger.reset_mock()
        MockCheckpointManager.reset_mock()
        MockExperimentConfig.reset_mock()

    @pytest.fixture
    def mock_env_instance(self):
        env = MockSymbolicDiscoveryEnv.return_value
        env.action_space = MagicMock()
        env.action_space.n = 5
        env.observation_space = MagicMock()
        env.observation_space.shape = (10,)
        env.get_action_mask = MagicMock(return_value=np.ones(5, dtype=bool))
        env.reset = MagicMock(return_value=(np.random.rand(10), {"info": "reset_info"}))
        env.step = MagicMock(return_value=(np.random.rand(10), 0.1, False, False, {"info": "step_info"}))
        return env

    @pytest.fixture
    def mock_policy_instance(self):
        policy = MockHypothesisNet.return_value
        # Mock parameters() to return something that has .device
        mock_param = MagicMock(spec=torch.Tensor)
        mock_param.device = "cpu" # Mock device
        policy.parameters = MagicMock(return_value=iter([mock_param]))
        policy.get_action = MagicMock(return_value=(0, -0.1, 0.5)) # action, log_prob, value
        return policy

    @pytest.fixture
    def sample_experiment_config_dict(self):
        return {
            'experiment_name': 'test_exp',
            'total_timesteps': 10000,
            'rollout_length': 128,
            'log_interval': 5,
            'eval_interval': 50,
            'max_episode_steps': 20,
            # PPOTrainer relevant defaults if not overridden by runner
            'learning_rate': 2e-4,
            'ppo_epochs': 8,
            'gamma': 0.98,
            'gae_lambda': 0.92,
            'clip_range': 0.1,
            'value_coef': 0.6,
            'entropy_coef': 0.02,
            'max_grad_norm': 0.6,
        }

    def test_init_no_resume(self, mock_env_instance, mock_policy_instance, sample_experiment_config_dict):
        # Mock trainer.load_from_checkpoint to return 0 (no resume)
        MockPPOTrainer.return_value.load_from_checkpoint = MagicMock(return_value=0)

        runner = BaseExperimentRunner(
            env=mock_env_instance,
            policy=mock_policy_instance,
            experiment_config=sample_experiment_config_dict,
            checkpoint_dir="./test_ckpts",
            log_dir="./test_logs"
        )

        assert runner.env == mock_env_instance
        assert runner.policy == mock_policy_instance
        assert runner.config == sample_experiment_config_dict

        MockPPOTrainer.assert_called_once_with(
            policy=mock_policy_instance,
            env=mock_env_instance,
            learning_rate=sample_experiment_config_dict['learning_rate'],
            n_epochs=sample_experiment_config_dict['ppo_epochs'],
            gamma=sample_experiment_config_dict['gamma'],
            gae_lambda=sample_experiment_config_dict['gae_lambda'],
            clip_range=sample_experiment_config_dict['clip_range'],
            value_coef=sample_experiment_config_dict['value_coef'],
            entropy_coef=sample_experiment_config_dict['entropy_coef'],
            max_grad_norm=sample_experiment_config_dict['max_grad_norm'],
            checkpoint_dir="./test_ckpts"
        )
        assert runner.trainer == MockPPOTrainer.return_value

        MockTrainingLogger.assert_called_once_with(log_dir="./test_logs", experiment_name='test_exp')
        assert runner.logger == MockTrainingLogger.return_value

        MockCheckpointManager.assert_called_once_with("./test_ckpts")
        assert runner.checkpoint_manager == MockCheckpointManager.return_value

        assert runner.total_timesteps == 10000
        assert runner.rollout_length == 128
        assert runner.log_interval == 5
        assert runner.eval_interval == 50
        assert runner.current_timesteps == 0
        MockPPOTrainer.return_value.load_from_checkpoint.assert_called_once()


    def test_init_with_resume(self, mock_env_instance, mock_policy_instance, sample_experiment_config_dict):
        MockPPOTrainer.return_value.load_from_checkpoint = MagicMock(return_value=5000) # Resuming from 5000

        runner = BaseExperimentRunner(
            env=mock_env_instance,
            policy=mock_policy_instance,
            experiment_config=sample_experiment_config_dict
        )
        assert runner.current_timesteps == 5000


    def test_run_experiment(self, mock_env_instance, mock_policy_instance, sample_experiment_config_dict):
        # Mock trainer methods
        mock_trainer_instance = MockPPOTrainer.return_value
        mock_trainer_instance.load_from_checkpoint = MagicMock(return_value=0) # Start fresh
        mock_trainer_instance.train = MagicMock()
        mock_trainer_instance.get_recent_average = MagicMock(side_effect=lambda metric, window_size: 0.5 if metric == "mean_reward_episode" else 0.1)

        runner = BaseExperimentRunner(mock_env_instance, mock_policy_instance, sample_experiment_config_dict)
        # Patch evaluate_final_policy for this test
        with patch.object(runner, 'evaluate_final_policy', return_value={"avg_eval_reward": 0.8}) as mock_eval_final:
            runner.run()

            mock_trainer_instance.train.assert_called_once_with(
                total_timesteps=sample_experiment_config_dict['total_timesteps'], # Remaining
                rollout_length=sample_experiment_config_dict['rollout_length'],
                n_epochs=sample_experiment_config_dict['ppo_epochs'],
                batch_size=sample_experiment_config_dict.get('batch_size', 64), # Default if not in dict
                log_interval=sample_experiment_config_dict['log_interval']
            )
            assert runner.current_timesteps == sample_experiment_config_dict['total_timesteps']
            mock_eval_final.assert_called_once()
            runner.logger.save_summary.assert_called_once_with({
                "final_performance": 0.5, # From get_recent_average mock
                "final_loss": 0.1
            })


    def test_evaluate_final_policy(self, mock_env_instance, mock_policy_instance, sample_experiment_config_dict):
        runner = BaseExperimentRunner(mock_env_instance, mock_policy_instance, sample_experiment_config_dict)
        runner.current_timesteps = runner.total_timesteps # Ensure it doesn't affect eval logic

        n_eval_episodes = 3
        results = runner.evaluate_final_policy(n_episodes=n_eval_episodes)

        assert mock_policy_instance.eval.call_count == 1 # Called at start of eval
        assert mock_env_instance.reset.call_count == n_eval_episodes
        # Each episode runs until done or max_episode_steps.
        # Mock step returns done=False until max_episode_steps.
        # max_episode_steps is 20 from config.
        assert mock_env_instance.step.call_count == n_eval_episodes * sample_experiment_config_dict['max_episode_steps']
        assert mock_policy_instance.get_action.call_count == n_eval_episodes * sample_experiment_config_dict['max_episode_steps']
        # Check deterministic=True was used
        for call_args in mock_policy_instance.get_action.call_args_list:
            assert call_args[1]['deterministic'] is True

        assert mock_policy_instance.train.call_count == 1 # Called at end of eval
        assert 'average_evaluation_reward' in results
        assert isinstance(results['average_evaluation_reward'], float)


    def test_save_runner_state(self, mock_env_instance, mock_policy_instance, sample_experiment_config_dict):
        runner = BaseExperimentRunner(mock_env_instance, mock_policy_instance, sample_experiment_config_dict)
        runner.current_timesteps = 1234
        runner.logger.get_live_metrics_report = MagicMock(return_value={"log_data": "here"})

        runner._save_runner_state(path="dummy_path_not_used_by_mock") # Path arg not used by mock

        runner.checkpoint_manager.save_checkpoint.assert_called_once_with(
            {'current_timesteps': 1234, 'logger_state': {"log_data": "here"}},
            1234,
            is_runner_state=True
        )

    def test_load_runner_state_success(self, mock_env_instance, mock_policy_instance, sample_experiment_config_dict):
        runner = BaseExperimentRunner(mock_env_instance, mock_policy_instance, sample_experiment_config_dict)

        loaded_state_data = {'current_timesteps': 5678, 'logger_state': {"some": "data"}}
        runner.checkpoint_manager.load_checkpoint = MagicMock(return_value=loaded_state_data)

        success = runner._load_runner_state(path="dummy_path_for_load")

        assert success is True
        assert runner.current_timesteps == 5678
        runner.checkpoint_manager.load_checkpoint.assert_called_once_with("dummy_path_for_load", is_runner_state=True)
        # Logger state reconstruction is not explicitly tested here, depends on TrainingLogger implementation.

    def test_load_runner_state_failure(self, mock_env_instance, mock_policy_instance, sample_experiment_config_dict):
        runner = BaseExperimentRunner(mock_env_instance, mock_policy_instance, sample_experiment_config_dict)
        initial_timesteps = runner.current_timesteps # Assuming it's 0 from no-resume init

        runner.checkpoint_manager.load_checkpoint = MagicMock(return_value=None) # Simulate no checkpoint found

        success = runner._load_runner_state()

        assert success is False
        assert runner.current_timesteps == initial_timesteps # Should not change
        runner.checkpoint_manager.load_checkpoint.assert_called_once_with(None, is_runner_state=True)

    # Example test for the __main__ block's logic if it were refactored into a runnable function
    # This is illustrative, as the actual __main__ is for direct execution.
    @patch('JanusAI.experiments.runner.base_runner.BaseExperimentRunner') # Patch the class itself
    @patch('JanusAI.experiments.runner.base_runner.Variable') # Patch Variable if used in example setup
    @patch('os.path.exists') # Mock os.path.exists
    @patch('shutil.rmtree') # Mock shutil.rmtree
    def test_main_block_simulation(self, mock_rmtree, mock_exists, MockVariableForMain, MockRunnerClass, tmp_path):
        # This test would simulate the setup done in the original __main__
        # For instance, if __main__ was a function `main_runner_example(config_dict, ...)`

        # Example: MockVariableForMain.side_effect = lambda name, index: MagicMock(name=name, index=index)

        # Mock os.path.exists to control cleanup logic
        mock_exists.return_value = True # Assume dirs exist for cleanup part

        # The actual __main__ block in base_runner.py is quite involved and sets up many mocks.
        # A direct test of it "as is" is more like an integration test.
        # For unit tests, we'd test the components it uses, which we've done for BaseExperimentRunner.

        # If we were to test a hypothetical function extracted from __main__:
        # main_runner_example(config_dict, checkpoint_dir=str(tmp_path/"ckpts"), log_dir=str(tmp_path/"logs"))
        # MockRunnerClass.assert_called_once()
        # MockRunnerClass.return_value.run.assert_called_once()
        # mock_rmtree.assert_any_call(str(tmp_path/"ckpts"))
        # mock_rmtree.assert_any_call(str(tmp_path/"logs"))
        pass # Placeholder, as direct test of __main__ is complex for unit level.
