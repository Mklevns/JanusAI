"""
Base Experiment Runner
======================

Defines the base class for running symbolic discovery experiments.
Handles the high-level orchestration of environment interaction,
policy training, and basic logging for a single experiment.
"""

import numpy as np
import torch
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Callable

# Import core components
from JanusAI.core.grammar.base_grammar import ProgressiveGrammar
from JanusAI.core.expressions.expression import Variable, Expression

# Import environment and policy (assuming specific types like SymbolicDiscoveryEnv and HypothesisNet)
from JanusAI.environments.base.symbolic_env import SymbolicDiscoveryEnv
from JanusAI.ml.networks.hypothesis_net import HypothesisNet
from JanusAI.ml.training.ppo_trainer import PPOTrainer # Assuming PPOTrainer is the default trainer

# Import logging utilities
from JanusAI.utils.logging.experiment_logger import TrainingLogger
from JanusAI.utils.io.checkpoint_manager import CheckpointManager # For saving/loading models

# Forward declaration for type hinting if needed (though Python 3.7+ usually handles this with strings)
# from janus.config.models import ExperimentConfig # If ExperimentConfig is available


class BaseExperimentRunner:
    """
    Orchestrates and executes a single symbolic discovery experiment.

    Handles environment setup, policy training, interaction loops,
    and logging of results. Designed to be extendable for more complex
    experiment setups (e.g., meta-learning, distributed training).
    """

    def __init__(self,
                 env: SymbolicDiscoveryEnv,
                 policy: HypothesisNet,
                 experiment_config: Any, # Can be a dict or a specific ExperimentConfig object
                 checkpoint_dir: str = "./checkpoints",
                 log_dir: str = "./logs"
                ):
        """
        Initializes the BaseExperimentRunner.

        Args:
            env: The environment instance for the experiment.
            policy: The policy network to be trained/evaluated.
            experiment_config: Configuration object or dictionary for the experiment.
            checkpoint_dir: Directory to save model checkpoints.
            log_dir: Directory for experiment logs.
        """
        self.env = env
        self.policy = policy
        self.config = experiment_config # Store the full config

        # Initialize the PPO trainer (can be configured via experiment_config)
        self.trainer = PPOTrainer(
            policy=self.policy,
            env=self.env,
            learning_rate=self.config.get('learning_rate', 3e-4),
            n_epochs=self.config.get('ppo_epochs', 10),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
            clip_range=self.config.get('clip_range', 0.2),
            value_coef=self.config.get('value_coef', 0.5),
            entropy_coef=self.config.get('entropy_coef', 0.01),
            max_grad_norm=self.config.get('max_grad_norm', 0.5),
            checkpoint_dir=checkpoint_dir # Pass checkpoint directory to trainer
        )

        self.logger = TrainingLogger(log_dir=log_dir, experiment_name=self.config.get('experiment_name', 'default_experiment'))
        self.checkpoint_manager = CheckpointManager(checkpoint_dir) # Use a separate manager for runner-level state
        
        # Training state
        self.total_timesteps = self.config.get('total_timesteps', 1_000_000)
        self.rollout_length = self.config.get('rollout_length', 2048)
        self.log_interval = self.config.get('log_interval', 10)
        self.eval_interval = self.config.get('eval_interval', 100) # How often to run evaluation

        # Load latest checkpoint if available to resume training
        self.current_timesteps = self.trainer.load_from_checkpoint()
        if self.current_timesteps > 0:
            print(f"Resuming experiment from timestep {self.current_timesteps}.")

        print("BaseExperimentRunner initialized.")
        print(f"Experiment: {self.config.get('experiment_name', 'Unnamed')}")
        print(f"Total timesteps: {self.total_timesteps}")
        print(f"Device: {self.policy.parameters().__next__().device}") # Assuming policy has parameters


    def run(self):
        """
        Starts and manages the main experiment loop.
        """
        print(f"\nStarting experiment '{self.config.get('experiment_name', 'Unnamed')}'...")
        print("-" * 50)

        start_time = time.time()
        
        while self.current_timesteps < self.total_timesteps:
            # Collect rollouts and train the policy
            # The PPO trainer handles its own logging of training metrics
            
            # This is where curriculum or task-specific parameters would be passed
            # to trainer.train() if it were a meta-training setup.
            # For BaseExperimentRunner, we assume the trainer just trains.
            
            # Ensure proper casting for `total_timesteps` and `rollout_length` if they are floats
            # (though config parsing should handle this)
            self.trainer.train(
                total_timesteps=self.total_timesteps - self.current_timesteps, # Remaining timesteps
                rollout_length=self.rollout_length,
                n_epochs=self.config.get('ppo_epochs', 10),
                batch_size=self.config.get('batch_size', 64),
                log_interval=self.log_interval # PPO trainer will log internally
            )
            
            # Update current timesteps from trainer's state after training completes a cycle
            # This assumes trainer.train() updates its internal timesteps and we can retrieve it.
            # A more robust way might be to get the timesteps from the trainer's checkpoint loading.
            # For now, let's assume `trainer.train` runs for `total_timesteps - current_timesteps` and exits.
            # If `trainer.train` runs in smaller chunks, this loop needs to reflect that.
            
            # Simplified: Assuming one call to trainer.train completes the run up to total_timesteps.
            self.current_timesteps = self.total_timesteps # For now, assumes trainer runs to completion

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("\nExperiment finished.")
        print(f"Total elapsed time: {elapsed_time:.2f} seconds.")

        # Final evaluation (optional)
        self.evaluate_final_policy()

        # Save final experiment summary
        self.logger.save_summary({
            "final_performance": self.trainer.get_recent_average("mean_reward_episode", window_size=50), # Get last 50 episodes
            "final_loss": self.trainer.get_recent_average("loss", window_size=50)
        })

    def evaluate_final_policy(self, n_episodes: int = 20) -> Dict[str, float]:
        """
        Evaluates the trained policy on a number of episodes.
        """
        print(f"\nEvaluating final policy over {n_episodes} episodes...")
        eval_rewards = []
        
        # Set policy to evaluation mode
        self.policy.eval()

        for i in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False
            truncated = False
            step_count = 0
            
            while not done and not truncated and step_count < self.config.get('max_episode_steps', 100):
                # Ensure obs is a tensor for policy.get_action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(next(self.policy.parameters()).device)
                
                # Get action mask from environment
                action_mask_np = self.env.get_action_mask()
                action_mask_tensor = torch.BoolTensor(action_mask_np).unsqueeze(0).to(obs_tensor.device)

                # Use deterministic action for evaluation
                action, _, _ = self.policy.get_action(obs_tensor, action_mask=action_mask_tensor, deterministic=True)
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                obs = next_obs
                done = terminated or truncated
                step_count += 1
            
            eval_rewards.append(episode_reward)
            print(f"  Eval Episode {i+1}: Reward={episode_reward:.4f}")
        
        avg_eval_reward = np.mean(eval_rewards)
        print(f"Average evaluation reward over {n_episodes} episodes: {avg_eval_reward:.4f}")
        
        # Set policy back to training mode
        self.policy.train()

        return {"average_evaluation_reward": avg_eval_reward}

    def _save_runner_state(self, path: str):
        """Saves the internal state of the runner (not just the policy)."""
        state = {
            'current_timesteps': self.current_timesteps,
            'logger_state': self.logger.get_live_metrics_report(), # Or full history
            # Add any other runner-specific state that needs to be checkpointed
        }
        # This is for runner's own state, not the model's
        self.checkpoint_manager.save_checkpoint(state, self.current_timesteps, is_runner_state=True)
        print(f"Runner state saved at {self.current_timesteps} timesteps.")

    def _load_runner_state(self, path: Optional[str] = None):
        """Loads the internal state of the runner."""
        loaded_state = self.checkpoint_manager.load_checkpoint(path, is_runner_state=True)
        if loaded_state:
            self.current_timesteps = loaded_state.get('current_timesteps', 0)
            # Reconstruct logger state if needed
            print(f"Runner state loaded. Resuming from {self.current_timesteps} timesteps.")
            return True
        return False


if __name__ == "__main__":
    # This __main__ block serves as a test for the BaseExperimentRunner.
    # It requires mock versions of Env, Policy, Trainer if full Janus modules are not available.

    print("--- Testing BaseExperimentRunner ---")

    # Mock components for testing
    class MockGrammar:
        def get_valid_actions(self, current_expression_node): return [0, 1, 2]
    
    class MockSymbolicDiscoveryEnv:
        def __init__(self, grammar, target_data, variables, max_depth, max_complexity, reward_config, action_space_size, provide_tree_structure):
            self.grammar = grammar
            self.target_data = target_data
            self.variables = variables
            self.max_depth = max_depth
            self.max_complexity = max_complexity
            self.reward_config = reward_config
            self.action_space = type('ActionSpace', (object,), {'n': action_space_size or 5})()
            self.observation_space = type('ObsSpace', (object,), {'shape': (64,)})() # Dummy obs dim
            self.provide_tree_structure = provide_tree_structure
            self.current_state = type('TreeState', (object,), {
                'root': type('Node', (object,), {'node_type': type('NT', (object,), {'value': 'operator'})(), 'children': []})(),
                'count_nodes': lambda : 1,
                'is_complete': lambda : False
            })()
            self._episode_step_count = 0
            self._max_episode_steps = 5 # Short episodes for testing
            
        def reset(self, seed=None, options=None):
            self._episode_step_count = 0
            initial_obs = np.random.rand(self.observation_space.shape[0])
            info = {'expression': 'x', 'complexity': 1}
            # Mock trajectory_data and variables for rewards (if env step provides it)
            info['trajectory_data'] = {'x': np.array([0.0]), 'y': np.array([0.0])}
            info['variables'] = [Variable("x", 0), Variable("y", 1)]
            return initial_obs, info

        def step(self, action):
            self._episode_step_count += 1
            next_obs = np.random.rand(self.observation_space.shape[0])
            reward = np.random.rand() * 0.1 # Small random reward
            terminated = self._episode_step_count >= self._max_episode_steps
            truncated = False
            info = {'expression': 'x + 1', 'complexity': 3, 'episode_reward': 0.5} # Mock info for intrinsic rewards
            # Ensure trajectory_data and variables are in info for rewards
            info['trajectory_data'] = {'x': np.array([0.0]), 'y': np.array([0.0])}
            info['variables'] = [Variable("x", 0), Variable("y", 1)]
            return next_obs, reward, terminated, truncated, info

        def get_action_mask(self): return np.ones(self.action_space.n, dtype=bool)

    class MockPolicy(torch.nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.linear = torch.nn.Linear(obs_dim, action_dim)
            self.value_head = torch.nn.Linear(obs_dim, 1)
        def forward(self, obs, action_mask=None, task_trajectories=None, tree_structure=None):
            logits = self.linear(obs)
            if action_mask is not None:
                logits[~action_mask] = -1e9
            return {'action_logits': logits, 'value': self.value_head(obs)}
        def get_action(self, obs, action_mask=None, task_trajectories=None, tree_structure=None, deterministic=False):
            outputs = self.forward(obs, action_mask, task_trajectories, tree_structure)
            if deterministic:
                action = torch.argmax(outputs['action_logits'], dim=-1)
            else:
                action = torch.distributions.Categorical(logits=outputs['action_logits']).sample()
            log_prob = torch.distributions.Categorical(logits=outputs['action_logits']).log_prob(action)
            return action.item(), log_prob.item(), outputs['value'].item()
        def eval(self): pass # Mock eval
        def train(self): pass # Mock train

    # Mock PPOTrainer (only train and load_from_checkpoint methods needed for runner)
    class MockPPOTrainer:
        def __init__(self, policy, env, **kwargs):
            self.policy = policy
            self.env = env
            self.episode_rewards = deque(maxlen=100) # Mock tracking
            self.loss_history = deque(maxlen=100)
            self.timesteps_trained = 0
        def train(self, total_timesteps, rollout_length, n_epochs, batch_size, log_interval):
            # Simulate training progress
            num_updates = total_timesteps // rollout_length
            for i in range(num_updates):
                mock_loss = np.random.rand() * 0.1
                self.loss_history.append(mock_loss)
                self.episode_rewards.append(np.random.rand() * 1.0) # Simulate some rewards
                self.timesteps_trained += rollout_length
                if (i + 1) % log_interval == 0:
                    print(f"  Mock Trainer: Update {i+1}, Loss: {mock_loss:.4f}, Avg Reward: {np.mean(self.episode_rewards):.4f}")
            # Ensure timesteps are updated as if it ran for total_timesteps
            self.timesteps_trained = total_timesteps 
        def load_from_checkpoint(self):
            # Simulate loading, return some initial timesteps if needed for resume test
            return 0 # Start from 0 for this test
        def get_recent_average(self, metric_name, window_size):
            if metric_name == "mean_reward_episode": return np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            if metric_name == "loss": return np.mean(self.loss_history) if self.loss_history else 0.0
            return 0.0

    # Override imported modules for testing context
    import sys
    sys.modules['janus.environments.base.symbolic_env'].SymbolicDiscoveryEnv = MockSymbolicDiscoveryEnv
    sys.modules['janus.ml.networks.hypothesis_net'].HypothesisNet = MockPolicy
    sys.modules['janus.ml.training.ppo_trainer'].PPOTrainer = MockPPOTrainer
    sys.modules['janus.core.expressions.expression'].Variable = Variable # Ensure real Variable is used for Variable in env


    # Setup
    grammar_test = MockGrammar()
    variables_test = [Variable("x", 0), Variable("y", 1)]
    target_data_test = np.random.rand(100, 3)

    env_test = MockSymbolicDiscoveryEnv(
        grammar=grammar_test,
        target_data=target_data_test,
        variables=variables_test,
        max_depth=5, max_complexity=10, reward_config={},
        action_space_size=5, provide_tree_structure=True
    )
    policy_test = MockPolicy(obs_dim=env_test.observation_space.shape[0], action_dim=env_test.action_space.n)

    # Experiment config
    experiment_config_dict = {
        'experiment_name': 'test_base_runner',
        'total_timesteps': 200, # Very short run for testing
        'rollout_length': 20,
        'ppo_epochs': 2,
        'batch_size': 10,
        'learning_rate': 1e-4,
        'max_episode_steps': 5 # Short episodes in mock env
    }

    # Clean up checkpoint/log directories from previous runs if any
    test_checkpoint_dir = "./runner_test_checkpoints"
    test_log_dir = "./runner_test_logs"
    if os.path.exists(test_checkpoint_dir):
        import shutil; shutil.rmtree(test_checkpoint_dir)
    if os.path.exists(test_log_dir):
        import shutil; shutil.rmtree(test_log_dir)

    # Initialize and run the runner
    runner = BaseExperimentRunner(
        env=env_test,
        policy=policy_test,
        experiment_config=experiment_config_dict,
        checkpoint_dir=test_checkpoint_dir,
        log_dir=test_log_dir
    )
    runner.run()

    # Verify logs/checkpoints are created
    assert os.path.exists(runner.logger.log_dir)
    assert os.path.exists(runner.logger.log_file_path)
    assert os.path.exists(runner.logger.summary_file_path)
    assert os.path.exists(os.path.join(test_checkpoint_dir, "checkpoint_*.pt"))

    # Clean up test directories
    import shutil
    shutil.rmtree(test_checkpoint_dir)
    shutil.rmtree(test_log_dir)
    print("\nCleaned up test directories.")

    # Restore original modules
    del sys.modules['janus.environments.base.symbolic_env'].SymbolicDiscoveryEnv
    del sys.modules['janus.ml.networks.hypothesis_net'].HypothesisNet
    del sys.modules['janus.ml.training.ppo_trainer'].PPOTrainer


    print("\nBaseExperimentRunner tests completed successfully.")

