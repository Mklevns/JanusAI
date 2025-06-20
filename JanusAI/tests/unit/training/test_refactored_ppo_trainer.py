# JanusAI/tests/unit/training/test_refactored_ppo_trainer.py
"""
Test Suite for Refactored PPO Trainer
=====================================

Demonstrates how the decoupled design enables comprehensive testing.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch
from typing import Dict

from JanusAI.ml.training.ppo_trainer import PPOTrainer, RolloutBuffer
from JanusAI.ml.networks.hypothesis_net import HypothesisNet
from JanusAI.environments.base.symbolic_env import SymbolicDiscoveryEnv


class TestRolloutBuffer(unittest.TestCase):
    """Test the RolloutBuffer component independently."""
    
    def setUp(self):
        self.buffer = RolloutBuffer()
    
    def test_add_and_reset(self):
        """Test adding data and resetting buffer."""
        # Add some data
        self.buffer.add(
            obs=np.array([1, 2, 3]),
            action=0,
            reward=1.0,
            value=0.5,
            log_prob=-0.69,
            done=False,
            action_mask=np.array([1, 1, 0]),
            tree_structure={'depth': 2}
        )
        
        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(len(self.buffer.observations), 1)
        
        # Reset
        self.buffer.reset()
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(len(self.buffer.observations), 0)
    
    def test_compute_advantages(self):
        """Test advantage computation."""
        # Add trajectory
        for i in range(5):
            self.buffer.add(
                obs=np.array([i]),
                action=0,
                reward=1.0,
                value=float(5 - i),  # Decreasing values
                log_prob=-0.69,
                done=(i == 4),
                action_mask=np.array([1, 1]),
                tree_structure=None
            )
        
        # Mock policy for final value
        mock_policy = Mock()
        mock_policy.return_value = {'value': torch.tensor([[0.0]])}
        
        # Compute advantages
        self.buffer.compute_returns_and_advantages(
            policy=mock_policy,
            gamma=0.99,
            gae_lambda=0.95
        )
        
        # Check results
        self.assertEqual(len(self.buffer.advantages), 5)
        self.assertEqual(len(self.buffer.returns), 5)
        
        # Advantages should be normalized
        adv = np.array(self.buffer.advantages)
        self.assertAlmostEqual(adv.mean(), 0.0, places=5)
        self.assertAlmostEqual(adv.std(), 1.0, places=5)
    
    def test_get_tensors(self):
        """Test conversion to tensors."""
        # Add data
        for i in range(3):
            self.buffer.add(
                obs=np.array([i, i+1]),
                action=i,
                reward=float(i),
                value=float(i),
                log_prob=float(-i),
                done=False,
                action_mask=np.array([1, 0]),
                tree_structure=None
            )
        
        # Get tensors
        data = self.buffer.get()
        
        # Check types and shapes
        self.assertIsInstance(data['observations'], torch.Tensor)
        self.assertEqual(data['observations'].shape, (3, 2))
        self.assertEqual(data['actions'].shape, (3,))
        self.assertIsInstance(data['tree_structures'], list)


class TestPPOTrainerComponents(unittest.TestCase):
    """Test individual components of PPO trainer."""
    
    def setUp(self):
        """Create mock environment and policy."""
        # Mock environment
        self.mock_env = Mock(spec=SymbolicDiscoveryEnv)
        self.mock_env.observation_space.shape = (10,)
        self.mock_env.action_space.n = 5
        self.mock_env.reset.return_value = (np.zeros(10), {})
        self.mock_env.step.return_value = (
            np.zeros(10),  # next_obs
            1.0,           # reward
            False,         # terminated
            False,         # truncated
            {}            # info
        )
        
        # Mock policy
        self.mock_policy = Mock(spec=HypothesisNet)
        self.mock_policy.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        self.mock_policy.return_value = {
            'action_logits': torch.randn(1, 5),
            'value': torch.randn(1, 1)
        }
        self.mock_policy.to.return_value = self.mock_policy
        
        # Create trainer
        self.trainer = PPOTrainer(
            policy=self.mock_policy,
            env=self.mock_env,
            n_epochs=2,
            batch_size=32
        )
    
    def test_collect_rollouts_isolation(self):
        """Test collect_rollouts in isolation."""
        # Collect data
        rollout_data = self.trainer.collect_rollouts(n_steps=10)
        
        # Verify environment interaction
        self.assertEqual(self.mock_env.step.call_count, 10)
        
        # Check returned data
        self.assertIn('observations', rollout_data)
        self.assertIn('actions', rollout_data)
        self.assertIn('rewards', rollout_data)
        self.assertIn('advantages', rollout_data)
        
        # Check shapes
        self.assertEqual(len(rollout_data['observations']), 10)
        self.assertEqual(len(rollout_data['rewards']), 10)
    
    def test_learn_with_synthetic_data(self):
        """Test learn method with synthetic data."""
        # Create synthetic rollout data
        n_samples = 100
        synthetic_data = {
            'observations': torch.randn(n_samples, 10),
            'actions': torch.randint(0, 5, (n_samples,)),
            'rewards': torch.randn(n_samples),
            'values': torch.randn(n_samples),
            'log_probs': torch.randn(n_samples),
            'advantages': torch.randn(n_samples),
            'returns': torch.randn(n_samples),
            'action_masks': torch.ones(n_samples, 5, dtype=torch.bool),
            'tree_structures': [None] * n_samples
        }
        
        # Learn from synthetic data
        metrics = self.trainer.learn(synthetic_data)
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('entropy', metrics)
        
        # Verify policy was called
        self.assertGreater(self.mock_policy.call_count, 0)
    
    def test_train_orchestration(self):
        """Test that train properly orchestrates collect and learn."""
        # Mock the component methods
        mock_rollout_data = {
            'observations': torch.randn(50, 10),
            'actions': torch.randint(0, 5, (50,)),
            'rewards': torch.randn(50),
            'values': torch.randn(50),
            'log_probs': torch.randn(50),
            'advantages': torch.randn(50),
            'returns': torch.randn(50),
            'action_masks': torch.ones(50, 5, dtype=torch.bool),
            'tree_structures': [None] * 50
        }
        
        self.trainer.collect_rollouts = Mock(return_value=mock_rollout_data)
        self.trainer.learn = Mock(return_value={
            'loss': 0.5,
            'policy_loss': 0.3,
            'value_loss': 0.2,
            'entropy': 0.1
        })
        
        # Run training
        history = self.trainer.train(
            total_timesteps=100,
            rollout_length=50,
            log_interval=1
        )
        
        # Verify orchestration
        self.assertEqual(self.trainer.collect_rollouts.call_count, 2)  # 100/50 = 2
        self.assertEqual(self.trainer.learn.call_count, 2)
        
        # Check history
        self.assertIn('timesteps', history)
        self.assertIn('loss', history)
        self.assertEqual(len(history['loss']), 2)


class TestPPOTrainerIntegration(unittest.TestCase):
    """Integration tests for the refactored PPO trainer."""
    
    def test_full_training_loop(self):
        """Test complete training loop with real components."""
        # Create simple environment
        from janus.core.grammar.base_grammar import ProgressiveGrammar
        from janus.core.expressions.expression import Variable
        
        grammar = ProgressiveGrammar()
        variables = [Variable('x', 0)]
        
        # Simple test data
        X = np.random.randn(50, 1)
        y = 2 * X[:, 0] + 1
        
        env = SymbolicDiscoveryEnv(
            grammar=grammar,
            X_data=X,
            y_data=y,
            variables=variables,
            max_depth=3
        )
        
        # Create policy
        policy = HypothesisNet(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.n,
            hidden_dim=32,
            encoder_type='mlp',
            grammar=grammar
        )
        
        # Create trainer
        trainer = PPOTrainer(
            policy=policy,
            env=env,
            n_epochs=1,
            batch_size=16,
            learning_rate=1e-3
        )
        
        # Train for a short time
        history = trainer.train(
            total_timesteps=64,
            rollout_length=32,
            log_interval=1
        )
        
        # Verify training occurred
        self.assertEqual(len(history['loss']), 2)
        self.assertEqual(trainer.total_timesteps, 64)
        self.assertGreater(trainer.n_updates, 0)
    
    def test_checkpointing(self):
        """Test save and load functionality."""
        # Create trainer with checkpoint directory
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                policy=self.mock_policy,
                env=self.mock_env,
                checkpoint_dir=tmpdir
            )
            
            # Set some state
            trainer.total_timesteps = 1000
            trainer.n_updates = 10
            trainer.episode_rewards.extend([1.0, 2.0, 3.0])
            
            # Save checkpoint
            success = trainer.save_checkpoint()
            self.assertTrue(success)
            
            # Create new trainer and load
            new_trainer = PPOTrainer(
                policy=self.mock_policy,
                env=self.mock_env,
                checkpoint_dir=tmpdir
            )
            
            # Load checkpoint
            state = new_trainer.load_from_checkpoint()
            
            # Verify state was restored
            self.assertIsNotNone(state)
            self.assertEqual(new_trainer.total_timesteps, 1000)
            self.assertEqual(new_trainer.n_updates, 10)
            self.assertEqual(list(new_trainer.episode_rewards), [1.0, 2.0, 3.0])


class TestAdvancedUseCases(unittest.TestCase):
    """Test advanced use cases enabled by the refactoring."""
    
    def test_custom_collection(self):
        """Test custom data collection."""
        class CustomCollectionTrainer(PPOTrainer):
            def collect_rollouts(self, n_steps, custom_param=None):
                # Custom collection logic
                data = super().collect_rollouts(n_steps)
                
                # Add custom data
                data['custom_field'] = custom_param
                
                return data
        
        # Use custom trainer
        trainer = CustomCollectionTrainer(
            policy=Mock(),
            env=Mock()
        )
        
        # Mock parent method
        trainer.rollout_buffer = Mock()
        trainer.rollout_buffer.get.return_value = {'observations': torch.randn(10, 5)}
        
        # Collect with custom parameter
        data = trainer.collect_rollouts(10, custom_param="test")
        
        self.assertEqual(data['custom_field'], "test")
    
    def test_offline_learning(self):
        """Test learning from pre-collected data."""
        # Create trainer (no environment interaction needed)
        trainer = PPOTrainer(
            policy=Mock(),
            env=Mock()
        )
        
        # Load pre-collected data
        offline_data = {
            'observations': torch.randn(1000, 10),
            'actions': torch.randint(0, 5, (1000,)),
            'rewards': torch.randn(1000),
            'values': torch.randn(1000),
            'log_probs': torch.randn(1000),
            'advantages': torch.randn(1000),
            'returns': torch.randn(1000),
            'action_masks': torch.ones(1000, 5, dtype=torch.bool),
            'tree_structures': [None] * 1000
        }
        
        # Learn from offline data
        metrics = trainer.learn(offline_data)
        
        # No environment interaction should occur
        trainer.env.step.assert_not_called()
        trainer.env.reset.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)