# JanusAI/tests/unit/ml/rewards/test_reward_handler.py
"""
Test Suite for the Formalized Reward System
==========================================

Comprehensive tests for RewardHandler and related components.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import json

from JanusAI.ml.rewards.reward_handler import RewardHandler, AdaptiveRewardHandler, RewardInfo
from JanusAI.ml.rewards.base_reward import BaseReward
from JanusAI.ml.rewards.reward_registry import (
    register_reward,
    create_reward_component,
    create_handler_from_preset,
    get_preset_config,
    REWARD_REGISTRY
)


class MockReward(BaseReward):
    """Mock reward component for testing."""
    
    def __init__(self, weight: float = 1.0, return_value: float = 0.5):
        super().__init__(weight)
        self.return_value = return_value
        self.call_count = 0
    
    def calculate_reward(self, **kwargs) -> float:
        self.call_count += 1
        self.last_kwargs = kwargs
        return self.return_value


class TestRewardHandler(unittest.TestCase):
    """Test the RewardHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_reward1 = MockReward(return_value=0.5)
        self.mock_reward2 = MockReward(return_value=-0.3)
        
        self.handler = RewardHandler(
            reward_components={
                self.mock_reward1: 0.6,
                self.mock_reward2: 0.4
            },
            normalize=False,
            log_rewards=False
        )
        
        self.test_kwargs = {
            'current_observation': np.array([1, 2, 3]),
            'action': 1,
            'next_observation': np.array([2, 3, 4]),
            'reward_from_env': 0.1,
            'done': False,
            'info': {'expression': 'test'}
        }
    
    def test_initialization(self):
        """Test handler initialization."""
        self.assertEqual(len(self.handler.components), 2)
        self.assertEqual(len(self.handler.weights), 2)
        self.assertFalse(self.handler.normalize)
        self.assertIsNone(self.handler.clip_range)
    
    def test_calculate_total_reward(self):
        """Test total reward calculation."""
        reward = self.handler.calculate_total_reward(**self.test_kwargs)
        
        # Expected: 0.6 * 0.5 + 0.4 * (-0.3) = 0.3 - 0.12 = 0.18
        self.assertAlmostEqual(reward, 0.18, places=5)
        
        # Check components were called
        self.assertEqual(self.mock_reward1.call_count, 1)
        self.assertEqual(self.mock_reward2.call_count, 1)
    
    def test_calculate_detailed_reward(self):
        """Test detailed reward calculation."""
        reward_info = self.handler.calculate_detailed_reward(**self.test_kwargs)
        
        # Check structure
        self.assertIsInstance(reward_info, RewardInfo)
        self.assertAlmostEqual(reward_info.total_reward, 0.18, places=5)
        
        # Check component rewards
        self.assertEqual(len(reward_info.component_rewards), 2)
        self.assertIn('MockReward', str(reward_info.component_rewards.keys()))
        
        # Check weighted rewards
        self.assertEqual(len(reward_info.weighted_rewards), 2)
        
        # Check metadata
        self.assertFalse(reward_info.metadata['normalized'])
        self.assertFalse(reward_info.metadata['clipped'])
    
    def test_normalization(self):
        """Test reward normalization."""
        # Create handler with normalization
        handler = RewardHandler(
            reward_components={
                MockReward(return_value=10.0): 0.5,
                MockReward(return_value=-5.0): 0.5
            },
            normalize=True
        )
        
        reward_info = handler.calculate_detailed_reward(**self.test_kwargs)
        
        # Check that large rewards are normalized
        for reward in reward_info.component_rewards.values():
            self.assertLessEqual(abs(reward), 1.0)
    
    def test_clipping(self):
        """Test reward clipping."""
        handler = RewardHandler(
            reward_components={
                MockReward(return_value=10.0): 1.0
            },
            clip_range=(-1.0, 1.0)
        )
        
        reward = handler.calculate_total_reward(**self.test_kwargs)
        self.assertEqual(reward, 1.0)  # Clipped to max
        
        # Test negative clipping
        handler.components[list(handler.components.keys())[0]].return_value = -10.0
        reward = handler.calculate_total_reward(**self.test_kwargs)
        self.assertEqual(reward, -1.0)  # Clipped to min
    
    def test_update_weights(self):
        """Test weight updating."""
        original_weights = dict(self.handler.weights)
        
        # Update weights
        new_weights = {
            list(self.handler.weights.keys())[0]: 0.8,
            list(self.handler.weights.keys())[1]: 0.2
        }
        self.handler.update_weights(new_weights)
        
        # Check weights changed
        self.assertNotEqual(dict(self.handler.weights), original_weights)
        
        # Calculate with new weights
        reward = self.handler.calculate_total_reward(**self.test_kwargs)
        # Expected: 0.8 * 0.5 + 0.2 * (-0.3) = 0.4 - 0.06 = 0.34
        self.assertAlmostEqual(reward, 0.34, places=5)
    
    def test_add_remove_component(self):
        """Test dynamic component management."""
        # Add component
        new_component = MockReward(return_value=1.0)
        self.handler.add_component('new', new_component, weight=0.5)
        
        self.assertEqual(len(self.handler.components), 3)
        self.assertIn('new', self.handler.weights)
        
        # Remove component
        self.handler.remove_component('new')
        self.assertEqual(len(self.handler.components), 2)
        self.assertNotIn('new', self.handler.weights)
    
    def test_error_handling(self):
        """Test error handling in reward calculation."""
        # Create component that raises error
        class ErrorReward(BaseReward):
            def calculate_reward(self, **kwargs):
                raise ValueError("Test error")
        
        handler = RewardHandler(
            reward_components={
                ErrorReward(): 0.5,
                MockReward(): 0.5
            },
            log_rewards=False
        )
        
        # Should handle error gracefully
        reward = handler.calculate_total_reward(**self.test_kwargs)
        self.assertEqual(reward, 0.25)  # Only MockReward contributes


class TestAdaptiveRewardHandler(unittest.TestCase):
    """Test the AdaptiveRewardHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = AdaptiveRewardHandler(
            reward_components={
                MockReward(return_value=0.8): 0.5,
                MockReward(return_value=0.2): 0.5
            },
            adaptation_rate=0.1,
            target_balance={'MockReward': 0.7},
            log_rewards=False
        )
    
    def test_adaptation(self):
        """Test weight adaptation."""
        initial_weights = dict(self.handler.weights)
        
        # Simulate multiple episodes
        for _ in range(10):
            # Simulate episode steps
            for _ in range(5):
                self.handler.calculate_detailed_reward(
                    current_observation=np.random.randn(5),
                    action=0,
                    next_observation=np.random.randn(5),
                    reward_from_env=0.0,
                    done=False,
                    info={}
                )
            
            # Trigger adaptation on episode end
            self.handler.calculate_detailed_reward(
                current_observation=np.random.randn(5),
                action=0,
                next_observation=np.random.randn(5),
                reward_from_env=0.0,
                done=True,
                info={}
            )
        
        # Weights should have changed
        self.assertNotEqual(dict(self.handler.weights), initial_weights)


class TestRewardRegistry(unittest.TestCase):
    """Test the reward registry system."""
    
    def test_registration(self):
        """Test reward component registration."""
        # Register a test component
        @register_reward("test_reward")
        class TestReward(BaseReward):
            def calculate_reward(self, **kwargs):
                return 0.0
        
        self.assertIn("test_reward", REWARD_REGISTRY)
        
        # Create component
        component = create_reward_component("test_reward")
        self.assertIsInstance(component, TestReward)
    
    def test_preset_configs(self):
        """Test preset configurations."""
        # Get preset
        config = get_preset_config('physics_discovery')
        
        self.assertIn('components', config)
        self.assertIn('weights', config)
        self.assertIn('conservation', config['weights'])
        
        # Create handler from preset
        handler = create_handler_from_preset('physics_discovery')
        self.assertIsInstance(handler, RewardHandler)
        self.assertIn('conservation', handler.get_component_names())
    
    def test_from_config(self):
        """Test creating handler from configuration."""
        config = {
            'components': ['novelty', 'complexity'],
            'weights': {
                'novelty': 0.6,
                'complexity': 0.4
            },
            'normalize': True,
            'clip_range': (-5.0, 5.0)
        }
        
        handler = RewardHandler.from_config(config)
        
        self.assertEqual(len(handler.components), 2)
        self.assertTrue(handler.normalize)
        self.assertEqual(handler.clip_range, (-5.0, 5.0))


class TestIntegration(unittest.TestCase):
    """Integration tests with real components."""
    
    def test_physics_discovery_scenario(self):
        """Test physics discovery reward scenario."""
        from janus.ml.rewards.intrinsic_rewards import (
            NoveltyReward,
            ComplexityReward,
            ConservationLawReward
        )
        
        handler = RewardHandler(
            reward_components={
                NoveltyReward(): 0.2,
                ComplexityReward(target_complexity=5): 0.3,
                ConservationLawReward(law_type='energy'): 0.5
            }
        )
        
        # Simulate discovering F=ma
        info = {
            'expression': 'F = m * a',
            'complexity': 3,
            'variables': ['F', 'm', 'a'],
            'trajectory_data': {
                'states': np.array([[1, 2, 2], [1.1, 2.2, 2.2]]),
                'actions': np.array([0.1])
            }
        }
        
        reward = handler.calculate_total_reward(
            current_observation=np.zeros(10),
            action=0,
            next_observation=np.zeros(10),
            reward_from_env=0.0,
            done=True,
            info=info
        )
        
        # Should get positive reward for simple, novel expression
        self.assertGreater(reward, 0)
    
    def test_configuration_serialization(self):
        """Test saving and loading configurations."""
        # Create handler
        handler = create_handler_from_preset('ai_interpretability')
        
        # Get configuration
        config = handler.get_config()
        
        # Save to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        # Load and recreate
        with open(temp_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Create new handler from loaded config
        # Note: This would need slight modification of from_config to handle
        # the format returned by get_config
        self.assertEqual(config['components'], loaded_config['components'])
        self.assertEqual(config['weights'], loaded_config['weights'])
        
        # Clean up
        import os
        os.unlink(temp_path)


class TestRewardHandlerWithEnvironment(unittest.TestCase):
    """Test RewardHandler integration with environments."""
    
    def test_environment_integration(self):
        """Test using RewardHandler in an environment."""
        from janus.environments.base.symbolic_env import SymbolicDiscoveryEnv
        from janus.core.grammar.base_grammar import ProgressiveGrammar
        from janus.core.expressions.expression import Variable
        
        # Create simple environment
        grammar = ProgressiveGrammar()
        variables = [Variable('x', 0)]
        X = np.random.randn(10, 1)
        y = 2 * X[:, 0] + 1
        
        # Create reward handler
        reward_handler = create_handler_from_preset('balanced_exploration')
        
        # Create environment with handler
        env = SymbolicDiscoveryEnv(
            grammar=grammar,
            X_data=X,
            y_data=y,
            variables=variables,
            reward_handler=reward_handler
        )
        
        # Test that reward calculation works
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Should get some reward value
        self.assertIsInstance(reward, (int, float))


if __name__ == '__main__':
    unittest.main(verbosity=2)