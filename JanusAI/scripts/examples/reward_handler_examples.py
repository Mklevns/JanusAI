# JanusAI/scripts/examples/reward_handler_examples.py
"""
Examples: Using the Formalized Reward System
============================================

This module demonstrates various ways to use the RewardHandler
for different Janus project scenarios.
"""

import numpy as np
from typing import Dict, Any

# Import reward components and handler
from JanusAI.ml.rewards.reward_handler import RewardHandler, AdaptiveRewardHandler
from JanusAI.ml.rewards.reward_registry import (
    create_handler_from_preset,
    create_reward_component,
    list_available_rewards
)
from JanusAI.ml.rewards.intrinsic_rewards import NoveltyReward, ComplexityReward
from JanusAI.ml.rewards.interpretability_reward import InterpretabilityReward


# Example 1: Basic Usage with Direct Components
def example_basic_usage():
    """Basic example of creating and using a RewardHandler."""
    print("=== Example 1: Basic Usage ===\n")
    
    # Create reward components
    novelty_reward = NoveltyReward(weight=1.0, novelty_threshold=0.1)
    complexity_reward = ComplexityReward(weight=1.0, target_complexity=10)
    
    # Create handler with weighted components
    handler = RewardHandler(
        reward_components={
            novelty_reward: 0.6,      # 60% weight on novelty
            complexity_reward: 0.4    # 40% weight on complexity
        },
        normalize=False,
        log_rewards=True
    )
    
    # Simulate environment step
    info = {
        'expression': 'x + sin(y)',
        'complexity': 8,
        'variables': []
    }
    
    # Calculate reward
    total_reward = handler.calculate_total_reward(
        current_observation=np.array([1, 2, 3]),
        action=0,
        next_observation=np.array([1, 2, 4]),
        reward_from_env=0.1,
        done=False,
        info=info
    )
    
    print(f"Total reward: {total_reward:.4f}\n")


# Example 2: Using String Names and Registry
def example_registry_usage():
    """Example using the reward registry system."""
    print("=== Example 2: Registry Usage ===\n")
    
    # List available rewards
    available = list_available_rewards()
    print("Available reward components:")
    for name, desc in available.items():
        print(f"  - {name}: {desc}")
    print()
    
    # Create handler using string names
    handler = RewardHandler(
        reward_components={
            "novelty": 0.3,
            "complexity": 0.3,
            "conservation": 0.4
        },
        normalize=True,
        clip_range=(-5.0, 5.0)
    )
    
    # Get configuration
    config = handler.get_config()
    print(f"Handler configuration: {config}\n")


# Example 3: Physics Discovery Scenario
def example_physics_discovery():
    """Example for physics law discovery."""
    print("=== Example 3: Physics Discovery ===\n")
    
    # Use preset configuration
    handler = create_handler_from_preset(
        'physics_discovery',
        # Override specific values
        weights={'conservation': 0.6, 'complexity': 0.2, 'novelty': 0.2}
    )
    
    # Simulate discovering F = ma
    info = {
        'expression': 'm * a',
        'complexity': 3,
        'variables': ['m', 'a'],
        'trajectory_data': {
            'states': np.array([[1, 2], [1.1, 2.2], [1.2, 2.4]]),
            'actions': np.array([0.1, 0.1])
        }
    }
    
    # Get detailed reward breakdown
    reward_info = handler.calculate_detailed_reward(
        current_observation=np.array([1, 2]),
        action=0,
        next_observation=np.array([1.1, 2.2]),
        reward_from_env=0.5,
        done=False,
        info=info
    )
    
    print(f"Total reward: {reward_info.total_reward:.4f}")
    print("\nComponent breakdown:")
    for name, reward in reward_info.component_rewards.items():
        weighted = reward_info.weighted_rewards[name]
        print(f"  {name}: {reward:.4f} (weighted: {weighted:.4f})")
    print()


# Example 4: AI Interpretability Scenario
def example_ai_interpretability():
    """Example for AI model interpretability."""
    print("=== Example 4: AI Interpretability ===\n")
    
    # Create custom interpretability-focused handler
    handler = RewardHandler(
        reward_components={
            InterpretabilityReward(
                fidelity_weight=0.5,
                simplicity_weight=0.3,
                consistency_weight=0.1,
                insight_weight=0.1
            ): 0.7,
            NoveltyReward(): 0.3
        },
        normalize=True
    )
    
    # Simulate explaining neural network behavior
    info = {
        'expression': 'sigmoid(0.5 * x1 + 0.3 * x2)',
        'complexity': 7,
        'variables': ['x1', 'x2'],
        'model_predictions': np.array([0.7, 0.8, 0.6]),
        'expression_predictions': np.array([0.71, 0.79, 0.62])
    }
    
    reward = handler.calculate_total_reward(
        current_observation=np.zeros(10),
        action=1,
        next_observation=np.zeros(10),
        reward_from_env=0.0,
        done=True,
        info=info
    )
    
    print(f"Interpretability reward: {reward:.4f}\n")


# Example 5: Adaptive Reward Handler
def example_adaptive_handler():
    """Example using adaptive weight adjustment."""
    print("=== Example 5: Adaptive Handler ===\n")
    
    # Create adaptive handler with target balance
    handler = AdaptiveRewardHandler(
        reward_components={
            "novelty": 0.5,
            "complexity": 0.5
        },
        adaptation_rate=0.05,
        target_balance={
            "novelty": 0.3,      # Want 30% contribution from novelty
            "complexity": 0.7    # Want 70% contribution from complexity
        }
    )
    
    print("Initial weights:", dict(handler.weights))
    
    # Simulate several episodes
    for episode in range(5):
        total_reward = 0.0
        
        # Simulate steps in episode
        for step in range(10):
            info = {
                'expression': f'x^{step}',
                'complexity': step + 1
            }
            
            reward = handler.calculate_total_reward(
                current_observation=np.random.randn(5),
                action=np.random.randint(0, 10),
                next_observation=np.random.randn(5),
                reward_from_env=np.random.randn(),
                done=(step == 9),
                info=info
            )
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total reward = {total_reward:.2f}, "
              f"Weights = {dict(handler.weights)}")
    
    print("\nWeights adapted based on contribution balance\n")


# Example 6: Environment Integration
def example_environment_integration():
    """Example of integrating RewardHandler with an environment."""
    print("=== Example 6: Environment Integration ===\n")
    
    from janus.environments.base.symbolic_env import SymbolicDiscoveryEnv
    
    class RewardHandlerEnv(SymbolicDiscoveryEnv):
        """Environment that uses RewardHandler for reward calculation."""
        
        def __init__(self, *args, reward_handler: RewardHandler = None, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Use provided handler or create default
            self.reward_handler = reward_handler or create_handler_from_preset(
                'balanced_exploration'
            )
        
        def _calculate_reward(self, action_valid: bool) -> float:
            """Override reward calculation to use RewardHandler."""
            # Get base reward from parent
            base_reward = super()._calculate_reward(action_valid)
            
            # Prepare info for reward handler
            info = self._get_info()
            info['action_valid'] = action_valid
            
            # Calculate total reward using handler
            total_reward = self.reward_handler.calculate_total_reward(
                current_observation=self._get_observation(),
                action=self._last_action if hasattr(self, '_last_action') else 0,
                next_observation=self._get_observation(),
                reward_from_env=base_reward,
                done=self._is_terminated(),
                info=info
            )
            
            return total_reward
    
    print("Created RewardHandlerEnv that uses centralized reward system\n")


# Example 7: Configuration-Based Setup
def example_config_based():
    """Example of configuration-based reward system setup."""
    print("=== Example 7: Configuration-Based Setup ===\n")
    
    # Define configuration (could come from YAML/JSON file)
    config = {
        'components': {
            'novelty': {'novelty_threshold': 0.15, 'history_size': 200},
            'complexity': {'target_complexity': 8, 'tolerance': 3},
            'fidelity': {}  # Use factory defaults
        },
        'weights': {
            'novelty': 0.25,
            'complexity': 0.25,
            'fidelity': 0.5
        },
        'normalize': True,
        'clip_range': (-10.0, 10.0),
        'log_rewards': True
    }
    
    # Create handler from config
    handler = RewardHandler.from_config(config)
    
    print(f"Created handler with {len(handler.components)} components")
    print(f"Components: {handler.get_component_names()}")
    print(f"Weights: {dict(handler.weights)}\n")
    
    # Can also save configuration
    saved_config = handler.get_config()
    print(f"Saved configuration: {saved_config}\n")


# Example 8: Dynamic Component Management
def example_dynamic_management():
    """Example of dynamically managing reward components."""
    print("=== Example 8: Dynamic Management ===\n")
    
    # Start with simple handler
    handler = RewardHandler(
        reward_components={"novelty": 1.0},
        normalize=False
    )
    
    print(f"Initial components: {handler.get_component_names()}")
    
    # Add complexity reward dynamically
    complexity_reward = ComplexityReward(target_complexity=5)
    handler.add_component("complexity", complexity_reward, weight=0.5)
    
    print(f"After adding: {handler.get_component_names()}")
    
    # Update weights
    handler.update_weights({"novelty": 0.7, "complexity": 0.3})
    print(f"Updated weights: {dict(handler.weights)}")
    
    # Remove component
    handler.remove_component("complexity")
    print(f"After removal: {handler.get_component_names()}\n")


# Example 9: Custom Reward Component
def example_custom_component():
    """Example of creating a custom reward component."""
    print("=== Example 9: Custom Component ===\n")
    
    from janus.ml.rewards.base_reward import BaseReward
    
    class LengthPenaltyReward(BaseReward):
        """Penalizes expressions that are too long."""
        
        def __init__(self, weight: float = 1.0, max_length: int = 50):
            super().__init__(weight)
            self.max_length = max_length
        
        def calculate_reward(self, **kwargs) -> float:
            info = kwargs.get('info', {})
            expression = info.get('expression', '')
            
            length = len(str(expression))
            if length > self.max_length:
                penalty = -((length - self.max_length) / self.max_length)
                return penalty
            return 0.0
    
    # Use custom component in handler
    handler = RewardHandler(
        reward_components={
            "novelty": 0.4,
            "complexity": 0.4,
            LengthPenaltyReward(max_length=30): 0.2
        }
    )
    
    info = {'expression': 'x' * 40}  # Long expression
    
    reward = handler.calculate_total_reward(
        current_observation=np.zeros(5),
        action=0,
        next_observation=np.zeros(5),
        reward_from_env=0.0,
        done=False,
        info=info
    )
    
    print(f"Reward with length penalty: {reward:.4f}\n")


def main():
    """Run all examples."""
    examples = [
        example_basic_usage,
        example_registry_usage,
        example_physics_discovery,
        example_ai_interpretability,
        example_adaptive_handler,
        example_environment_integration,
        example_config_based,
        example_dynamic_management,
        example_custom_component
    ]
    
    for example in examples:
        example()
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()