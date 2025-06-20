# JanusAI/ml/rewards/reward_registry.py
"""
Reward Component Registry
========================

This module provides a registry system for reward components, making it easy
to create and configure reward components by name.
"""

from typing import Dict, Type, Any, Optional, Callable
import inspect
from functools import partial

from janus.ml.rewards.base_reward import BaseReward
from janus.ml.rewards.intrinsic_rewards import (
    NoveltyReward,
    ComplexityReward,
    ConservationLawReward
)
from janus.ml.rewards.interpretability_reward import InterpretabilityReward


# Global registry for reward components
REWARD_REGISTRY: Dict[str, Type[BaseReward]] = {}
REWARD_FACTORIES: Dict[str, Callable] = {}


def register_reward(name: str, reward_class: Type[BaseReward] = None):
    """
    Decorator to register a reward component class.
    
    Usage:
        @register_reward("novelty")
        class NoveltyReward(BaseReward):
            ...
    
    Or:
        register_reward("novelty", NoveltyReward)
    """
    def decorator(cls):
        REWARD_REGISTRY[name] = cls
        return cls
    
    if reward_class is None:
        # Used as decorator
        return decorator
    else:
        # Direct registration
        REWARD_REGISTRY[name] = reward_class
        return reward_class


def register_reward_factory(name: str, factory: Callable):
    """
    Register a factory function for creating reward components.
    
    This is useful for components that require complex initialization.
    
    Args:
        name: Name for the factory
        factory: Callable that returns a BaseReward instance
    """
    REWARD_FACTORIES[name] = factory


def create_reward_component(name: str, **kwargs) -> BaseReward:
    """
    Create a reward component by name.
    
    Args:
        name: Name of the reward component
        **kwargs: Arguments passed to the component constructor
        
    Returns:
        BaseReward instance
    """
    # Check factories first
    if name in REWARD_FACTORIES:
        return REWARD_FACTORIES[name](**kwargs)
    
    # Check registry
    if name in REWARD_REGISTRY:
        reward_class = REWARD_REGISTRY[name]
        return reward_class(**kwargs)
    
    raise ValueError(f"Unknown reward component: {name}. "
                    f"Available: {list(REWARD_REGISTRY.keys()) + list(REWARD_FACTORIES.keys())}")


def get_reward_class(name: str) -> Type[BaseReward]:
    """Get the reward class for a given name."""
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward component: {name}")
    return REWARD_REGISTRY[name]


def list_available_rewards() -> Dict[str, str]:
    """
    List all available reward components with their descriptions.
    
    Returns:
        Dictionary mapping component names to descriptions
    """
    available = {}
    
    # From registry
    for name, cls in REWARD_REGISTRY.items():
        doc = cls.__doc__ or "No description available"
        available[name] = doc.strip().split('\n')[0]
    
    # From factories
    for name, factory in REWARD_FACTORIES.items():
        doc = factory.__doc__ or "No description available"
        available[name] = doc.strip().split('\n')[0]
    
    return available


# Register built-in reward components
register_reward("novelty", NoveltyReward)
register_reward("complexity", ComplexityReward)
register_reward("conservation", ConservationLawReward)
register_reward("interpretability", InterpretabilityReward)


# Define some useful factory functions
def create_fidelity_reward(weight: float = 1.0, **kwargs) -> BaseReward:
    """Create a fidelity-focused interpretability reward."""
    return InterpretabilityReward(
        weight=weight,
        fidelity_weight=0.8,
        simplicity_weight=0.1,
        consistency_weight=0.05,
        insight_weight=0.05,
        **kwargs
    )


def create_simplicity_reward(weight: float = 1.0, **kwargs) -> BaseReward:
    """Create a simplicity-focused interpretability reward."""
    return InterpretabilityReward(
        weight=weight,
        fidelity_weight=0.2,
        simplicity_weight=0.6,
        consistency_weight=0.1,
        insight_weight=0.1,
        **kwargs
    )


def create_physics_reward(weight: float = 1.0, **kwargs) -> BaseReward:
    """Create a physics-focused conservation reward."""
    return ConservationLawReward(
        weight=weight,
        law_type='energy',
        data_key='trajectory_data',
        **kwargs
    )


# Register factories
register_reward_factory("fidelity", create_fidelity_reward)
register_reward_factory("simplicity", create_simplicity_reward)
register_reward_factory("physics", create_physics_reward)


# Configuration-based reward creation
class RewardConfig:
    """Configuration class for reward components."""
    
    def __init__(self, component_type: str, **params):
        self.component_type = component_type
        self.params = params
    
    def create(self) -> BaseReward:
        """Create the reward component from this configuration."""
        return create_reward_component(self.component_type, **self.params)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'component_type': self.component_type,
            'params': self.params
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'RewardConfig':
        """Create from dictionary representation."""
        return cls(
            component_type=config['component_type'],
            **config.get('params', {})
        )


# Preset reward configurations
REWARD_PRESETS = {
    'physics_discovery': {
        'components': {
            'conservation': {'law_type': 'energy', 'weight': 1.0},
            'complexity': {'target_complexity': 10, 'tolerance': 5},
            'novelty': {'novelty_threshold': 0.1}
        },
        'weights': {
            'conservation': 0.5,
            'complexity': 0.3,
            'novelty': 0.2
        },
        'normalize': True
    },
    
    'ai_interpretability': {
        'components': {
            'interpretability': {
                'fidelity_weight': 0.4,
                'simplicity_weight': 0.3,
                'consistency_weight': 0.2,
                'insight_weight': 0.1
            },
            'novelty': {'novelty_threshold': 0.15}
        },
        'weights': {
            'interpretability': 0.7,
            'novelty': 0.3
        },
        'normalize': True,
        'clip_range': (-10.0, 10.0)
    },
    
    'balanced_exploration': {
        'components': ['novelty', 'complexity', 'conservation'],
        'weights': {
            'novelty': 0.4,
            'complexity': 0.3,
            'conservation': 0.3
        },
        'normalize': False
    },
    
    'curriculum_learning': {
        'components': {
            'complexity': {'target_complexity': 5, 'tolerance': 2},
            'fidelity': {},
            'novelty': {'history_size': 500}
        },
        'weights': {
            'complexity': 0.2,
            'fidelity': 0.6,
            'novelty': 0.2
        },
        'normalize': True,
        'clip_range': (-5.0, 5.0)
    }
}


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    Get a preset reward configuration.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Configuration dictionary
    """
    if preset_name not in REWARD_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available: {list(REWARD_PRESETS.keys())}")
    
    return REWARD_PRESETS[preset_name].copy()


def create_handler_from_preset(preset_name: str, **overrides) -> 'RewardHandler':
    """
    Create a RewardHandler from a preset configuration.
    
    Args:
        preset_name: Name of the preset
        **overrides: Override specific configuration values
        
    Returns:
        RewardHandler instance
    """
    from janus.ml.rewards.reward_handler import RewardHandler
    
    config = get_preset_config(preset_name)
    
    # Apply overrides
    for key, value in overrides.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    return RewardHandler.from_config(config)