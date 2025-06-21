# JanusAI/ml/rewards/reward_handler.py
"""
Reward Handler - Centralized Reward System
==========================================

This module provides a flexible, composable reward system that combines
multiple reward components with configurable weights.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import OrderedDict
import logging
from dataclasses import dataclass

from janus_ai.ml.rewards.base_reward import BaseReward


@dataclass
class RewardInfo:
    """Container for detailed reward calculation information."""
    total_reward: float
    component_rewards: Dict[str, float]
    component_weights: Dict[str, float]
    weighted_rewards: Dict[str, float]
    metadata: Dict[str, Any]


class RewardHandler:
    """
    Centralized reward handler that composes multiple reward components.
    
    This handler allows for flexible experimentation with different reward
    formulations by combining weighted BaseReward components.
    """
    
    def __init__(
        self,
        reward_components: Dict[Union[str, BaseReward], float],
        normalize: bool = False,
        clip_range: Optional[Tuple[float, float]] = None,
        log_rewards: bool = True
    ):
        """
        Initialize the reward handler.
        
        Args:
            reward_components: Dictionary mapping reward components to their weights.
                              Keys can be either BaseReward instances or string names.
                              Example: {FidelityReward(): 0.5, NoveltyReward(): 0.3}
                              or {"fidelity": 0.5, "novelty": 0.3} with registry
            normalize: Whether to normalize component rewards before weighting
            clip_range: Optional (min, max) range to clip the total reward
            log_rewards: Whether to log detailed reward information
        """
        self.components = OrderedDict()
        self.weights = OrderedDict()
        self.normalize = normalize
        self.clip_range = clip_range
        self.log_rewards = log_rewards
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Process reward components
        for component, weight in reward_components.items():
            if isinstance(component, str):
                # String name - look up in registry
                component_instance = self._get_component_from_registry(component)
                component_name = component
            else:
                # Direct BaseReward instance
                component_instance = component
                component_name = component.__class__.__name__
            
            self.components[component_name] = component_instance
            self.weights[component_name] = weight
        
        # Validate weights sum to 1.0 if requested
        self._validate_configuration()
    
    def _get_component_from_registry(self, name: str) -> BaseReward:
        """Get reward component from registry by name."""
        # Import here to avoid circular imports
        from janus_ai.ml.rewards.reward_registry import REWARD_REGISTRY
        
        if name not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward component: {name}. "
                           f"Available: {list(REWARD_REGISTRY.keys())}")
        
        return REWARD_REGISTRY[name]
    
    def _validate_configuration(self):
        """Validate the reward handler configuration."""
        # Check that all components are BaseReward instances
        for name, component in self.components.items():
            if not isinstance(component, BaseReward):
                raise TypeError(f"Component '{name}' must be a BaseReward instance, "
                              f"got {type(component)}")
        
        # Log configuration
        if self.log_rewards:
            self.logger.info(f"RewardHandler initialized with {len(self.components)} components:")
            for name, weight in self.weights.items():
                self.logger.info(f"  - {name}: weight={weight:.3f}")
    
    def calculate_total_reward(
        self,
        current_observation: Any,
        action: Any,
        next_observation: Any,
        reward_from_env: float,
        done: bool,
        info: Dict[str, Any]
    ) -> float:
        """
        Calculate the total weighted reward from all components.
        
        This is the simple interface that returns just the total reward value.
        
        Args:
            current_observation: Current state/observation
            action: Action taken
            next_observation: Next state/observation
            reward_from_env: Base reward from environment
            done: Whether episode is done
            info: Additional information dictionary
            
        Returns:
            Total weighted reward
        """
        reward_info = self.calculate_detailed_reward(
            current_observation, action, next_observation,
            reward_from_env, done, info
        )
        return reward_info.total_reward
    
    def calculate_detailed_reward(
        self,
        current_observation: Any,
        action: Any,
        next_observation: Any,
        reward_from_env: float,
        done: bool,
        info: Dict[str, Any]
    ) -> RewardInfo:
        """
        Calculate detailed reward information from all components.
        
        This method provides complete information about the reward calculation,
        including individual component contributions.
        
        Returns:
            RewardInfo object with detailed breakdown
        """
        component_rewards = OrderedDict()
        weighted_rewards = OrderedDict()
        metadata = {
            'normalized': self.normalize,
            'clipped': False,
            'original_env_reward': reward_from_env
        }
        
        # Calculate individual component rewards
        for name, component in self.components.items():
            try:
                # Each component calculates its own reward
                component_reward = component.calculate_reward(
                    current_observation=current_observation,
                    action=action,
                    next_observation=next_observation,
                    reward_from_env=reward_from_env,
                    done=done,
                    info=info
                )
                component_rewards[name] = component_reward
                
            except Exception as e:
                self.logger.warning(f"Error calculating {name} reward: {e}")
                component_rewards[name] = 0.0
        
        # Normalize if requested
        if self.normalize:
            component_rewards = self._normalize_rewards(component_rewards)
        
        # Apply weights
        total_reward = 0.0
        for name, reward in component_rewards.items():
            weight = self.weights[name]
            weighted_reward = weight * reward
            weighted_rewards[name] = weighted_reward
            total_reward += weighted_reward
        
        # Apply clipping if specified
        if self.clip_range is not None:
            original_total = total_reward
            total_reward = np.clip(total_reward, self.clip_range[0], self.clip_range[1])
            if total_reward != original_total:
                metadata['clipped'] = True
                metadata['original_total'] = original_total
        
        # Log if enabled
        if self.log_rewards:
            self._log_reward_details(component_rewards, weighted_rewards, total_reward)
        
        return RewardInfo(
            total_reward=total_reward,
            component_rewards=component_rewards,
            component_weights=dict(self.weights),
            weighted_rewards=weighted_rewards,
            metadata=metadata
        )
    
    def _normalize_rewards(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Normalize reward components to [-1, 1] range."""
        normalized = OrderedDict()
        
        for name, reward in rewards.items():
            # Simple normalization - can be made more sophisticated
            if abs(reward) > 1.0:
                normalized[name] = np.sign(reward) * np.tanh(abs(reward))
            else:
                normalized[name] = reward
        
        return normalized
    
    def _log_reward_details(
        self,
        component_rewards: Dict[str, float],
        weighted_rewards: Dict[str, float],
        total_reward: float
    ):
        """Log detailed reward calculation information."""
        self.logger.debug(f"Reward calculation:")
        self.logger.debug(f"  Total: {total_reward:.4f}")
        for name in component_rewards:
            self.logger.debug(f"  {name}: {component_rewards[name]:.4f} "
                            f"(weighted: {weighted_rewards[name]:.4f})")
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update component weights dynamically.
        
        Args:
            new_weights: Dictionary of component names to new weights
        """
        for name, weight in new_weights.items():
            if name not in self.weights:
                self.logger.warning(f"Unknown component '{name}' in weight update")
                continue
            self.weights[name] = weight
        
        if self.log_rewards:
            self.logger.info("Updated reward weights:")
            for name, weight in self.weights.items():
                self.logger.info(f"  - {name}: {weight:.3f}")
    
    def add_component(self, name: str, component: BaseReward, weight: float):
        """
        Add a new reward component dynamically.
        
        Args:
            name: Name for the component
            component: BaseReward instance
            weight: Weight for the component
        """
        if not isinstance(component, BaseReward):
            raise TypeError(f"Component must be a BaseReward instance")
        
        self.components[name] = component
        self.weights[name] = weight
        
        if self.log_rewards:
            self.logger.info(f"Added new component '{name}' with weight {weight:.3f}")
    
    def remove_component(self, name: str):
        """Remove a reward component."""
        if name in self.components:
            del self.components[name]
            del self.weights[name]
            if self.log_rewards:
                self.logger.info(f"Removed component '{name}'")
    
    def get_component_names(self) -> List[str]:
        """Get list of all component names."""
        return list(self.components.keys())
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as a dictionary."""
        return {
            'components': list(self.components.keys()),
            'weights': dict(self.weights),
            'normalize': self.normalize,
            'clip_range': self.clip_range,
            'log_rewards': self.log_rewards
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RewardHandler':
        """
        Create a RewardHandler from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with keys:
                    - components: List of component names or dict of name->params
                    - weights: Dict of component name to weight
                    - normalize: Whether to normalize (optional)
                    - clip_range: Tuple of (min, max) (optional)
                    - log_rewards: Whether to log (optional)
        
        Returns:
            RewardHandler instance
        """
        from janus_ai.ml.rewards.reward_registry import create_reward_component
        
        # Process components
        component_dict = {}
        if isinstance(config['components'], list):
            # Simple list of names - use default parameters
            for name in config['components']:
                component = create_reward_component(name)
                component_dict[component] = config['weights'].get(name, 1.0)
        else:
            # Dict with parameters
            for name, params in config['components'].items():
                component = create_reward_component(name, **params)
                component_dict[component] = config['weights'].get(name, 1.0)
        
        return cls(
            reward_components=component_dict,
            normalize=config.get('normalize', False),
            clip_range=config.get('clip_range'),
            log_rewards=config.get('log_rewards', True)
        )


class AdaptiveRewardHandler(RewardHandler):
    """
    Extended reward handler with adaptive weight adjustment.
    
    This handler can automatically adjust weights based on training progress
    or other metrics.
    """
    
    def __init__(
        self,
        reward_components: Dict[Union[str, BaseReward], float],
        adaptation_rate: float = 0.01,
        target_balance: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize adaptive reward handler.
        
        Args:
            reward_components: Initial reward components and weights
            adaptation_rate: Rate of weight adaptation (0.0 = no adaptation)
            target_balance: Target contribution ratios for each component
            **kwargs: Additional arguments passed to RewardHandler
        """
        super().__init__(reward_components, **kwargs)
        self.adaptation_rate = adaptation_rate
        self.target_balance = target_balance or {}
        
        # Track statistics for adaptation
        self.reward_history = {name: [] for name in self.components}
        self.contribution_history = {name: [] for name in self.components}
    
    def calculate_detailed_reward(
        self,
        current_observation: Any,
        action: Any,
        next_observation: Any,
        reward_from_env: float,
        done: bool,
        info: Dict[str, Any]
    ) -> RewardInfo:
        """Calculate reward and update statistics for adaptation."""
        # Get base calculation
        reward_info = super().calculate_detailed_reward(
            current_observation, action, next_observation,
            reward_from_env, done, info
        )
        
        # Track statistics
        total = abs(reward_info.total_reward) + 1e-8
        for name, reward in reward_info.component_rewards.items():
            self.reward_history[name].append(reward)
            contribution = abs(reward_info.weighted_rewards[name]) / total
            self.contribution_history[name].append(contribution)
        
        # Adapt weights if needed
        if done and self.adaptation_rate > 0:
            self._adapt_weights()
        
        return reward_info
    
    def _adapt_weights(self):
        """Adapt weights based on contribution history and targets."""
        # Calculate average contributions
        avg_contributions = {}
        for name in self.components:
            if len(self.contribution_history[name]) > 0:
                avg_contributions[name] = np.mean(self.contribution_history[name][-100:])
            else:
                avg_contributions[name] = 0.0
        
        # Adapt weights towards target balance
        new_weights = dict(self.weights)
        for name in self.components:
            if name in self.target_balance:
                target = self.target_balance[name]
                current = avg_contributions[name]
                
                # Adjust weight to move contribution towards target
                adjustment = self.adaptation_rate * (target - current)
                new_weights[name] = max(0.0, self.weights[name] + adjustment)
        
        # Normalize weights to sum to original sum
        weight_sum = sum(new_weights.values())
        if weight_sum > 0:
            original_sum = sum(self.weights.values())
            for name in new_weights:
                new_weights[name] = new_weights[name] * original_sum / weight_sum
        
        # Update weights
        self.update_weights(new_weights)
        
        # Clear old history to prevent memory growth
        for name in self.components:
            if len(self.reward_history[name]) > 1000:
                self.reward_history[name] = self.reward_history[name][-500:]
                self.contribution_history[name] = self.contribution_history[name][-500:]