# File: JanusAI/ml/rewards/enhanced_intrinsic_rewards.py
"""
Intrinsic Reward Mechanisms for Exploration

This module implements various intrinsic reward mechanisms including:
- Information Gain (MaxInfoRL)
- Pre-trained Network Distillation (PreND)
- Goal Matching for LLM-guided exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np
from abc import ABC, abstractmethod

from janus.ml.networks.dynamics_ensemble import DynamicsEnsemble
from janus.core.expressions import Expression


class BaseIntrinsicReward(ABC):
    """Base class for intrinsic reward mechanisms."""
    
    @abstractmethod
    def calculate_reward(self, **kwargs) -> float:
        """Calculate the intrinsic reward."""
        pass
    
    @abstractmethod
    def update(self, **kwargs):
        """Update the reward mechanism (e.g., train predictors)."""
        pass


class InformationGainReward(BaseIntrinsicReward):
    """
    Information Gain reward based on epistemic uncertainty.
    
    Rewards the agent for exploring expressions where the dynamics ensemble
    is most uncertain, encouraging exploration of unknown regions.
    """
    
    def __init__(self, 
                 ensemble: DynamicsEnsemble,
                 scale_factor: float = 1.0,
                 normalize: bool = True,
                 temperature: float = 1.0):
        """
        Args:
            ensemble: The dynamics ensemble for uncertainty estimation
            scale_factor: Scaling factor for the reward
            normalize: Whether to normalize rewards using running statistics
            temperature: Temperature for uncertainty scaling
        """
        self.ensemble = ensemble
        self.scale_factor = scale_factor
        self.normalize = normalize
        self.temperature = temperature
        
        # Running statistics for normalization
        if normalize:
            self.running_mean = 0.0
            self.running_var = 1.0
            self.running_count = 0
            self.beta = 0.99  # Exponential moving average factor
    
    def calculate_reward(self, 
                        expression_embedding: torch.Tensor,
                        return_components: bool = False,
                        **kwargs) -> float:
        """
        Calculate information gain reward based on ensemble uncertainty.
        
        Args:
            expression_embedding: Embedding of the current expression
            return_components: Whether to return reward components
            
        Returns:
            Intrinsic reward value (or dict if return_components=True)
        """
        # Get information gain from ensemble
        info_gain = self.ensemble.get_information_gain(expression_embedding)
        
        # Get uncertainty components
        predictions = self.ensemble.predict(expression_embedding)
        epistemic_uncertainty = predictions['epistemic_uncertainty'].mean().item()
        aleatoric_uncertainty = predictions['aleatoric_uncertainty'].mean().item()
        
        # Temperature scaling
        scaled_info_gain = info_gain / self.temperature
        
        # Normalize if requested
        if self.normalize:
            normalized_gain = self._normalize_reward(scaled_info_gain)
        else:
            normalized_gain = scaled_info_gain
        
        # Final reward
        reward = self.scale_factor * normalized_gain
        
        if return_components:
            return {
                'total_reward': reward,
                'raw_info_gain': info_gain,
                'epistemic_uncertainty': epistemic_uncertainty,
                'aleatoric_uncertainty': aleatoric_uncertainty,
                'normalized_gain': normalized_gain
            }
        
        return reward
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        # Update running statistics
        self.running_count += 1
        
        if self.running_count == 1:
            self.running_mean = reward
            self.running_var = 0.0
        else:
            # Exponential moving average
            delta = reward - self.running_mean
            self.running_mean += (1 - self.beta) * delta
            self.running_var = self.beta * self.running_var + (1 - self.beta) * delta**2
        
        # Normalize
        std = np.sqrt(self.running_var + 1e-8)
        normalized = (reward - self.running_mean) / std
        
        # Clip to reasonable range
        return np.clip(normalized, -3.0, 3.0)
    
    def update(self, 
               expression_embeddings: torch.Tensor,
               true_rewards: torch.Tensor,
               **kwargs):
        """
        Update the dynamics ensemble with new experience.
        
        Args:
            expression_embeddings: Batch of expression embeddings
            true_rewards: Corresponding true rewards
        """
        metrics = self.ensemble.train_step(expression_embeddings, true_rewards)
        return metrics
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics of the reward mechanism."""
        stats = {
            'running_mean': self.running_mean,
            'running_std': np.sqrt(self.running_var),
            'scale_factor': self.scale_factor,
            'temperature': self.temperature
        }
        
        # Add ensemble statistics
        if hasattr(self.ensemble, 'training_stats'):
            recent_disagreements = self.ensemble.training_stats['disagreements'][-100:]
            if recent_disagreements:
                stats['mean_disagreement'] = np.mean(recent_disagreements)
                stats['disagreement_trend'] = (
                    np.mean(recent_disagreements[-10:]) - 
                    np.mean(recent_disagreements[:10])
                ) if len(recent_disagreements) >= 20 else 0.0
        
        return stats


class PreNDIntrinsicReward(BaseIntrinsicReward):
    """
    Pre-trained Network Distillation reward.
    
    Uses a powerful pre-trained model as a fixed target and rewards
    the agent for finding states that are hard to predict.
    """
    
    def __init__(self,
                 target_net: nn.Module,
                 predictor_hidden_dim: int = 256,
                 learning_rate: float = 1e-4,
                 reward_scale: float = 1.0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            target_net: Pre-trained network (frozen)
            predictor_hidden_dim: Hidden dimension for predictor network
            learning_rate: Learning rate for predictor
            reward_scale: Scaling factor for rewards
            device: Device for computation
        """
        self.device = device
        self.reward_scale = reward_scale
        
        # Freeze target network
        self.target_net = target_net.to(device)
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        # Determine target feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 512).to(device)  # Adjust input size as needed
            target_features = self._get_target_features(dummy_input)
            self.target_feature_dim = target_features.shape[-1]
        
        # Create predictor network
        self.predictor = self._build_predictor(predictor_hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Statistics tracking
        self.prediction_errors = []
        self.reward_history = []
    
    def _build_predictor(self, hidden_dim: int) -> nn.Module:
        """Build the predictor network."""
        return nn.Sequential(
            nn.Linear(512, hidden_dim),  # Adjust input dim as needed
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.target_feature_dim)
        )
    
    def _get_target_features(self, state_representation: torch.Tensor) -> torch.Tensor:
        """Extract features from the target network."""
        with torch.no_grad():
            # This assumes the target network outputs features directly
            # Modify based on your specific pre-trained model
            if hasattr(self.target_net, 'encode'):
                features = self.target_net.encode(state_representation)
            elif hasattr(self.target_net, 'forward_features'):
                features = self.target_net.forward_features(state_representation)
            else:
                features = self.target_net(state_representation)
            
            # Handle different output formats
            if isinstance(features, dict):
                # For models that return dictionaries (e.g., transformers)
                features = features.get('pooler_output', features.get('last_hidden_state', features))
            
            # Pool if necessary
            if features.dim() > 2:
                features = features.mean(dim=1)  # Simple average pooling
            
            return features
    
    def calculate_reward(self, 
                        state_representation: torch.Tensor,
                        return_components: bool = False,
                        **kwargs) -> float:
        """
        Calculate PreND reward as prediction error.
        
        The reward is high when the predictor fails to match the target,
        indicating a novel/interesting state.
        """
        with torch.no_grad():
            # Get target features
            target_features = self._get_target_features(state_representation)
            
            # Get predictor features
            self.predictor.eval()
            predictor_features = self.predictor(state_representation)
            
            # Calculate prediction error
            prediction_error = F.mse_loss(predictor_features, target_features).item()
            
            # Convert error to reward (higher error = higher reward)
            # Use a bounded transformation to avoid extreme values
            raw_reward = 1.0 - np.exp(-prediction_error)
            scaled_reward = self.reward_scale * raw_reward
            
            # Track statistics
            self.prediction_errors.append(prediction_error)
            self.reward_history.append(scaled_reward)
        
        if return_components:
            return {
                'total_reward': scaled_reward,
                'prediction_error': prediction_error,
                'raw_reward': raw_reward,
                'target_norm': target_features.norm().item(),
                'predictor_norm': predictor_features.norm().item()
            }
        
        return scaled_reward
    
    def update(self, state_representations: torch.Tensor, **kwargs):
        """
        Train the predictor to match target features.
        
        Args:
            state_representations: Batch of state representations
        """
        self.predictor.train()
        
        # Get target features
        with torch.no_grad():
            target_features = self._get_target_features(state_representations)
        
        # Forward pass
        predictor_features = self.predictor(state_representations)
        
        # Calculate loss
        loss = self.loss_fn(predictor_features, target_features)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {'predictor_loss': loss.item()}
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics."""
        if not self.prediction_errors:
            return {}
        
        recent_errors = self.prediction_errors[-100:]
        recent_rewards = self.reward_history[-100:]
        
        return {
            'mean_prediction_error': np.mean(recent_errors),
            'std_prediction_error': np.std(recent_errors),
            'mean_reward': np.mean(recent_rewards),
            'error_trend': (
                np.mean(recent_errors[-10:]) - np.mean(recent_errors[:10])
            ) if len(recent_errors) >= 20 else 0.0
        }


class GoalMatchingReward(BaseIntrinsicReward):
    """
    Reward for matching LLM-suggested goal expressions.
    
    Encourages the agent to explore specific regions of the expression space
    suggested by a language model.
    """
    
    def __init__(self,
                 grammar,  # ProgressiveGrammar instance
                 max_reward: float = 10.0,
                 distance_temperature: float = 1.0):
        """
        Args:
            grammar: Grammar for parsing expressions
            max_reward: Maximum reward for exact match
            distance_temperature: Temperature for distance-based rewards
        """
        self.grammar = grammar
        self.max_reward = max_reward
        self.distance_temperature = distance_temperature
        
        self.current_goal_expression = None
        self.goal_features = None
        self.goal_achievement_history = []
    
    def set_goal(self, goal_expression_str: str):
        """Set a new goal expression from LLM suggestion."""
        try:
            self.current_goal_expression = self.grammar.parse(goal_expression_str)
            self.goal_features = self._extract_expression_features(self.current_goal_expression)
            return True
        except Exception as e:
            print(f"Failed to parse goal expression: {e}")
            self.current_goal_expression = None
            self.goal_features = None
            return False
    
    def _extract_expression_features(self, expression: Expression) -> Dict[str, float]:
        """Extract features from an expression for comparison."""
        features = {
            'depth': expression.depth(),
            'node_count': len(expression.nodes()) if hasattr(expression, 'nodes') else 1,
            'operator_count': 0,
            'variable_count': 0,
            'constant_count': 0
        }
        
        # Count different node types
        if hasattr(expression, 'traverse'):
            for node in expression.traverse():
                if hasattr(node, 'type'):
                    if node.type == 'operator':
                        features['operator_count'] += 1
                    elif node.type == 'variable':
                        features['variable_count'] += 1
                    elif node.type == 'constant':
                        features['constant_count'] += 1
        
        return features
    
    def calculate_reward(self,
                        generated_expression: Expression,
                        return_components: bool = False,
                        **kwargs) -> float:
        """
        Calculate reward based on similarity to goal expression.
        """
        if self.current_goal_expression is None:
            return 0.0
        
        # Extract features from generated expression
        generated_features = self._extract_expression_features(generated_expression)
        
        # Calculate structural distance
        structural_distance = self._calculate_structural_distance(
            generated_features, 
            self.goal_features
        )
        
        # Calculate tree edit distance if available
        tree_distance = 0.0
        if hasattr(self.grammar, 'tree_distance'):
            tree_distance = self.grammar.tree_distance(
                generated_expression,
                self.current_goal_expression
            )
        
        # Combined distance
        total_distance = 0.7 * structural_distance + 0.3 * tree_distance
        
        # Convert distance to reward
        raw_reward = np.exp(-total_distance / self.distance_temperature)
        scaled_reward = self.max_reward * raw_reward
        
        # Track achievement
        self.goal_achievement_history.append({
            'distance': total_distance,
            'reward': scaled_reward,
            'exact_match': total_distance < 0.01
        })
        
        if return_components:
            return {
                'total_reward': scaled_reward,
                'structural_distance': structural_distance,
                'tree_distance': tree_distance,
                'total_distance': total_distance,
                'exact_match': total_distance < 0.01
            }
        
        return scaled_reward
    
    def _calculate_structural_distance(self, 
                                     features1: Dict[str, float],
                                     features2: Dict[str, float]) -> float:
        """Calculate normalized distance between feature dictionaries."""
        distance = 0.0
        
        for key in features1:
            if key in features2:
                # Normalize by max value to handle different scales
                max_val = max(abs(features1[key]), abs(features2[key]), 1.0)
                distance += abs(features1[key] - features2[key]) / max_val
        
        # Normalize by number of features
        return distance / len(features1)
    
    def update(self, **kwargs):
        """No updates needed for goal matching."""
        pass
    
    def get_statistics(self) -> Dict[str, float]:
        """Get goal achievement statistics."""
        if not self.goal_achievement_history:
            return {}
        
        recent = self.goal_achievement_history[-100:]
        
        return {
            'mean_distance': np.mean([h['distance'] for h in recent]),
            'mean_reward': np.mean([h['reward'] for h in recent]),
            'exact_matches': sum(h['exact_match'] for h in recent),
            'achievement_rate': sum(h['exact_match'] for h in recent) / len(recent)
        }


class CombinedIntrinsicReward(BaseIntrinsicReward):
    """
    Combines multiple intrinsic reward mechanisms with adaptive weighting.
    """
    
    def __init__(self, reward_components: Dict[str, BaseIntrinsicReward],
                 initial_weights: Optional[Dict[str, float]] = None,
                 adaptive_weights: bool = True):
        """
        Args:
            reward_components: Dictionary of reward mechanisms
            initial_weights: Initial weights for each component
            adaptive_weights: Whether to adapt weights based on performance
        """
        self.components = reward_components
        
        # Initialize weights
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            # Equal weights by default
            self.weights = {name: 1.0 / len(reward_components) 
                          for name in reward_components}
        
        self.adaptive_weights = adaptive_weights
        
        # Track component contributions
        self.contribution_history = {name: [] for name in reward_components}
        
    def calculate_reward(self, return_components: bool = False, **kwargs) -> float:
        """Calculate combined reward from all components."""
        component_rewards = {}
        
        for name, component in self.components.items():
            try:
                # Pass relevant kwargs to each component
                component_kwargs = self._filter_kwargs_for_component(name, kwargs)
                reward = component.calculate_reward(**component_kwargs)
                component_rewards[name] = reward
                
                # Track contribution
                self.contribution_history[name].append(reward)
                
            except Exception as e:
                print(f"Error in {name} reward calculation: {e}")
                component_rewards[name] = 0.0
        
        # Weighted sum
        total_reward = sum(
            self.weights[name] * component_rewards[name]
            for name in component_rewards
        )
        
        # Adapt weights if enabled
        if self.adaptive_weights and len(self.contribution_history[list(self.components.keys())[0]]) > 100:
            self._adapt_weights()
        
        if return_components:
            return {
                'total_reward': total_reward,
                'components': component_rewards,
                'weights': self.weights.copy()
            }
        
        return total_reward
    
    def _filter_kwargs_for_component(self, component_name: str, kwargs: Dict) -> Dict:
        """Filter kwargs relevant for each component."""
        # Map component names to their expected kwargs
        component_kwargs_map = {
            'information_gain': ['expression_embedding'],
            'prend': ['state_representation'],
            'goal_matching': ['generated_expression']
        }
        
        expected_kwargs = component_kwargs_map.get(component_name, [])
        return {k: v for k, v in kwargs.items() if k in expected_kwargs}
    
    def _adapt_weights(self):
        """Adapt weights based on component performance."""
        # Calculate recent statistics for each component
        recent_stats = {}
        
        for name in self.components:
            recent_contributions = self.contribution_history[name][-100:]
            
            # Calculate utility metrics
            mean_contribution = np.mean(recent_contributions)
            std_contribution = np.std(recent_contributions)
            
            # Utility combines mean and variance (exploration bonus)
            utility = mean_contribution + 0.1 * std_contribution
            recent_stats[name] = utility
        
        # Normalize utilities to get new weights
        total_utility = sum(recent_stats.values())
        
        if total_utility > 0:
            for name in self.components:
                # Smooth weight update
                new_weight = recent_stats[name] / total_utility
                self.weights[name] = 0.9 * self.weights[name] + 0.1 * new_weight
        
        # Ensure weights sum to 1
        weight_sum = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= weight_sum
    
    def update(self, **kwargs):
        """Update all components."""
        results = {}
        
        for name, component in self.components.items():
            try:
                component_kwargs = self._filter_kwargs_for_component(name, kwargs)
                result = component.update(**component_kwargs)
                if result:
                    results[name] = result
            except Exception as e:
                print(f"Error updating {name}: {e}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        stats = {
            'weights': self.weights.copy(),
            'components': {}
        }
        
        for name, component in self.components.items():
            stats['components'][name] = component.get_statistics()
        
        # Add contribution statistics
        for name in self.components:
            if self.contribution_history[name]:
                recent = self.contribution_history[name][-100:]
                stats['components'][name]['mean_contribution'] = np.mean(recent)
                stats['components'][name]['contribution_trend'] = (
                    np.mean(recent[-10:]) - np.mean(recent[:10])
                ) if len(recent) >= 20 else 0.0
        
        return stats