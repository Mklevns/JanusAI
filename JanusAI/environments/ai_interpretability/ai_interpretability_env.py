"""
Refactored AI Interpretability Environment
=========================================

Updated implementation with fixes for import paths, error handling,
and integration with the refactored codebase.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import sympy as sp
import logging

from janus_ai.core.expressions.expression import Expression, Variable
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus_ai.ai_interpretability.grammars.neural_grammar import NeuralGrammar
# Fixed import path for evaluate_expression_on_data
from janus_ai.utils.math.operations import evaluate_expression_on_data


@dataclass
class AIBehaviorData:
    """Represents input-output data from an AI model."""
    inputs: np.ndarray  # Shape: (n_samples, input_dim)
    outputs: np.ndarray  # Shape: (n_samples, output_dim)
    intermediate_activations: Optional[Dict[str, np.ndarray]] = None
    attention_weights: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate data consistency."""
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError(
                f"Sample count mismatch: inputs={self.inputs.shape[0]}, "
                f"outputs={self.outputs.shape[0]}"
            )


class AIInterpretabilityEnv(SymbolicDiscoveryEnv):
    """
    Environment for discovering interpretable explanations of AI behavior.

    This refactored version cleanly separates inputs and outputs and provides
    robust error handling and integration with the Janus reward system.
    """

    def __init__(
        self,
        ai_model: nn.Module,
        grammar: NeuralGrammar,
        behavior_data: AIBehaviorData,
        interpretation_mode: str = 'global',
        output_index: Optional[int] = None,
        include_activations: bool = False,
        max_depth: int = 10,
        max_complexity: int = 30,
        reward_config: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize AI interpretability environment.

        Args:
            ai_model: The AI model to interpret
            grammar: Neural grammar for expression generation
            behavior_data: Input-output data from the model
            interpretation_mode: 'global', 'local', or 'modular'
            output_index: Which output to explain (for multi-output models)
            include_activations: Whether to include intermediate activations as variables
            max_depth: Maximum expression tree depth
            max_complexity: Maximum expression complexity
            reward_config: Custom reward configuration
            **kwargs: Additional arguments passed to parent class
        """
        self.ai_model = ai_model
        self.interpretation_mode = interpretation_mode
        self.behavior_data = behavior_data
        self.output_index = output_index
        self.include_activations = include_activations
        self.logger = logging.getLogger(__name__)

        if reward_config is None:
            reward_config = {}

        # Validate interpretation mode
        valid_modes = ['global', 'local', 'modular']
        if interpretation_mode not in valid_modes:
            raise ValueError(f"interpretation_mode must be one of {valid_modes}")

        # Prepare X_data and y_data
        try:
            X_data, y_data = self._prepare_data()
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            raise

        # Extract variables that correspond to X_data columns
        variables = self._extract_variables_from_model()

        # Set up reward configuration with AI-specific defaults
        if not reward_config: # Check if reward_config is an empty dict
            reward_config = {
                'mse_weight': -1.0,
                'complexity_penalty': -0.02,
                'fidelity_bonus': 0.5,
                'interpretability_bonus': 0.3
            }

        # Initialize parent with separated data
        super().__init__(
            grammar=grammar,
            X_data=X_data,
            y_data=y_data,
            variables=variables,
            max_depth=max_depth,
            max_complexity=max_complexity,
            reward_config=reward_config,
            task_type='ai_interpretability',  # Set specific task type
            **kwargs
        )

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input features (X) and target outputs (y).

        Returns:
            Tuple of (X_data, y_data)
        """
        # Start with model inputs as features
        X_data = self.behavior_data.inputs.copy()

        # Optionally add intermediate activations as additional features
        if self.include_activations and self.behavior_data.intermediate_activations:
            activation_arrays = []

            for layer_name, activations in sorted(self.behavior_data.intermediate_activations.items()):
                # Validate activation data
                if activations.shape[0] != X_data.shape[0]:
                    self.logger.warning(
                        f"Activation {layer_name} sample count mismatch. "
                        f"Expected {X_data.shape[0]}, got {activations.shape[0]}. Skipping."
                    )
                    continue

                # Flatten activations if needed
                if activations.ndim > 2:
                    activations = activations.reshape(activations.shape[0], -1)
                activation_arrays.append(activations)

            if activation_arrays:
                X_data = np.hstack([X_data] + activation_arrays)
                self.logger.info(f"Added {len(activation_arrays)} activation layers to features")

        # Handle target outputs
        y_data = self.behavior_data.outputs.copy()

        # Select specific output if specified
        if self.output_index is not None:
            if y_data.ndim == 1:
                raise ValueError("output_index specified but y_data is 1D")
            if self.output_index >= y_data.shape[1]:
                raise ValueError(
                    f"output_index {self.output_index} out of bounds for "
                    f"y_data with {y_data.shape[1]} outputs"
                )
            y_data = y_data[:, self.output_index:self.output_index+1]

        # Ensure y_data is 2D for consistency
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)

        self.logger.info(
            f"Prepared data: X_data={X_data.shape}, y_data={y_data.shape}, "
            f"interpretation_mode={self.interpretation_mode}"
        )

        return X_data, y_data

    def _extract_variables_from_model(self) -> List[Variable]:
        """
        Extract variables based on model structure and data.

        Returns:
            List of Variable objects corresponding to X_data columns
        """
        variables = []

        # Add input variables
        n_inputs = self.behavior_data.inputs.shape[1]
        for i in range(n_inputs):
            var = Variable(
                name=f'input_{i}',
                index=i,
                properties={
                    'type': 'input',
                    'original_index': i,
                    'description': f'Model input feature {i}',
                    'interpretation_mode': self.interpretation_mode
                }
            )
            variables.append(var)

        # Add activation variables if included
        if self.include_activations and self.behavior_data.intermediate_activations:
            current_idx = n_inputs

            for layer_name, activations in sorted(self.behavior_data.intermediate_activations.items()):
                # Skip if activation data is invalid
                if activations.shape[0] != self.behavior_data.inputs.shape[0]:
                    continue

                # Determine number of features after flattening
                if activations.ndim > 2:
                    n_features = np.prod(activations.shape[1:])
                else:
                    n_features = activations.shape[1]

                for j in range(n_features):
                    var = Variable(
                        name=f'{layer_name}_act_{j}',
                        index=current_idx,
                        properties={
                            'type': 'activation',
                            'layer': layer_name,
                            'neuron_index': j,
                            'description': f'Activation from {layer_name}, neuron {j}',
                            'interpretation_mode': self.interpretation_mode
                        }
                    )
                    variables.append(var)
                    current_idx += 1

        self.logger.info(f"Extracted {len(variables)} variables from model")
        return variables

    def _calculate_reward(self, action_valid: bool) -> float:
        """
        Calculate reward with AI-specific considerations.

        Extends parent reward with fidelity and interpretability bonuses.
        """
        # Get base reward from parent
        base_reward = super()._calculate_reward(action_valid)

        # If expression is complete, add AI-specific rewards
        if self.current_state and self.current_state.is_complete():
            expression = self.current_state.to_expression()
            if expression:
                try:
                    # Add fidelity bonus (how well does it match the model?)
                    fidelity_bonus = self._calculate_fidelity_bonus(expression)

                    # Add interpretability bonus (how understandable is it?)
                    interp_bonus = self._calculate_interpretability_bonus(expression)

                    # Weight the bonuses according to reward config
                    total_bonus = (
                        self.reward_config.get('fidelity_bonus', 0.5) * fidelity_bonus +
                        self.reward_config.get('interpretability_bonus', 0.3) * interp_bonus
                    )

                    return base_reward + total_bonus

                except Exception as e:
                    self.logger.warning(f"Error calculating AI-specific rewards: {e}")

        return base_reward

    def _calculate_fidelity_bonus(self, expression: Expression) -> float:
        """Calculate how well the expression matches the AI model's behavior."""
        try:
            # Evaluate expression on our data
            predictions = self._evaluate_expression_safely(expression)
            if predictions is None:
                return 0.0

            # Get actual model outputs
            y_true = self.y_data[:, 0] if self.y_data.ndim > 1 else self.y_data

            # Handle shape mismatches
            min_len = min(len(predictions), len(y_true))
            predictions = predictions[:min_len]
            y_true = y_true[:min_len]

            # Calculate correlation (robust to different scales)
            try:
                correlation = np.corrcoef(predictions.flatten(), y_true.flatten())[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0

            # Convert correlation to bonus (0.0 to 1.0)
            abs_correlation = abs(correlation)

            # Nonlinear bonus scaling for high fidelity
            if abs_correlation > 0.95:
                return 1.0
            elif abs_correlation > 0.9:
                return 0.8
            elif abs_correlation > 0.8:
                return 0.5
            elif abs_correlation > 0.6:
                return 0.2
            else:
                return 0.0

        except Exception as e:
            self.logger.debug(f"Fidelity calculation failed: {e}")
            return 0.0

    def _calculate_interpretability_bonus(self, expression: Expression) -> float:
        """Bonus for expressions that are human-interpretable."""
        try:
            expr_str = str(expression)
            bonus = 0.0

            # Reward simple, recognizable patterns
            # Linear combinations are interpretable
            if '+' in expr_str and any(op in expr_str for op in ['*', '/']):
                bonus += 0.2

            # Reward use of meaningful input variables (not just activations)
            input_vars_used = sum(1 for var in self.variables
                                if var.name in expr_str and var.properties.get('type') == 'input')
            bonus += 0.1 * min(input_vars_used, 3)

            # Penalize overly complex expressions
            complexity_penalty = max(0, len(expr_str) - 50) * 0.002
            bonus -= complexity_penalty

            # Bonus for common interpretable functions
            interpretable_funcs = ['sin', 'cos', 'exp', 'log', 'sqrt']
            func_bonus = sum(0.05 for func in interpretable_funcs if func in expr_str)
            bonus += min(func_bonus, 0.15)  # Cap function bonus

            # Mode-specific bonuses
            if self.interpretation_mode == 'local':
                # Local interpretability values simplicity
                if len(expr_str) < 30:
                    bonus += 0.1
            elif self.interpretation_mode == 'global':
                # Global interpretability values generalizability
                if input_vars_used >= 2:
                    bonus += 0.1

            return max(0.0, min(1.0, bonus))  # Clamp to [0, 1]

        except Exception as e:
            self.logger.debug(f"Interpretability calculation failed: {e}")
            return 0.0

    def _evaluate_expression_safely(self, expression: Expression) -> Optional[np.ndarray]:
        """Safely evaluate expression with error handling."""
        try:
            # Create data dictionary for evaluation
            data_dict = {}
            for i, var in enumerate(self.variables):
                if i < self.X_data.shape[1]:
                    data_dict[var.name] = self.X_data[:, i]

            # Use the utility function with proper error handling
            result = evaluate_expression_on_data(str(expression), data_dict)

            # Validate result
            if result is None or len(result) == 0:
                return None

            # Handle any infinities or NaNs
            if not np.all(np.isfinite(result)):
                self.logger.debug("Expression evaluation produced non-finite values")
                return None

            return result

        except Exception as e:
            self.logger.debug(f"Expression evaluation failed: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AI model being interpreted."""
        info = {
            'model_type': type(self.ai_model).__name__,
            'interpretation_mode': self.interpretation_mode,
            'n_input_features': self.behavior_data.inputs.shape[1],
            'n_output_features': self.behavior_data.outputs.shape[1],
            'n_samples': self.behavior_data.inputs.shape[0],
            'includes_activations': self.include_activations,
            'output_index': self.output_index
        }

        if self.include_activations and self.behavior_data.intermediate_activations:
            info['activation_layers'] = list(self.behavior_data.intermediate_activations.keys())

        return info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with AI-specific information."""
        obs, info = super().reset(**kwargs)

        # Add AI-specific info
        info.update(self.get_model_info())

        return obs, info


# Helper function for creating AI interpretability environments
def create_attention_interpretability_env(
    model: nn.Module,
    input_data: np.ndarray,
    attention_outputs: np.ndarray,
    grammar: Optional[NeuralGrammar] = None,
    layer_name: str = "attention",
    **kwargs
) -> AIInterpretabilityEnv:
    """
    Convenience function for creating environments to interpret attention mechanisms.

    Args:
        model: The transformer model
        input_data: Input sequences or embeddings
        attention_outputs: Attention weights or patterns to explain
        grammar: Neural grammar (creates default if None)
        layer_name: Name of the attention layer
        **kwargs: Additional arguments for environment

    Returns:
        Configured AIInterpretabilityEnv for attention interpretation
    """
    if grammar is None:
        grammar = NeuralGrammar()  # Use default neural grammar

    behavior_data = AIBehaviorData(
        inputs=input_data,
        outputs=attention_outputs,
        metadata={'layer_name': layer_name, 'task': 'attention_interpretation'}
    )

    return AIInterpretabilityEnv(
        ai_model=model,
        grammar=grammar,
        behavior_data=behavior_data,
        interpretation_mode='modular',  # Attention is modular by nature
        **kwargs
    )
