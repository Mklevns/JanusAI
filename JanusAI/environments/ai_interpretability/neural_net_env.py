"""
Refactored AI Interpretability Environment
=========================================

Updated to use separated X_data and y_data for cleaner implementation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import sympy as sp

from janus_ai.core.expressions.expression import Expression, Variable
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus_ai.ai_interpretability.grammars.neural_grammar import NeuralGrammar


@dataclass
class AIBehaviorData:
    """Represents input-output data from an AI model."""
    inputs: np.ndarray  # Shape: (n_samples, input_dim)
    outputs: np.ndarray  # Shape: (n_samples, output_dim)
    intermediate_activations: Optional[Dict[str, np.ndarray]] = None
    attention_weights: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class AIInterpretabilityEnv(SymbolicDiscoveryEnv):
    """
    Environment for discovering interpretable explanations of AI behavior.
    
    This refactored version cleanly separates inputs and outputs.
    """
    
    def __init__(
        self,
        ai_model: nn.Module,
        grammar: NeuralGrammar,
        behavior_data: AIBehaviorData,
        interpretation_mode: str = 'global',
        output_index: Optional[int] = None,
        include_activations: bool = False,
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
            **kwargs: Additional arguments passed to parent class
        """
        self.ai_model = ai_model
        self.interpretation_mode = interpretation_mode
        self.behavior_data = behavior_data
        self.output_index = output_index
        self.include_activations = include_activations
        
        # Prepare X_data and y_data
        X_data, y_data = self._prepare_data()
        
        # Extract variables that correspond to X_data columns
        variables = self._extract_variables_from_model()
        
        # Initialize parent with separated data
        super().__init__(
            grammar=grammar,
            X_data=X_data,
            y_data=y_data,
            variables=variables,
            **kwargs
        )
        
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input features (X) and target outputs (y).
        
        Returns:
            Tuple of (X_data, y_data)
        """
        # Start with model inputs as features
        X_data = self.behavior_data.inputs
        
        # Optionally add intermediate activations as additional features
        if self.include_activations and self.behavior_data.intermediate_activations:
            activation_arrays = []
            for layer_name, activations in sorted(self.behavior_data.intermediate_activations.items()):
                # Flatten activations if needed
                if activations.ndim > 2:
                    activations = activations.reshape(activations.shape[0], -1)
                activation_arrays.append(activations)
            
            if activation_arrays:
                X_data = np.hstack([X_data] + activation_arrays)
        
        # Handle target outputs
        y_data = self.behavior_data.outputs
        
        # Select specific output if specified
        if self.output_index is not None:
            if y_data.ndim == 1:
                raise ValueError("output_index specified but y_data is 1D")
            y_data = y_data[:, self.output_index:self.output_index+1]
        
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
                    'description': f'Model input feature {i}'
                }
            )
            variables.append(var)
        
        # Add activation variables if included
        if self.include_activations and self.behavior_data.intermediate_activations:
            current_idx = n_inputs
            
            for layer_name, activations in sorted(self.behavior_data.intermediate_activations.items()):
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
                            'description': f'Activation from {layer_name}, neuron {j}'
                        }
                    )
                    variables.append(var)
                    current_idx += 1
        
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
                # Add fidelity bonus (how well does it match the model?)
                fidelity_bonus = self._calculate_fidelity_bonus(expression)
                
                # Add interpretability bonus (how understandable is it?)
                interp_bonus = self._calculate_interpretability_bonus(expression)
                
                return base_reward + fidelity_bonus + interp_bonus
        
        return base_reward
    
    def _calculate_fidelity_bonus(self, expression: Expression) -> float:
        """Calculate how well the expression matches the AI model's behavior."""
        # This is a simplified version - could be enhanced
        try:
            # Evaluate expression
            predictions = self._evaluate_expression_on_data(expression)
            
            # Get actual model outputs
            y_true = self.y_data[:, 0] if self.y_data.ndim > 1 else self.y_data
            
            # Calculate agreement
            correlation = np.corrcoef(predictions.flatten(), y_true.flatten())[0, 1]
            
            # High correlation gets bonus
            if correlation > 0.9:
                return 0.5
            elif correlation > 0.8:
                return 0.2
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_interpretability_bonus(self, expression: Expression) -> float:
        """Bonus for expressions that are human-interpretable."""
        # Simple heuristic based on expression properties
        expr_str = str(expression)
        
        # Reward simple, recognizable patterns
        bonus = 0.0
        
        # Linear combinations are interpretable
        if expr_str.count('+') > 0 and expr_str.count('*') > 0:
            bonus += 0.1
        
        # Penalize very complex expressions
        if len(expr_str) > 100:
            bonus -= 0.2
        
        # Reward use of meaningful variables
        meaningful_vars = sum(1 for var in self.variables 
                            if var.name in expr_str and var.properties.get('type') == 'input')
        bonus += 0.05 * min(meaningful_vars, 3)
        
        return bonus
    
    def _evaluate_expression_on_data(self, expression: Expression) -> np.ndarray:
        """Evaluate expression on the environment's data."""
        from janus_ai.core.expressions.symbolic_math import evaluate_expression_on_data
        
        return evaluate_expression_on_data(
            str(expression),
            [var.name for var in self.variables],
            self.X_data
        )


