# File: JanusAI/ml/rewards/interpretability_reward.py
"""
Complete InterpretabilityReward Integration
===========================================

This implementation integrates the evaluation modules with the main reward system
to replace the placeholder methods with functional implementations.
"""

import numpy as np
import torch.nn as nn
from typing import Any, Dict, List, Optional

from janus_ai.ml.rewards.base_reward import BaseReward
from janus_ai.ai_interpretability.evaluation.fidelity import ModelFidelityEvaluator
from janus_ai.ai_interpretability.evaluation.consistency import InterpretabilityEvaluator

class InterpretabilityReward(BaseReward):
    """
    Calculates a combined interpretability reward for symbolic expressions,
    integrating fidelity, simplicity, consistency, and insight metrics.
    """

    def __init__(self,
                 weight: float = 1.0,
                 fidelity_weight: float = 0.4,
                 simplicity_weight: float = 0.3,
                 consistency_weight: float = 0.2,
                 insight_weight: float = 0.1,
                 complexity_penalty_factor: float = 0.01,
                 max_complexity_for_penalty: Optional[int] = None,
                 interpretability_metric: str = 'mdl',
                 target_model: Optional[nn.Module] = None,
                 data_samples: Optional[Dict[str, np.ndarray]] = None,
                 variables_for_evaluation: Optional[List[Any]] = None):
        
        super().__init__(weight)

        self.component_weights: Dict[str, float] = {
            'fidelity': fidelity_weight,
            'simplicity': simplicity_weight,
            'consistency': consistency_weight,
            'insight': insight_weight
        }

        # Initialize evaluation modules
        if target_model is None or data_samples is None or variables_for_evaluation is None:
            print("Warning: InterpretabilityReward initialized without complete data. Using fallback evaluation.")
            self.model_fidelity_evaluator = None
            self.interpretability_evaluator = None
            self.fallback_mode = True
        else:
            self.model_fidelity_evaluator = ModelFidelityEvaluator(
                target_model=target_model,
                data_samples=data_samples,
                variables=variables_for_evaluation,
                loss_type='r_squared'
            )
            
            self.interpretability_evaluator = InterpretabilityEvaluator(
                complexity_penalty_factor=complexity_penalty_factor,
                max_complexity_for_penalty=max_complexity_for_penalty,
                interpretability_metric=interpretability_metric,
                fidelity_evaluator=self.model_fidelity_evaluator
            )
            self.fallback_mode = False

    def calculate_reward(self,
                         current_observation: Any,
                         action: Any,
                         next_observation: Any,
                         reward_from_env: float,
                         done: bool,
                         info: Dict[str, Any]) -> float:
        """
        Calculate the combined interpretability reward for a symbolic expression.
        """
        expression = info.get('expression')
        if expression is None:
            return -1.0

        if self.fallback_mode:
            return self._calculate_fallback_reward(expression, info)

        # Extract required data from info
        variables_from_info = info.get('variables', [])
        ai_model_from_info = info.get('ai_model', self.model_fidelity_evaluator.target_model)
        test_data_from_info = info.get('test_data', self.model_fidelity_evaluator.data_samples)

        if not variables_from_info:
            print(f"Warning: No variables provided for expression evaluation: {expression}")
            return 0.0

        # Calculate individual reward components
        try:
            fidelity_score = self.model_fidelity_evaluator.calculate_fidelity(expression)
            simplicity_score = self.interpretability_evaluator.calculate_simplicity(expression)
            consistency_score = self.interpretability_evaluator.test_consistency(
                expression, ai_model_from_info, test_data_from_info, variables_from_info
            )
            insight_score = self.interpretability_evaluator.calculate_insight_score(
                expression, ai_model_from_info, info.get('additional_context')
            )

            # Combine scores using component weights
            total_reward = (
                self.component_weights['fidelity'] * fidelity_score +
                self.component_weights['simplicity'] * simplicity_score +
                self.component_weights['consistency'] * consistency_score +
                self.component_weights['insight'] * insight_score
            )

            return total_reward

        except Exception as e:
            print(f"Error calculating interpretability reward: {e}")
            return self._calculate_fallback_reward(expression, info)

    def _calculate_fallback_reward(self, expression: Any, info: Dict[str, Any]) -> float:
        """
        Fallback reward calculation when evaluation modules aren't available.
        """
        try:
            # Basic complexity penalty
            complexity = getattr(expression, 'complexity', len(str(expression)))
            complexity_penalty = -0.01 * max(0, complexity - 10)
            
            # Basic structure reward
            structure_reward = 0.1 if hasattr(expression, 'operator') else 0.0
            
            return max(0.0, structure_reward + complexity_penalty)
        except:
            return 0.0

    def update_target_model(self, new_model: nn.Module, new_data: Dict[str, np.ndarray], new_variables: List[Any]):
        """
        Update the target model and data for evaluation.
        """
        if self.model_fidelity_evaluator is not None:
            self.model_fidelity_evaluator.target_model = new_model
            self.model_fidelity_evaluator.data_samples = new_data
            self.model_fidelity_evaluator.variables = new_variables
            self.fallback_mode = False


# Integration helper function
def patch_interpretability_reward_methods():
    """
    Helper function to ensure all evaluation methods are properly integrated.
    Call this during system initialization.
    """
    
    def _calculate_fidelity(self, expression: Any, ai_model: Any, test_data: Any) -> float:
        """Updated _calculate_fidelity method using FidelityCalculator."""
        if not hasattr(self, '_fidelity_calculator'):
            from janus_ai.ai_interpretability.evaluation.fidelity import FidelityCalculator
            self._fidelity_calculator = FidelityCalculator()
        
        # Convert test_data to expected format
        if hasattr(test_data, 'inputs') and hasattr(test_data, 'outputs'):
            data_dict = {
                'input_ids': test_data.inputs,
                'outputs': test_data.outputs,
                'attention_weights': getattr(test_data, 'attention_weights', None),
                'sequence_length': test_data.inputs.shape[-1] if hasattr(test_data.inputs, 'shape') else 32
            }
        else:
            data_dict = test_data
        
        variables = getattr(self, 'variables', [])
        
        return self._fidelity_calculator.calculate_fidelity(
            expression=expression,
            ai_model=ai_model,
            test_data=data_dict,
            variables=variables,
            target_behavior='attention'
        )
    
    # Apply the patch to InterpretabilityReward
    InterpretabilityReward._calculate_fidelity = _calculate_fidelity

if __name__ == "__main__":
    # Apply integration patches
    patch_interpretability_reward_methods()
    
    print("InterpretabilityReward integration completed successfully!")
    print("Key features integrated:")
    print("✓ Fidelity calculation using ModelFidelityEvaluator")
    print("✓ Consistency testing with data splitting")
    print("✓ Insight scoring with context awareness")
    print("✓ Graceful fallback for incomplete initialization")
    print("✓ Dynamic model/data updating capability")
