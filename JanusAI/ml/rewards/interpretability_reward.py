"""
Interpretability Reward (RL Component)
======================================

Defines the `InterpretabilityReward` class as a `BaseReward` subclass.
This reward component integrates various interpretability metrics (fidelity,
simplicity, consistency, insight) by delegating their calculation to the
`InterpretabilityEvaluator` and combining them into a single scalar reward
for the Reinforcement Learning agent.
"""

import numpy as np
import torch.nn as nn # For type hinting target_model as nn.Module
from typing import Any, Dict, List, Optional, Union

# Import BaseReward for inheritance
from JanusAI.ml.rewards.base_reward import BaseReward

# Import the new evaluation utilities
from JanusAI.ai_interpretability.evaluation.fidelity import ModelFidelityEvaluator
from JanusAI.ai_interpretability.evaluation.consistency import InterpretabilityEvaluator

# Placeholder for AI model classes
try:
    from janus.ml.networks.hypothesis_net import HypothesisNet, AIHypothesisNet
except ImportError:
    print("Warning: HypothesisNet or AIHypothesisNet not found for type hinting in interpretability_reward.py. Using generic nn.Module.")
    HypothesisNet = nn.Module
    AIHypothesisNet = nn.Module


class InterpretabilityReward(BaseReward):
    """
    Calculates a combined interpretability reward for symbolic expressions,
    acting as a `BaseReward` component in an RL loop.
    It leverages `ModelFidelityEvaluator` and `InterpretabilityEvaluator`
    to quantify fidelity, simplicity, consistency, and insight.
    """

    def __init__(self,
                 weight: float = 1.0, # Overall weight for this combined interpretability reward
                 fidelity_weight: float = 0.4,
                 simplicity_weight: float = 0.3,
                 consistency_weight: float = 0.2,
                 insight_weight: float = 0.1,
                 complexity_penalty_factor: float = 0.01,
                 max_complexity_for_penalty: Optional[int] = None,
                 interpretability_metric: str = 'mdl', # Primary qualitative metric for insight
                 target_model: Optional[Union[HypothesisNet, AIHypothesisNet, nn.Module]] = None,
                 data_samples: Optional[Dict[str, np.ndarray]] = None, # For ModelFidelityEvaluator
                 variables_for_evaluation: Optional[List[Any]] = None # List of Variable objects for evaluation
                ):
        """
        Initializes the InterpretabilityReward.

        Args:
            weight: The overall weight by which this reward component's output
                    will be scaled when added to the total reward.
            fidelity_weight, simplicity_weight, consistency_weight, insight_weight:
                Weights for the individual interpretability components.
            complexity_penalty_factor, max_complexity_for_penalty:
                Parameters for the simplicity metric.
            interpretability_metric: Type of qualitative metric for insight.
            target_model: The AI model whose behavior is being interpreted.
            data_samples: Input/output data samples from the AI model.
            variables_for_evaluation: List of `Variable` objects relevant to the data and expressions.
                                      Necessary for symbolic expression evaluation.
        """
        super().__init__(weight) # Initialize BaseReward with overall weight

        self.component_weights: Dict[str, float] = { # Renamed to avoid conflict with BaseReward's `weight`
            'fidelity': fidelity_weight,
            'simplicity': simplicity_weight,
            'consistency': consistency_weight,
            'insight': insight_weight
        }

        if target_model is None or data_samples is None or variables_for_evaluation is None:
            # If critical components for fidelity/consistency are missing, warn and disable these parts
            print("Warning: InterpretabilityReward initialized without complete data/model/variables. Fidelity and Consistency metrics will be 0.")
            self.model_fidelity_evaluator = None
            self.interpretability_evaluator = None
        else:
            # Initialize the ModelFidelityEvaluator for fidelity calculation
            self.model_fidelity_evaluator = ModelFidelityEvaluator(
                target_model=target_model,
                data_samples=data_samples,
                variables=variables_for_evaluation, # Pass Variable objects
                loss_type='r_squared' # R-squared is a good default for reward scaling
            )
            
            # Initialize the InterpretabilityEvaluator for simplicity, consistency, and insight
            self.interpretability_evaluator = InterpretabilityEvaluator(
                complexity_penalty_factor=complexity_penalty_factor,
                max_complexity_for_penalty=max_complexity_for_penalty,
                interpretability_metric=interpretability_metric,
                fidelity_evaluator=self.model_fidelity_evaluator # Pass the initialized fidelity evaluator
            )

    def calculate_reward(self,
                         current_observation: Any, # From BaseReward signature
                         action: Any,              # From BaseReward signature
                         next_observation: Any,    # From BaseReward signature
                         reward_from_env: float,   # From BaseReward signature
                         done: bool,               # From BaseReward signature
                         info: Dict[str, Any]) -> float:
        """
        Calculates the combined interpretability reward for a symbolic expression,
        conforming to the `BaseReward` interface.

        Args:
            info: Must contain 'expression' (the hypothesized expression)
                  and 'variables' (list of Variable objects used in environment)
                  from the environment.

        Returns:
            float: The calculated interpretability reward (unweighted by this
                   class's `self.weight`, which is applied by `BaseReward.__call__`).
        """
        expression = info.get('expression')
        if expression is None:
            return -1.0 # Significant penalty for no expression

        # Ensure evaluators are initialized
        if self.model_fidelity_evaluator is None or self.interpretability_evaluator is None:
            return 0.0 # Return 0 if evaluators couldn't be set up

        # Extract variables from info, crucial for evaluation methods
        variables_from_info = info.get('variables', [])
        if not variables_from_info:
            print(f"Warning: InterpretabilityReward received info without 'variables'. Cannot compute metrics for '{expression}'.")
            return 0.0 # Cannot proceed without variables

        # Calculate individual reward components by delegating to evaluators
        fidelity_score = self.model_fidelity_evaluator.calculate_fidelity(expression)
        simplicity_score = self.interpretability_evaluator.calculate_simplicity(expression)
        
        # Consistency and Insight require the AI model and data, which are provided via info
        # if not set in init, or implicitly handled by evaluators if they have internal access.
        # Here we pass directly from info, as the evaluators expect these arguments.
        ai_model_from_info = info.get('ai_model', self.model_fidelity_evaluator.target_model)
        test_data_from_info = info.get('test_data', self.model_fidelity_evaluator.data_samples) # Use data from fidelity evaluator if available
        
        consistency_score = self.interpretability_evaluator.test_consistency(
            expression, ai_model_from_info, test_data_from_info, variables_from_info
        )
        insight_score = self.interpretability_evaluator.calculate_insight_score(
            expression, ai_model_from_info, info.get('additional_context')
        )

        # Combine scores using the defined component weights
        total_unweighted_reward = (
            self.component_weights['fidelity'] * fidelity_score +
            self.component_weights['simplicity'] * simplicity_score +
            self.component_weights['consistency'] * consistency_score +
            self.component_weights['insight'] * insight_score
        )

        return total_unweighted_reward


if __name__ == "__main__":
    # This __main__ block demonstrates the usage of InterpretabilityReward.

    # Mock `Expression`, `Variable`, and `AIModel` classes for testing.
    # In a fully integrated system, these would be directly imported.
    try:
        from janus.core.expressions.expression import Expression as RealExpression, Variable as RealVariable
    except ImportError:
        print("Using mock Expression/Variable for interpretability_reward.py test.")
        @dataclass(eq=True, frozen=False)
        class RealVariable:
            name: str
            index: int
            properties: Dict[str, Any] = field(default_factory=dict)
            symbolic: sp.Symbol = field(init=False)
            def __post_init__(self): self.symbolic = sp.Symbol(self.name)
            def __hash__(self): return hash((self.name, self.index))
            def __str__(self): return self.name
        Variable = RealVariable

        @dataclass(eq=False, frozen=False)
        class RealExpression:
            operator: str
            operands: List[Any]
            _symbolic: Optional[sp.Expr] = field(init=False, repr=False)
            _complexity: int = field(init=False, repr=False)
            def __post_init__(self):
                if self.operator == 'var' and isinstance(self.operands[0], RealVariable): self._symbolic = self.operands[0].symbolic
                elif self.operator == 'const': self._symbolic = sp.Float(self.operands[0])
                elif self.operator == '+': self._symbolic = self.operands[0].symbolic + self.operands[1].symbolic if all(hasattr(o, 'symbolic') for o in self.operands) else sp.Symbol('dummy_add')
                else: self._symbolic = sp.sympify(self.operator + "(" + ",".join([str(op) for op in self.operands]) + ")")
                self._complexity = len(str(self._symbolic).replace(" ", "")) # Mock complexity
            @property
            def symbolic(self) -> sp.Expr: return self._symbolic
            @property
            def complexity(self) -> int: return self._complexity
            def __str__(self) -> str: return str(self.symbolic)
    Expression = RealExpression

    class MockAIModel(nn.Module): # Mock for target_model
        def __init__(self, input_dim, output_dim): super().__init__(); self.linear = nn.Linear(input_dim, output_dim)
        def forward(self, x): return self.linear(x)

    # Mock evaluate_expression_on_data and calculate_expression_complexity for internal use
    # If janus.core.expressions.symbolic_math or janus.utils.math.operations not fully available
    try:
        from janus.core.expressions.symbolic_math import evaluate_expression_on_data as real_eval_expr_on_data
    except ImportError:
        def evaluate_expression_on_data(expr_str: str, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
            if 'x_0' in data_dict: return data_dict['x_0'] * 2.0 + 1.0
            return np.full(100, 0.0)
    try:
        from janus.utils.math.operations import calculate_expression_complexity as real_calc_expr_comp
    except ImportError:
        def calculate_expression_complexity(expr_str: str) -> int: return len(expr_str)

    print("--- Testing InterpretabilityReward (RL Component) ---")

    # 1. Setup Dummy AI Model and Data for the Reward
    input_dim = 1; output_dim = 1; num_samples = 100
    dummy_input_X = np.arange(num_samples).reshape(-1, input_dim).astype(np.float32)
    dummy_output_Y = (dummy_input_X * 2.0 + 1.0).reshape(-1, output_dim).astype(np.float32) # y = 2x + 1

    data_samples_for_reward = {'input_X': dummy_input_X, 'output_Y': dummy_output_Y}
    variables_for_reward = [Variable("x_0", 0)]

    mock_ai_model_for_reward = MockAIModel(input_dim, output_dim)

    # 2. Initialize InterpretabilityReward
    interpretability_reward_instance = InterpretabilityReward(
        weight=1.0, # Overall weight for this reward
        fidelity_weight=0.5,
        simplicity_weight=0.2,
        consistency_weight=0.2,
        insight_weight=0.1,
        complexity_penalty_factor=0.01,
        max_complexity_for_penalty=10,
        interpretability_metric='mdl',
        target_model=mock_ai_model_for_reward, # Pass the AI model
        data_samples=data_samples_for_reward, # Pass the data
        variables_for_evaluation=variables_for_reward # Pass variables
    )

    print(f"Reward Initialized with Overall Weight: {interpretability_reward_instance.weight}")
    print(f"Component Weights: {interpretability_reward_instance.component_weights}")


    # 3. Create Test Expressions (using the mocked Expression class)
    perfect_expr_reward = Expression(operator='+', operands=[
        Expression(operator='*', operands=[Expression(operator='const', operands=[2.0]), Variable("x_0", 0)]),
        Expression(operator='const', operands=[1.0])
    ])

    imperfect_expr_reward = Expression(operator='+', operands=[
        Expression(operator='*', operands=[Expression(operator='const', operands=[1.5]), Variable("x_0", 0)]),
        Expression(operator='const', operands=[0.5])
    ])
    
    # Info dict for `calculate_reward` (conforms to BaseReward's `info` argument)
    info_perfect = {
        'expression': perfect_expr_reward, 
        'variables': variables_for_reward,
        # 'ai_model' and 'test_data' could be passed here if not in init, but init handles it.
        'additional_context': {'ai_interpretability_target': 'linear_regression'}
    }
    info_imperfect = {
        'expression': imperfect_expr_reward, 
        'variables': variables_for_reward,
        'additional_context': {'ai_interpretability_target': 'linear_regression'}
    }
    info_no_expr = {
        'expression': None, 
        'variables': variables_for_reward,
        'additional_context': {'ai_interpretability_target': 'linear_regression'}
    }


    # 4. Calculate Rewards using the BaseReward-compliant `calculate_reward` method
    # Signature: current_observation, action, next_observation, reward_from_env, done, info
    reward_perfect = interpretability_reward_instance.calculate_reward(None, None, None, 0.0, False, info_perfect)
    reward_imperfect = interpretability_reward_instance.calculate_reward(None, None, None, 0.0, False, info_imperfect)
    reward_no_expr = interpretability_reward_instance.calculate_reward(None, None, None, 0.0, False, info_no_expr)

    print(f"\nReward for Perfect Expression: {reward_perfect:.4f}")
    print(f"Reward for Imperfect Expression: {reward_imperfect:.4f}")
    print(f"Reward for No Expression: {reward_no_expr:.4f}")

    # Check overall effect with __call__ (which applies `self.weight`)
    total_reward_perfect = interpretability_reward_instance(None, None, None, 0.0, False, info_perfect)
    print(f"Total Reward (weighted) for Perfect Expression: {total_reward_perfect:.4f}")


    print("\nInterpretabilityReward tests completed.")

