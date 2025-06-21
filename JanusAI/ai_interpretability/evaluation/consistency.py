"""
Consistency and Other Interpretability Metrics for AI Interpretability
=====================================================================

Provides a consolidated evaluator for various aspects of interpretability,
including simplicity, consistency across data subsets, and qualitative insight.
These metrics are used to quantify the quality of symbolic explanations for AI models.
"""

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Import core components
from JanusAI.core.expressions.expression import Expression, Variable
from JanusAI.core.expressions.symbolic_math import evaluate_expression_on_data, are_expressions_equivalent_sympy # For comparison and evaluation
from JanusAI.utils.math.operations import calculate_expression_complexity # For simplicity metrics

# Import the new ModelFidelityEvaluator for consistency checks
from JanusAI.ai_interpretability.evaluation.fidelity import ModelFidelityEvaluator

# Placeholder for AI model classes
try:
    from JanusAI.ml.networks.hypothesis_net import HypothesisNet, AIHypothesisNet
except ImportError:
    print("Warning: HypothesisNet or AIHypothesisNet not found for type hinting in consistency.py. Using generic nn.Module.")
    HypothesisNet = nn.Module
    AIHypothesisNet = nn.Module


class InterpretabilityEvaluator:
    """
    Evaluates symbolic expressions based on multiple interpretability criteria:
    simplicity, consistency (generalization), and insightfulness.
    """

    def __init__(self,
                 complexity_penalty_factor: float = 0.01,
                 max_complexity_for_penalty: Optional[int] = None,
                 interpretability_metric: str = 'mdl', # Primary qualitative metric for insight
                 fidelity_evaluator: Optional[ModelFidelityEvaluator] = None # Optional external fidelity evaluator
                ):
        """
        Initializes the InterpretabilityEvaluator.

        Args:
            complexity_penalty_factor: Factor for penalizing high complexity.
            max_complexity_for_penalty: Complexity threshold above which penalty applies.
            interpretability_metric: A string indicating the primary qualitative metric
                                     (e.g., 'mdl', 'simplicity', 'alignment').
            fidelity_evaluator: An optional instance of `ModelFidelityEvaluator`. If provided,
                                this evaluator will be used for consistency checks. If None,
                                consistency checks may be limited.
        """
        self.complexity_penalty_factor = complexity_penalty_factor
        self.max_complexity_for_penalty = max_complexity_for_penalty
        self.interpretability_metric = interpretability_metric.lower()
        self.fidelity_evaluator = fidelity_evaluator

        if self.interpretability_metric not in ['mdl', 'simplicity', 'alignment']:
            print(f"Warning: Unsupported interpretability_metric: {interpretability_metric}. Defaulting to 'simplicity'.")
            self.interpretability_metric = 'simplicity'

    def calculate_simplicity(self, expression: Union[str, Expression]) -> float:
        """
        Calculates a simplicity score for a given expression.
        Lower complexity results in a higher simplicity score.

        Args:
            expression: The symbolic expression (string or Expression object).

        Returns:
            float: A score between 0.0 and 1.0, where 1.0 is highest simplicity.
        """
        # Get complexity from Expression object or calculate from string
        complexity = getattr(expression, 'complexity', calculate_expression_complexity(str(expression)))
        
        # Base simplicity score (inverse proportional to complexity)
        base_simplicity_score = 1.0 / (1.0 + complexity)
        
        # Apply penalty for excessive complexity if threshold is set
        penalty = 0.0
        if self.max_complexity_for_penalty is not None and complexity > self.max_complexity_for_penalty:
            penalty = self.complexity_penalty_factor * (complexity - self.max_complexity_for_penalty)
        
        simplicity_score = base_simplicity_score - penalty
        return max(0.0, min(1.0, simplicity_score)) # Clip score to [0, 1] range


    def test_consistency(self, 
                         expression: Union[str, Expression], 
                         ai_model: Union[HypothesisNet, AIHypothesisNet, nn.Module], 
                         test_data: Dict[str, np.ndarray],
                         variables: List[Variable] # Required for evaluation
                        ) -> float:
        """
        Tests the consistency (generalization) of the symbolic expression across different
        subsets of the provided `test_data`. This is done by splitting the data into folds
        and evaluating the fidelity of the expression on each fold. High consistency implies
        low variance in fidelity across folds.

        Args:
            expression: The symbolic expression to test.
            ai_model: The target AI model.
            test_data: A dictionary containing 'input_X' and 'output_Y' of the AI model.
            variables: List of `Variable` objects corresponding to inputs in `test_data['input_X']`.

        Returns:
            float: A consistency score between 0.0 and 1.0. 1.0 indicates perfect consistency.
        """
        inputs = test_data.get('inputs', test_data.get('input_X'))
        outputs = test_data.get('outputs', test_data.get('output_Y'))

        if inputs is None or outputs is None or len(inputs) < 20: # Need sufficient data for folds
            return 0.5  # Neutral score if insufficient data

        try:
            n_samples = len(inputs)
            n_folds = min(5, n_samples // 10) # Aim for at least 10 samples per fold, max 5 folds
            
            if n_folds < 2:
                return 0.5 # Cannot test consistency with too little data for multiple folds
            
            fold_size = n_samples // n_folds
            fidelity_scores_per_fold = []
            
            for i in range(n_folds):
                start_idx = i * fold_size
                end_idx = start_idx + fold_size if i < n_folds - 1 else n_samples
                
                fold_inputs = inputs[start_idx:end_idx]
                fold_outputs = outputs[start_idx:end_idx]
                
                # Create a temporary ModelFidelityEvaluator for this fold if not provided at init
                # or use the pre-initialized one.
                if self.fidelity_evaluator:
                    # Create a new evaluator for this fold's data subset if it's data-specific
                    # Or modify existing one, but creating new is safer for fold isolation.
                    # This implies ModelFidelityEvaluator can be quickly re-initialized.
                    # For simplicity, create a temp one with the subset.
                    fold_data_samples = {'input_X': fold_inputs, 'output_Y': fold_outputs}
                    temp_evaluator = ModelFidelityEvaluator(ai_model, fold_data_samples, variables, loss_type='r_squared')
                    fold_fidelity = temp_evaluator.calculate_fidelity(expression)
                else:
                    # Fallback to direct evaluation if no dedicated fidelity evaluator is available
                    # This requires `evaluate_expression_on_data` to be called directly.
                    # It would rely on the caller providing a generic way to map inputs to `variables`.
                    fold_evaluation_data_dict = {}
                    for j, var_obj in enumerate(variables):
                        if j < fold_inputs.shape[1]:
                            fold_evaluation_data_dict[var_obj.name] = fold_inputs[:, j]
                    
                    predicted_outputs_fold = evaluate_expression_on_data(str(expression), fold_evaluation_data_dict)
                    
                    if predicted_outputs_fold is None or predicted_outputs_fold.size == 0 or np.any(np.isnan(predicted_outputs_fold)):
                        fold_fidelity = 0.0
                    else:
                        predicted_outputs_fold = np.asarray(predicted_outputs_fold).flatten()
                        fold_outputs_flat = np.asarray(fold_outputs).flatten()
                        valid_mask = np.isfinite(predicted_outputs_fold) & np.isfinite(fold_outputs_flat)
                        if np.any(valid_mask):
                            ss_res = np.sum((fold_outputs_flat[valid_mask] - predicted_outputs_fold[valid_mask]) ** 2)
                            ss_tot = np.sum((fold_outputs_flat[valid_mask] - np.mean(fold_outputs_flat[valid_mask])) ** 2)
                            fold_r_squared = 1 - (ss_res / (ss_tot + 1e-9))
                            fold_fidelity = max(0.0, fold_r_squared)
                        else:
                            fold_fidelity = 0.0

                fidelity_scores_per_fold.append(fold_fidelity)
            
            if not fidelity_scores_per_fold:
                return 0.0 # No valid fidelity scores across any fold

            mean_fidelity = np.mean(fidelity_scores_per_fold)
            fidelity_std = np.std(fidelity_scores_per_fold)
            
            # Consistency is high if mean fidelity is good and standard deviation is low
            # Scale std to [0,1] range (max std would be 0.5 if values are 0 and 1)
            consistency_score = mean_fidelity * (1.0 - min(1.0, fidelity_std * 2.0)) # Higher std -> lower consistency
            
            return max(0.0, min(1.0, consistency_score))

        except Exception as e:
            # print(f"Warning: Consistency test failed for '{expression}': {e}")
            return 0.0 # Return 0 for calculation errors


    def calculate_insight_score(self, 
                                 expression: Union[str, Expression], 
                                 ai_model: Union[HypothesisNet, AIHypothesisNet, nn.Module], 
                                 additional_context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculates an insightfulness score for a symbolic expression.
        Rewards expressions that reveal meaningful patterns or relationships,
        potentially aligning with known AI model structures or general scientific principles.

        Args:
            expression: The symbolic expression to evaluate.
            ai_model: The target AI model.
            additional_context: Optional dictionary with extra information, e.g.,
                                'ai_interpretability_target' (e.g., 'attention', 'activation'),
                                or 'known_constants' from physics.

        Returns:
            float: An insight score between 0.0 and 1.0.
        """
        insight_score = 0.0
        
        try:
            # Get symbolic representation from the Expression object or parse string
            sympy_expr = expression.symbolic if hasattr(expression, 'symbolic') else sp.sympify(str(expression))

            if sympy_expr is None:
                return 0.0

            expr_str = str(sympy_expr) # Use string for basic pattern matching

            # 1. Structural simplicity / common patterns (can overlap with complexity)
            if sympy_expr.is_polynomial():
                insight_score += 0.2
            if sympy_expr.is_rational_function(): # P(x)/Q(x)
                insight_score += 0.1

            # 2. Presence of meaningful mathematical functions (often indicative of real-world phenomena)
            meaningful_functions_keywords = ['exp', 'log', 'sin', 'cos', 'tanh', 'sigmoid', 'sqrt', 'abs']
            for func_keyword in meaningful_functions_keywords:
                if func_keyword in expr_str.lower(): # Case-insensitive check
                    insight_score += 0.05 # Small bonus per meaningful function

            # 3. Number of variables: penalize too few (trivial) or too many (overfitting/uninterpretable)
            n_variables = len(sympy_expr.free_symbols)
            if 1 <= n_variables <= 3: # Optimal number of variables for human interpretation
                insight_score += 0.3
            elif n_variables == 0: # Constant expression, low insight unless specific context
                insight_score -= 0.1
            elif n_variables > 5: # Too many variables, potential overfitting or hard to grasp
                insight_score -= 0.2

            # 4. Alignment with known AI model structures/patterns (context-dependent)
            if additional_context and 'ai_interpretability_target' in additional_context:
                target_pattern = str(additional_context['ai_interpretability_target']).lower()
                
                # If interpreting an attention mechanism (expecting softmax-like patterns)
                if 'attention' in target_pattern:
                    if 'exp' in expr_str or 'log' in expr_str or '*' in expr_str: # e.g., softmax(score) = exp(score) / sum(exp(score))
                        insight_score += 0.2
                
                # If interpreting an activation function (expecting non-linearities, thresholds)
                elif 'activation' in target_pattern:
                    if any(kw in expr_str for kw in ['Piecewise', 'Max', 'Heaviside', 'Abs', 'floor', 'ceiling', 'sigmoid', 'tanh', 'relu']):
                        insight_score += 0.2
            
            # 5. Penalize very high complexity (might indicate uninterpretable spaghetti code)
            complexity = getattr(expression, 'complexity', calculate_expression_complexity(expr_str))
            if complexity > 30: # Arbitrary high complexity threshold
                insight_score -= 0.15
            
            return max(0.0, min(1.0, insight_score)) # Clip score to [0, 1] range

        except Exception as e:
            # print(f"Warning: Insight calculation failed for '{expression}': {e}")
            return 0.0 # Return 0 for calculation errors


if __name__ == "__main__":
    # Mock symbolic_math, Expression, Variable, HypothesisNet for testing
    try:
        from JanusAI.core.expressions.expression import Expression as RealExpression, Variable as RealVariable
        from JanusAI.core.expressions.symbolic_math import evaluate_expression_on_data as real_eval_expr_on_data
        from JanusAI.core.expressions.symbolic_math import are_expressions_equivalent_sympy as real_are_eq_sympy
        from JanusAI.utils.math.operations import calculate_expression_complexity as real_calc_expr_comp
    except ImportError:
        print("Using mock dependencies for consistency.py test.")
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

        def evaluate_expression_on_data(expr_str: str, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
            if 'x_0' in data_dict: return data_dict['x_0'] * 2.0 + 1.0 # Mock linear data
            return np.full(100, 0.0)
        def are_expressions_equivalent_sympy(expr1: sp.Expr, expr2: sp.Expr, symbols: List[sp.Symbol], tolerance: float) -> bool:
            return sp.simplify(expr1) == sp.simplify(expr2) # Simple equivalence
        def calculate_expression_complexity(expr_str: str) -> int:
            return len(expr_str.replace(" ", "")) # Mock complexity

    class MockAIModel(nn.Module): # Mock for target_model
        def __init__(self): super().__init__(); self.linear = nn.Linear(1,1); nn.init.constant_(self.linear.weight, 2.0); nn.init.constant_(self.linear.bias, 1.0)
        def forward(self, x): return self.linear(x)

    print("--- Testing InterpretabilityEvaluator ---")

    # 1. Setup Data and Model for Fidelity/Consistency
    input_dim = 1; output_dim = 1; num_samples = 100
    dummy_input_X = np.arange(num_samples).reshape(-1, input_dim).astype(np.float32)
    dummy_output_Y = (dummy_input_X * 2.0 + 1.0).reshape(-1, output_dim).astype(np.float32) # y = 2x + 1
    data_samples = {'input_X': dummy_input_X, 'output_Y': dummy_output_Y}
    variables = [Variable("x_0", 0)]

    mock_ai_model = MockAIModel()

    # Create FidelityEvaluator instance (required by InterpretabilityEvaluator)
    fidelity_evaluator = ModelFidelityEvaluator(mock_ai_model, data_samples, variables, loss_type='r_squared')

    # 2. Initialize InterpretabilityEvaluator
    evaluator = InterpretabilityEvaluator(
        complexity_penalty_factor=0.01,
        max_complexity_for_penalty=10, # Max complexity before penalty
        interpretability_metric='mdl',
        fidelity_evaluator=fidelity_evaluator # Pass the instance
    )

    # 3. Create Test Expressions
    perfect_expr = Expression(operator='+', operands=[
        Expression(operator='*', operands=[Expression(operator='const', operands=[2.0]), Variable("x_0", 0)]),
        Expression(operator='const', operands=[1.0])
    ]) # Complexity approx 8 (2*x_0+1)

    simple_wrong_expr = Expression(operator='+', operands=[Variable("x_0", 0), Expression(operator='const', operands=[0.0])]) # x_0, Complexity approx 3
    
    complex_accurate_expr = Expression(operator='+', operands=[
        perfect_expr,
        Expression(operator='*', operands=[
            Expression(operator='sin', operands=[Variable("x_0", 0)]),
            Expression(operator='exp', operands=[Variable("x_0", 0)])
        ])
    ]) # e.g., (2*x_0 + 1) + sin(x_0)*exp(x_0), Complexity > 10 (will be penalized)

    # 4. Test Simplicity
    simplicity_perfect = evaluator.calculate_simplicity(perfect_expr)
    simplicity_simple_wrong = evaluator.calculate_simplicity(simple_wrong_expr)
    simplicity_complex_accurate = evaluator.calculate_simplicity(complex_accurate_expr)

    print(f"\nSimplicity (perfect_expr, C={perfect_expr.complexity}): {simplicity_perfect:.4f}")
    print(f"Simplicity (simple_wrong_expr, C={simple_wrong_expr.complexity}): {simplicity_simple_wrong:.4f}")
    print(f"Simplicity (complex_accurate_expr, C={complex_accurate_expr.complexity}): {simplicity_complex_accurate:.4f}")
    assert simplicity_simple_wrong > simplicity_perfect > simplicity_complex_accurate # Expected order

    # 5. Test Consistency
    # Consistency requires splitting data, so prepare data_samples for it
    consistency_perfect = evaluator.test_consistency(perfect_expr, mock_ai_model, data_samples, variables)
    consistency_complex = evaluator.test_consistency(complex_accurate_expr, mock_ai_model, data_samples, variables)
    print(f"\nConsistency (perfect_expr): {consistency_perfect:.4f} (Expected high)")
    print(f"Consistency (complex_accurate_expr): {consistency_complex:.4f} (Expected lower due to noise/potential overfitting)")
    assert consistency_perfect > consistency_complex # Perfect should be more consistent

    # 6. Test Insight Score
    insight_perfect = evaluator.calculate_insight_score(perfect_expr, mock_ai_model)
    insight_complex = evaluator.calculate_insight_score(complex_accurate_expr, mock_ai_model)
    insight_with_context = evaluator.calculate_insight_score(perfect_expr, mock_ai_model, additional_context={'ai_interpretability_target': 'activation'})

    print(f"\nInsight (perfect_expr): {insight_perfect:.4f}")
    print(f"Insight (complex_accurate_expr): {insight_complex:.4f}")
    print(f"Insight (perfect_expr with 'activation' context): {insight_with_context:.4f}")
    assert insight_perfect > 0.0 # Should get some base score
    assert insight_complex < insight_perfect # Complex might get lower insight
    assert insight_with_context >= insight_perfect # Context should give non-negative bonus

    print("\nInterpretabilityEvaluator tests completed.")

