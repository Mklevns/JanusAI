"""
Tests for ai_interpretability/evaluation/consistency.py
"""
import pytest
import numpy as np
import sympy as sp
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Mock dependencies from janus_ai.core and JanusAI.utils
# These mocks are based on the ones in the __main__ block of consistency.py
@dataclass(eq=True, frozen=False)
class MockVariable:
    name: str
    index: int
    properties: Dict[str, Any] = field(default_factory=dict)
    symbolic: sp.Symbol = field(init=False)

    def __post_init__(self):
        self.symbolic = sp.Symbol(self.name)

    def __hash__(self):
        return hash((self.name, self.index))

    def __str__(self):
        return self.name

@dataclass(eq=False, frozen=False)
class MockExpression:
    operator: str
    operands: List[Any]
    _symbolic: Optional[sp.Expr] = field(init=False, repr=False, default=None)
    _complexity: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        try:
            if self.operator == 'var' and isinstance(self.operands[0], MockVariable):
                self._symbolic = self.operands[0].symbolic
            elif self.operator == 'const':
                # Ensure operand is a float for sp.Float
                self._symbolic = sp.Float(float(self.operands[0]))
            elif self.operator == '+':
                if len(self.operands) == 2 and all(hasattr(o, 'symbolic') and o.symbolic is not None for o in self.operands):
                    self._symbolic = self.operands[0].symbolic + self.operands[1].symbolic
                else:
                    self._symbolic = sp.Symbol('dummy_add') # Fallback for incomplete operands
            elif self.operator == '*':
                 if len(self.operands) == 2 and all(hasattr(o, 'symbolic') and o.symbolic is not None for o in self.operands):
                    self._symbolic = self.operands[0].symbolic * self.operands[1].symbolic
                 else:
                    self._symbolic = sp.Symbol('dummy_mul') # Fallback
            elif self.operator in ['sin', 'exp']: # Example unary ops
                if len(self.operands) == 1 and hasattr(self.operands[0], 'symbolic') and self.operands[0].symbolic is not None:
                    func = getattr(sp, self.operator)
                    self._symbolic = func(self.operands[0].symbolic)
                else:
                    self._symbolic = sp.Symbol(f'dummy_{self.operator}')
            else: # Fallback for other operators
                op_strs = []
                for op in self.operands:
                    if hasattr(op, '_symbolic') and op._symbolic is not None:
                        op_strs.append(str(op._symbolic))
                    else:
                        op_strs.append(str(op)) # Convert operand to string if not symbolic
                self._symbolic = sp.sympify(self.operator + "(" + ",".join(op_strs) + ")", locals={'Symbol': sp.Symbol})

        except Exception: # pylint: disable=broad-except
            # Fallback if SymPy parsing fails
            self._symbolic = sp.Symbol(f"dummy_{self.operator}_{'_'.join(map(str, self.operands))}")

        # Calculate complexity based on the string representation of the symbolic expression
        if self._symbolic is not None:
            self._complexity = len(str(self._symbolic).replace(" ", ""))
        else:
            self._complexity = 100 # High complexity for unparseable


    @property
    def symbolic(self) -> Optional[sp.Expr]:
        return self._symbolic

    @property
    def complexity(self) -> int:
        return self._complexity

    def __str__(self) -> str:
        return str(self.symbolic) if self.symbolic is not None else "InvalidExpression"

class MockAIModel(nn.Module):
    def __init__(self, weight=2.0, bias=1.0):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.constant_(self.linear.weight, weight)
        nn.init.constant_(self.linear.bias, bias)

    def forward(self, x):
        return self.linear(x)

# Mock ModelFidelityEvaluator
class MockModelFidelityEvaluator:
    def __init__(self, ai_model, data_samples, variables, loss_type='r_squared'):
        self.ai_model = ai_model
        self.data_samples = data_samples
        self.variables = variables
        self.loss_type = loss_type

    def calculate_fidelity(self, expression: Union[str, MockExpression]) -> float:
        # Simplified fidelity calculation for testing
        # This mock assumes perfect fidelity for a specific expression string
        if str(expression) == "2.0*x_0 + 1.0":
            return 1.0
        elif "sin" in str(expression) or "exp" in str(expression): # Penalize complex ones
            return 0.7
        # Simulate data evaluation for consistency test if expression is not perfect
        # This part is crucial for the consistency test's internal workings if it uses this mock
        try:
            expr_obj = expression if isinstance(expression, MockExpression) else MockExpression(operator="sympify", operands=[str(expression)])
            expr_str = str(expr_obj.symbolic)

            if 'x_0' in expr_str: # A very crude check
                # Try to evaluate based on a simple linear model if 'x_0' is present
                # This part needs to be more robust if the test_consistency relies heavily on it
                # For now, assume it's testing variance, so a fixed but not perfect score is okay
                return 0.8 # Assume decent but not perfect fidelity for other x_0 expressions
            return 0.5 # Default for unknown expressions
        except Exception: # pylint: disable=broad-except
            return 0.1


# Actual imports from the module to be tested
from ai_interpretability.evaluation.consistency import InterpretabilityEvaluator
# Patch the dependencies in the tested module
from ai_interpretability.evaluation import consistency as consistency_module
consistency_module.Expression = MockExpression
consistency_module.Variable = MockVariable
consistency_module.HypothesisNet = MockAIModel
consistency_module.AIHypothesisNet = MockAIModel
consistency_module.ModelFidelityEvaluator = MockModelFidelityEvaluator
consistency_module.calculate_expression_complexity = lambda expr_str: len(expr_str.replace(" ", ""))
consistency_module.evaluate_expression_on_data = lambda expr_str, data_dict: data_dict.get('x_0', np.array([0])) * 2.0 + 1.0 if 'x_0' in expr_str else np.zeros_like(data_dict.get('x_0', np.array([0])))
consistency_module.are_expressions_equivalent_sympy = lambda e1, e2, s, t: sp.simplify(e1) == sp.simplify(e2)


@pytest.fixture
def mock_variables():
    return [MockVariable("x_0", 0)]

@pytest.fixture
def mock_ai_model_instance():
    return MockAIModel()

@pytest.fixture
def mock_data_samples():
    num_samples = 100
    input_X = np.arange(num_samples).reshape(-1, 1).astype(np.float32)
    output_Y = (input_X * 2.0 + 1.0).reshape(-1, 1).astype(np.float32) # y = 2x + 1
    return {'input_X': input_X, 'output_Y': output_Y, 'inputs': input_X, 'outputs': output_Y}


@pytest.fixture
def mock_fidelity_evaluator(mock_ai_model_instance, mock_data_samples, mock_variables):
    return MockModelFidelityEvaluator(mock_ai_model_instance, mock_data_samples, mock_variables)

@pytest.fixture
def interpretability_evaluator(mock_fidelity_evaluator):
    return InterpretabilityEvaluator(
        complexity_penalty_factor=0.01,
        max_complexity_for_penalty=10,
        interpretability_metric='mdl',
        fidelity_evaluator=mock_fidelity_evaluator
    )

@pytest.fixture
def perfect_expression(mock_variables):
    # Represents 2.0*x_0 + 1.0
    return MockExpression(operator='+', operands=[
        MockExpression(operator='*', operands=[MockExpression(operator='const', operands=[2.0]), mock_variables[0]]),
        MockExpression(operator='const', operands=[1.0])
    ])

@pytest.fixture
def simple_wrong_expression(mock_variables):
    # Represents x_0
    return MockExpression(operator='var', operands=[mock_variables[0]])


@pytest.fixture
def complex_accurate_expression(perfect_expression, mock_variables):
    # Represents (2.0*x_0 + 1.0) + sin(x_0)*exp(x_0)
    # Note: MockExpression's post_init needs to handle 'sin' and 'exp' or be generic
    return MockExpression(operator='+', operands=[
        perfect_expression,
        MockExpression(operator='*', operands=[
            MockExpression(operator='sin', operands=[mock_variables[0]]),
            MockExpression(operator='exp', operands=[mock_variables[0]])
        ])
    ])

def test_calculate_simplicity(interpretability_evaluator, perfect_expression, simple_wrong_expression, complex_accurate_expression):
    simplicity_perfect = interpretability_evaluator.calculate_simplicity(perfect_expression)
    simplicity_simple_wrong = interpretability_evaluator.calculate_simplicity(simple_wrong_expression)
    simplicity_complex_accurate = interpretability_evaluator.calculate_simplicity(complex_accurate_expression)

    # Expected: simple_wrong is simpler than perfect_expr, which is simpler than complex_accurate_expr
    # perfect_expression complexity: "2.0*x_0+1.0" -> 9. Penalty if > 10. Base = 1/(1+9) = 0.1
    # simple_wrong_expression complexity: "x_0" -> 3. Base = 1/(1+3) = 0.25
    # complex_accurate_expression complexity: e.g. "2.0*x_0+1.0+sin(x_0)*exp(x_0)" -> 28. Penalty = 0.01 * (28-10) = 0.18. Base = 1/(1+28) ~ 0.034. Score = 0.034 - 0.18 = negative, so 0.

    assert simplicity_simple_wrong > simplicity_perfect
    # The complex expression will have a very high complexity, likely resulting in a score of 0 after penalty
    assert simplicity_perfect > simplicity_complex_accurate or simplicity_complex_accurate == 0.0

    # Test string input for expression
    simplicity_string_expr = interpretability_evaluator.calculate_simplicity("x_0 + y_0") # Complexity = 7
    expected_simplicity_string = 1.0 / (1.0 + 7) # 1/8 = 0.125
    assert abs(simplicity_string_expr - expected_simplicity_string) < 1e-6


def test_test_consistency(interpretability_evaluator, mock_ai_model_instance, mock_data_samples, mock_variables, perfect_expression, complex_accurate_expression):
    # Consistency with the mock fidelity evaluator that returns 1.0 for perfect_expression
    consistency_perfect = interpretability_evaluator.test_consistency(perfect_expression, mock_ai_model_instance, mock_data_samples, mock_variables)

    # Consistency for a more complex expression (mock fidelity might return lower or more varied scores)
    consistency_complex = interpretability_evaluator.test_consistency(complex_accurate_expression, mock_ai_model_instance, mock_data_samples, mock_variables)

    # With the current MockModelFidelityEvaluator, perfect_expression gets 1.0 fidelity, complex gets 0.7.
    # So, mean_fidelity for perfect is 1.0, std is 0. Consistency = 1.0 * (1-0) = 1.0
    # For complex, mean_fidelity is 0.7, std is 0. Consistency = 0.7 * (1-0) = 0.7
    assert consistency_perfect > 0.9  # Expect high consistency for the "perfect" expression
    assert consistency_complex < consistency_perfect # Complex or less accurate should be less consistent

    # Test with insufficient data (should return 0.5)
    small_data_samples = {'input_X': np.array([[1],[2]]), 'output_Y': np.array([[3],[5]]), 'inputs': np.array([[1],[2]]), 'outputs': np.array([[3],[5]])}
    consistency_insufficient_data = interpretability_evaluator.test_consistency(perfect_expression, mock_ai_model_instance, small_data_samples, mock_variables)
    assert consistency_insufficient_data == 0.5

    # Test with an expression string that causes calculation error in the fallback path
    # (e.g., if evaluate_expression_on_data returns None or NaN)
    # To simulate this, we can make evaluate_expression_on_data return NaN for a specific string
    original_eval = consistency_module.evaluate_expression_on_data
    def failing_eval(expr_str, data_dict):
        if expr_str == "fail_me":
            return np.array([np.nan, np.nan])
        return original_eval(expr_str, data_dict) # Call original for other cases

    consistency_module.evaluate_expression_on_data = failing_eval
    consistency_fail = interpretability_evaluator.test_consistency("fail_me", mock_ai_model_instance, mock_data_samples, mock_variables)
    assert consistency_fail == 0.0
    consistency_module.evaluate_expression_on_data = original_eval # Restore


def test_calculate_insight_score(interpretability_evaluator, mock_ai_model_instance, perfect_expression, complex_accurate_expression, simple_wrong_expression):
    insight_perfect = interpretability_evaluator.calculate_insight_score(perfect_expression, mock_ai_model_instance)
    insight_complex = interpretability_evaluator.calculate_insight_score(complex_accurate_expression, mock_ai_model_instance)
    insight_simple_wrong = interpretability_evaluator.calculate_insight_score(simple_wrong_expression, mock_ai_model_instance)

    # Perfect expression (2.0*x_0 + 1.0) - polynomial, 1 var
    # Expected: polynomial(0.2) + 1 var(0.3) = 0.5
    assert abs(insight_perfect - 0.5) < 1e-6


    # Complex expression ((2.0*x_0+1.0)+sin(x_0)*exp(x_0)) - has exp, sin, 1 var, high complexity
    # Expected: poly(0.2), rational(0.1), exp(0.05), sin(0.05), 1 var(0.3) - complexity penalty (0.15 if C > 30)
    # C = len("2.0*x_0+1.0+sin(x_0)*exp(x_0)") = 28. No penalty.
    # Score = 0.2 (poly) + 0.1 (rational) + 0.05 (exp) + 0.05 (sin) + 0.3 (1 var) = 0.7
    # The check for polynomial might be tricky for SymPy with transcendental functions.
    # If not considered polynomial: 0.1 (rational) + 0.05 (exp) + 0.05 (sin) + 0.3 (1 var) = 0.5
    # Let's assume it's not strictly polynomial.
    str_complex_expr = str(complex_accurate_expression.symbolic)
    expected_insight_complex = 0.0
    if complex_accurate_expression.symbolic.is_polynomial(): expected_insight_complex += 0.2
    if complex_accurate_expression.symbolic.is_rational_function(): expected_insight_complex += 0.1
    if "exp" in str_complex_expr: expected_insight_complex += 0.05
    if "sin" in str_complex_expr: expected_insight_complex += 0.05
    if len(complex_accurate_expression.symbolic.free_symbols) == 1: expected_insight_complex += 0.3
    # Recalculate complexity: "2.0*x_0+1.0+exp(x_0)*sin(x_0)" = 29
    if complex_accurate_expression.complexity > 30: expected_insight_complex -=0.15

    # Based on the structure: not polynomial due to sin/exp. Is rational.
    # Expected = 0.1 (rational) + 0.05 (exp) + 0.05 (sin) + 0.3 (1 var) = 0.5
    assert abs(insight_complex - 0.5) < 1e-6


    # Simple wrong expression (x_0) - polynomial, 1 var
    # Expected: polynomial(0.2) + 1 var(0.3) = 0.5
    assert abs(insight_simple_wrong - 0.5) < 1e-6

    # Test with context
    insight_with_context = interpretability_evaluator.calculate_insight_score(
        perfect_expression, mock_ai_model_instance, additional_context={'ai_interpretability_target': 'activation'}
    )
    # perfect_expression is "2.0*x_0+1.0". Context 'activation'. No specific keywords like Piecewise, Max etc.
    # So, context bonus should be 0. Score remains 0.5.
    assert abs(insight_with_context - 0.5) < 1e-6

    # Test with context that should trigger bonus
    # Expression "Abs(x_0)"
    abs_expr = MockExpression(operator="Abs", operands=[simple_wrong_expression.operands[0]]) # Abs(x_0)
    insight_abs_with_context = interpretability_evaluator.calculate_insight_score(
        abs_expr, mock_ai_model_instance, additional_context={'ai_interpretability_target': 'activation'}
    )
    # Abs(x_0): is_polynomial=False (for sympy Abs), is_rational=True. 1 var. "Abs" keyword.
    # Score = 0.1 (rational) + 0.05 (abs in meaningful_functions) + 0.3 (1 var) + 0.2 (activation context bonus for Abs) = 0.65
    # Complexity of "Abs(x_0)" = 7. No penalty.
    assert abs(insight_abs_with_context - 0.65) < 1e-6

    # Test expression that results in None from sympify
    class NoneSymbolicExpression(MockExpression):
        @property
        def symbolic(self):
            return None

    none_expr = NoneSymbolicExpression("invalid", [])
    insight_none = interpretability_evaluator.calculate_insight_score(none_expr, mock_ai_model_instance)
    assert insight_none == 0.0

    # Test expression with 0 variables (constant)
    const_expr = MockExpression(operator='const', operands=[5.0]) # "5.0"
    insight_const = interpretability_evaluator.calculate_insight_score(const_expr, mock_ai_model_instance)
    # is_polynomial=True (0.2), is_rational=True (0.1), 0 vars (-0.1) = 0.2
    assert abs(insight_const - 0.2) < 1e-6

    # Test expression with >5 variables
    # Need to create mock variables for this
    multi_vars = [MockVariable(f"x_{i}", i) for i in range(6)]
    # Construct expression like x_0+x_1+x_2+x_3+x_4+x_5
    curr_expr = MockExpression(operator='var', operands=[multi_vars[0]])
    for i in range(1, 6):
        curr_expr = MockExpression(operator='+', operands=[curr_expr, MockExpression(operator='var', operands=[multi_vars[i]])])

    insight_multi_var = interpretability_evaluator.calculate_insight_score(curr_expr, mock_ai_model_instance)
    # is_polynomial=True (0.2), is_rational=True (0.1), >5 vars (-0.2) = 0.1
    # Complexity of "x_0+x_1+x_2+x_3+x_4+x_5" is 23. No penalty.
    assert abs(insight_multi_var - 0.1) < 1e-6

def test_interpretability_evaluator_init_warning():
    """Test that a warning is printed for unsupported interpretability_metric."""
    evaluator = InterpretabilityEvaluator(interpretability_metric='unsupported_metric')
    assert evaluator.interpretability_metric == 'simplicity' # Default fallback
    # Note: Checking for printed output (e.g. to stderr/stdout) is possible with capsys fixture
    # but for now, just check the fallback behavior.

def test_consistency_no_fidelity_evaluator(mock_ai_model_instance, mock_data_samples, mock_variables, perfect_expression):
    """Test consistency calculation when no fidelity_evaluator is provided at init."""
    evaluator = InterpretabilityEvaluator() # No fidelity_evaluator passed

    # The internal fallback uses evaluate_expression_on_data
    # Our patched version: data_dict.get('x_0', np.array([0])) * 2.0 + 1.0 if 'x_0' in expr_str else np.zeros_like(...)
    # For perfect_expression (str: "2.0*x_0+1.0"), this should yield perfect predictions.
    # Thus, R^2 should be 1.0 for each fold.
    # Mean fidelity = 1.0, std = 0. Consistency = 1.0 * (1 - 0) = 1.0
    consistency_score = evaluator.test_consistency(perfect_expression, mock_ai_model_instance, mock_data_samples, mock_variables)
    assert consistency_score > 0.99 # Expect near perfect consistency

    # Test with an expression that our mock evaluate_expression_on_data handles differently
    # e.g. an expression string without 'x_0'
    other_expr_str = "y_0 + 1" # This will result in 0 from mock evaluate_expression_on_data
    consistency_other = evaluator.test_consistency(other_expr_str, mock_ai_model_instance, mock_data_samples, mock_variables)
    # If predictions are all 0, and actual outputs are not all 0, R^2 will be < 1.
    # Let's trace: predicted_outputs_fold will be all zeros.
    # fold_outputs_flat is (dummy_input_X * 2.0 + 1.0), e.g. [1,3,5...]
    # ss_res = sum((fold_outputs_flat - 0)^2) = sum(fold_outputs_flat^2)
    # ss_tot = sum((fold_outputs_flat - mean(fold_outputs_flat))^2)
    # r_squared = 1 - ss_res / ss_tot. If ss_res > ss_tot, r_squared can be negative. Max(0, r_squared).
    # If model predicts constant 0 for non-constant data, R2 is typically low or 0.
    assert consistency_other <= 0.1 # Expect low consistency
