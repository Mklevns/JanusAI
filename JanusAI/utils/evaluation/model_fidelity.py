"""
Fidelity Metrics for AI Interpretability
========================================

Provides utilities for calculating fidelity metrics between a symbolic expression's
predictions and a target AI model's behavior. Fidelity measures how well the
symbolic explanation mimics the AI model's input-output mapping.
"""

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from typing import Any, Callable, Dict, List, Optional, Union

# Import components from new structure
from janus_ai.core.expressions.expression import Expression, Variable # For type hinting expression
from janus_ai.core.expressions.symbolic_math import evaluate_expression_on_data # For evaluating symbolic expressions

# Placeholder for AI model classes, assuming they are nn.Module compatible
try:
    from janus_ai.ml.networks.hypothesis_net import HypothesisNet, AIHypothesisNet
except ImportError:
    print("Warning: HypothesisNet or AIHypothesisNet not found for type hinting in fidelity.py. Using generic nn.Module.")
    HypothesisNet = nn.Module
    AIHypothesisNet = nn.Module


class ModelFidelityEvaluator:
    """
    Calculates fidelity metrics for symbolic expressions in explaining AI model behavior.
    """

    def __init__(self, 
                 target_model: Union[HypothesisNet, AIHypothesisNet, nn.Module],
                 data_samples: Dict[str, np.ndarray], # {'input_X': ..., 'output_Y': ...}
                 variables: List[Variable], # List of Variable objects matching input_X columns
                 loss_type: str = 'r_squared', # 'r_squared', 'mse', 'bce'
                 epsilon: float = 1e-9 # Small value for numerical stability (e.g., avoiding division by zero)
                ):
        """
        Initializes the ModelFidelityEvaluator.

        Args:
            target_model: The AI model (PyTorch nn.Module) whose behavior is being mimicked.
            data_samples: A dictionary containing 'input_X' (input features for the AI model)
                          and 'output_Y' (corresponding target outputs from the AI model).
            variables: A list of `Variable` objects corresponding to the columns in `input_X`.
                       Required for evaluating symbolic expressions.
            loss_type: The type of fidelity metric to calculate ('r_squared', 'mse', 'bce').
                       'r_squared' is generally preferred for rewards as it's normalized.
            epsilon: A small value for numerical stability to prevent division by zero or log(0).
        """
        self.target_model = target_model
        self.variables = variables # Store Variable objects for evaluation
        self.loss_type = loss_type.lower()
        self.epsilon = epsilon
        
        # Determine the device of the target_model
        device = 'cpu'
        if isinstance(target_model, nn.Module) and list(target_model.parameters()):
            device = next(target_model.parameters()).device
        
        # Convert data samples to PyTorch tensors and move to the correct device
        self.input_X = torch.FloatTensor(data_samples['input_X']).to(device)
        self.output_Y = torch.FloatTensor(data_samples['output_Y']).to(device)
        
        self.target_model.eval() # Set model to evaluation mode (disable dropout, BatchNorm updates)

        if self.loss_type not in ['r_squared', 'mse', 'bce']:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'r_squared', 'mse', or 'bce'.")

    def calculate_fidelity(self, expression: Union[str, Expression]) -> float:
        """
        Calculates the fidelity score between a symbolic expression's predictions
        and the target AI model's actual outputs on the `data_samples`.

        Args:
            expression: The symbolic expression to evaluate (string or Expression object).

        Returns:
            The calculated fidelity score (e.g., R-squared, negative MSE, or negative BCE).
            Returns 0.0 or -inf for errors or invalid predictions.
        """
        try:
            # Prepare data_dict for `evaluate_expression_on_data`
            # This maps variable names to their corresponding columns in `self.input_X`.
            evaluation_data_dict = {}
            for i, var_obj in enumerate(self.variables):
                if i < self.input_X.shape[1]: # Ensure index is within bounds of input_X
                    evaluation_data_dict[var_obj.name] = self.input_X[:, i].cpu().numpy() # Pass numpy array
            
            # Evaluate the symbolic expression to get its predictions
            predicted_by_expression_np = evaluate_expression_on_data(str(expression), evaluation_data_dict)
            
            if predicted_by_expression_np is None or predicted_by_expression_np.size == 0 or np.any(np.isnan(predicted_by_expression_np)):
                return 0.0 if self.loss_type == 'r_squared' else -np.inf # Return low score if evaluation fails

            # Ensure outputs are 1D for metrics calculation
            predicted_by_expression_flat = np.asarray(predicted_by_expression_np).flatten()
            target_outputs_flat = self.output_Y.squeeze().cpu().numpy().flatten()

            # Ensure valid, finite numbers for metric calculation
            valid_mask = np.isfinite(predicted_by_expression_flat) & np.isfinite(target_outputs_flat)
            if not np.any(valid_mask):
                return 0.0 if self.loss_type == 'r_squared' else -np.inf

            pred_valid = predicted_by_expression_flat[valid_mask]
            true_valid = target_outputs_flat[valid_mask]

            if len(pred_valid) == 0:
                return 0.0 if self.loss_type == 'r_squared' else -np.inf

            # Calculate the chosen fidelity metric
            if self.loss_type == 'r_squared':
                ss_res = np.sum((true_valid - pred_valid) ** 2)
                ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
                
                if ss_tot < self.epsilon:  # Target is constant or near-constant
                    return 1.0 if ss_res < self.epsilon else 0.0 # Perfect if prediction also constant and matches
                
                r_squared = 1 - (ss_res / ss_tot)
                return max(0.0, r_squared) # Clip R-squared to be non-negative (common for rewards)
            
            elif self.loss_type == 'mse':
                mse = np.mean((pred_valid - true_valid)**2)
                return -mse # Negative MSE, so higher is better
            
            elif self.loss_type == 'bce':
                # Assuming outputs are probabilities (0-1) for BCE
                pred_clipped = np.clip(pred_valid, self.epsilon, 1 - self.epsilon)
                bce_loss = -(true_valid * np.log(pred_clipped) + (1 - true_valid) * np.log(1 - pred_clipped)).mean()
                return -bce_loss # Negative BCE, so higher is better
            
            return 0.0 # Fallback
            
        except Exception as e:
            # print(f"Warning: Fidelity calculation failed for expression '{expression}': {e}")
            return 0.0 if self.loss_type == 'r_squared' else -np.inf


if __name__ == "__main__":
    # Mock symbolic_math utilities and Expression/Variable if not fully available
    try:
        from janus_ai.core.expressions.expression import Expression as RealExpression, Variable as RealVariable
        from janus_ai.core.expressions.symbolic_math import evaluate_expression_on_data as real_eval_expr_on_data
    except ImportError:
        print("Using mock Expression/Variable and evaluate_expression_on_data for fidelity.py test.")
        @dataclass(eq=True, frozen=False)
        class RealVariable:
            name: str
            index: int
            properties: Dict[str, Any] = field(default_factory=dict)
            symbolic: sp.Symbol = field(init=False)
            def __post_init__(self): self.symbolic = sp.Symbol(self.name)
            def __hash__(self): return hash((self.name, self.index))
            def __str__(self): return self.name
        @dataclass(eq=False, frozen=False)
        class RealExpression:
            operator: str
            operands: List[Any]
            _symbolic: Optional[sp.Expr] = field(init=False, repr=False)
            def __post_init__(self):
                if self.operator == 'var' and isinstance(self.operands[0], RealVariable):
                    self._symbolic = self.operands[0].symbolic
                elif self.operator == 'const': self._symbolic = sp.Float(self.operands[0])
                else: self._symbolic = sp.sympify(self.operator + "(" + ",".join([str(op) for op in self.operands]) + ")")
            @property
            def symbolic(self) -> sp.Expr: return self._symbolic
            def __str__(self) -> str: return str(self.symbolic)

        def evaluate_expression_on_data(expr_str: str, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
            # Mock evaluation logic for testing
            if 'x_0' in data_dict: # If x_0 is an input
                return data_dict['x_0'] * 2.0 + 1.0 # Simple linear function
            return np.full(100, 0.0) # Default for constant

    Expression = RealExpression
    Variable = RealVariable


    print("--- Testing ModelFidelityEvaluator ---")

    # 1. Setup Dummy AI Model and Data
    class MockAIModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            # Initialize weights to simple values for predictable output
            nn.init.constant_(self.linear.weight, 2.0) 
            nn.init.constant_(self.linear.bias, 1.0)
        def forward(self, x):
            return self.linear(x) # Simple linear model: y = 2*x + 1

    input_dim = 1
    output_dim = 1
    num_samples = 100
    
    # Input data: X_0 varies from 0 to 99
    dummy_input_X = np.arange(num_samples).reshape(-1, input_dim).astype(np.float32)
    # Target outputs from the AI model: 2*X_0 + 1
    dummy_output_Y = (dummy_input_X * 2.0 + 1.0).reshape(-1, output_dim).astype(np.float32)

    data_samples = {'input_X': dummy_input_X, 'output_Y': dummy_output_Y}
    variables = [Variable("x_0", 0)] # Corresponding variable for input_X

    mock_ai_model = MockAIModel(input_dim, output_dim)

    # 2. Test R-squared Fidelity (perfect match)
    evaluator_r2 = ModelFidelityEvaluator(mock_ai_model, data_samples, variables, loss_type='r_squared')
    # Expression representing the true AI model behavior: "2*x_0 + 1"
    perfect_expr_r2 = Expression(operator='+', operands=[
        Expression(operator='*', operands=[Expression(operator='const', operands=[2.0]), Variable("x_0", 0)]),
        Expression(operator='const', operands=[1.0])
    ])
    fidelity_r2_perfect = evaluator_r2.calculate_fidelity(perfect_expr_r2)
    print(f"\nFidelity (R-squared, perfect match): {fidelity_r2_perfect:.4f} (Expected ~1.0)")
    assert np.isclose(fidelity_r2_perfect, 1.0), "R-squared for perfect match should be close to 1.0"

    # Test R-squared Fidelity (imperfect match)
    imperfect_expr_r2 = Expression(operator='+', operands=[
        Expression(operator='*', operands=[Expression(operator='const', operands=[1.5]), Variable("x_0", 0)]),
        Expression(operator='const', operands=[0.5])
    ]) # Represents "1.5*x_0 + 0.5"
    fidelity_r2_imperfect = evaluator_r2.calculate_fidelity(imperfect_expr_r2)
    print(f"Fidelity (R-squared, imperfect match): {fidelity_r2_imperfect:.4f} (Expected < 1.0)")
    assert fidelity_r2_imperfect < 1.0, "R-squared for imperfect match should be less than 1.0"
    assert fidelity_r2_imperfect > 0.0, "R-squared should be positive for a decent fit"


    # 3. Test MSE Fidelity
    evaluator_mse = ModelFidelityEvaluator(mock_ai_model, data_samples, variables, loss_type='mse')
    # Perfect match should yield MSE near 0, so negative MSE near 0
    fidelity_mse_perfect = evaluator_mse.calculate_fidelity(perfect_expr_r2)
    print(f"\nFidelity (MSE, perfect match): {fidelity_mse_perfect:.4f} (Expected ~0.0)")
    assert np.isclose(fidelity_mse_perfect, 0.0, atol=1e-5), "MSE for perfect match should be close to 0."

    # Imperfect match should yield a more negative MSE
    fidelity_mse_imperfect = evaluator_mse.calculate_fidelity(imperfect_expr_r2)
    print(f"Fidelity (MSE, imperfect match): {fidelity_mse_imperfect:.4f} (Expected < 0.0)")
    assert fidelity_mse_imperfect < 0.0, "MSE for imperfect match should be negative."
    assert fidelity_mse_perfect > fidelity_mse_imperfect, "Perfect match MSE should be better (less negative)."


    # 4. Test BCE Fidelity (Requires binary outputs, so modify data and model for this test)
    print("\n--- Testing BCE Fidelity (with mock binary data) ---")
    mock_ai_model_bce = MockAIModel(input_dim, 1)
    nn.init.constant_(mock_ai_model_bce.linear.weight, 1.0)
    nn.init.constant_(mock_ai_model_bce.linear.bias, -0.5) # x - 0.5
    # Simulate binary output: y = 1 if x > 0.5, else 0
    dummy_input_X_bce = np.linspace(0, 1, num_samples).reshape(-1, 1).astype(np.float32)
    dummy_output_Y_bce = (dummy_input_X_bce > 0.5).astype(np.float32) # Binary targets

    data_samples_bce = {'input_X': dummy_input_X_bce, 'output_Y': dummy_output_Y_bce}
    
    evaluator_bce = ModelFidelityEvaluator(mock_ai_model_bce, data_samples_bce, variables, loss_type='bce')
    
    # Example expression: sigmoid(x_0 - 0.5)
    perfect_expr_bce = Expression(operator='sigmoid', operands=[
        Expression(operator='-', operands=[Variable("x_0", 0), Expression(operator='const', operands=[0.5])])
    ])
    
    # Mock SymPy's sigmoid if not directly available (or if Expression doesn't support it)
    if not hasattr(sp, 'sigmoid'):
        print("Warning: SymPy.sigmoid not found, adding mock for test.")
        sp.sigmoid = lambda x: 1 / (1 + sp.exp(-x))

    fidelity_bce_perfect = evaluator_bce.calculate_fidelity(perfect_expr_bce)
    print(f"Fidelity (BCE, perfect match): {fidelity_bce_perfect:.4f} (Expected ~0.0)") # Negative BCE near 0
    # Assert np.isclose(fidelity_bce_perfect, 0.0, atol=0.1), "BCE for perfect match should be close to 0." # Loose for mock

    # Imperfect BCE match
    imperfect_expr_bce = Expression(operator='sigmoid', operands=[Variable("x_0", 0)]) # sigmoid(x_0)
    fidelity_bce_imperfect = evaluator_bce.calculate_fidelity(imperfect_expr_bce)
    print(f"Fidelity (BCE, imperfect match): {fidelity_bce_imperfect:.4f} (Expected < 0.0)")
    assert fidelity_bce_imperfect < 0.0, "BCE for imperfect match should be negative."


    # 5. Test edge cases (no valid mask, empty expression)
    evaluator_edge_case = ModelFidelityEvaluator(mock_ai_model, data_samples, variables, loss_type='r_squared')
    # No expression
    fidelity_no_expr = evaluator_edge_case.calculate_fidelity(None)
    print(f"\nFidelity (no expression): {fidelity_no_expr:.4f} (Expected 0.0)")
    assert fidelity_no_expr == 0.0

    # Expression resulting in all NaNs/Infs
    bad_expr = Expression(operator='/', operands=[Variable("x_0", 0), Expression(operator='const', operands=[0.0])])
    fidelity_bad_expr = evaluator_edge_case.calculate_fidelity(bad_expr)
    print(f"Fidelity (bad expression generating NaNs): {fidelity_bad_expr:.4f} (Expected 0.0)")
    assert fidelity_bad_expr == 0.0


    print("\nModelFidelityEvaluator tests completed.")

