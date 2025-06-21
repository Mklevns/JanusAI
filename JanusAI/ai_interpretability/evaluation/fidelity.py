# JanusAI/ai_interpretability/evaluation/fidelity.py
"""
Complete implementation of the FidelityCalculator for AI interpretability.
This is the core function that measures how well our symbolic expressions
capture the behavior of attention heads in transformers.
"""

import numpy as np
import torch
import sympy as sp
from typing import Any, Dict, Optional, Union, Tuple, List
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings


class FidelityCalculator:
    """
    Robust fidelity calculation for AI interpretability experiments.
    
    Measures how well symbolic expressions reproduce AI model behavior,
    specifically designed for attention pattern discovery.
    """
    
    def __init__(self, 
                 correlation_threshold: float = 0.1,
                 max_eval_points: int = 10000,
                 numerical_tolerance: float = 1e-6):
        self.correlation_threshold = correlation_threshold
        self.max_eval_points = max_eval_points
        self.numerical_tolerance = numerical_tolerance
        
    def calculate_fidelity(self, 
                          expression: Any,
                          ai_model: Any,
                          test_data: Dict[str, np.ndarray],
                          variables: list,
                          target_behavior: str = 'attention') -> float:
        """
        Calculate fidelity between symbolic expression and AI model behavior.
        
        Args:
            expression: Symbolic expression to evaluate
            ai_model: The AI model (transformer) being interpreted
            test_data: Dictionary containing input data and attention patterns
            variables: List of Variable objects for symbolic evaluation
            target_behavior: Type of behavior to match ('attention', 'output', etc.)
            
        Returns:
            float: Fidelity score between 0 and 1
        """
        try:
            # Step 1: Extract ground truth behavior from AI model
            ground_truth = self._extract_ground_truth(ai_model, test_data, target_behavior)
            
            # Step 2: Evaluate symbolic expression on same inputs
            predicted = self._evaluate_expression(expression, test_data, variables)
            
            # Step 3: Calculate correlation-based fidelity
            fidelity_score = self._compute_correlation_fidelity(ground_truth, predicted)
            
            return max(0.0, min(1.0, fidelity_score))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"Fidelity calculation failed: {e}")
            return 0.0
    
    def _extract_ground_truth(self, 
                             ai_model: Any, 
                             test_data: Dict[str, np.ndarray],
                             target_behavior: str) -> np.ndarray:
        """Extract the actual behavior we want to approximate."""
        
        if target_behavior == 'attention':
            # For pre-computed attention weights
            if 'attention_weights' in test_data:
                layer = test_data.get('target_layer', 0)
                head = test_data.get('target_head', None)
                
                attn_weights = test_data['attention_weights']
                if isinstance(attn_weights, torch.Tensor):
                    attn_weights = attn_weights.cpu().numpy()
                
                # Handle different attention weight formats
                if len(attn_weights.shape) == 4:  # [batch, heads, seq, seq]
                    if head is not None:
                        return attn_weights[:, head, :, :]
                    else:
                        return attn_weights.mean(axis=1)  # Average over heads
                elif len(attn_weights.shape) == 3:  # [batch, seq, seq]
                    return attn_weights
                else:
                    return attn_weights.reshape(-1)
            
            # Otherwise, run forward pass to get attention
            input_ids = test_data.get('input_ids')
            if input_ids is None:
                raise ValueError("No input_ids in test_data")
            
            with torch.no_grad():
                if isinstance(input_ids, np.ndarray):
                    input_ids = torch.tensor(input_ids, device=ai_model.device)
                
                outputs = ai_model(input_ids, output_attentions=True)
                
                # Extract specific layer/head attention
                layer_idx = test_data.get('target_layer', 0)
                head_idx = test_data.get('target_head', None)
                
                attention_weights = outputs.attentions[layer_idx]
                
                if head_idx is not None:
                    # Specific head
                    attention_weights = attention_weights[:, head_idx, :, :]
                else:
                    # Average over all heads
                    attention_weights = attention_weights.mean(dim=1)
                
                return attention_weights.cpu().numpy()
        
        else:
            raise NotImplementedError(f"Target behavior '{target_behavior}' not implemented")
    
    def _evaluate_expression(self, 
                           expression: Any,
                           test_data: Dict[str, np.ndarray],
                           variables: List[Any]) -> np.ndarray:
        """Evaluate symbolic expression on test data."""
        
        # Create variable substitutions based on test data
        var_substitutions = self._create_variable_substitutions(test_data, variables)
        
        # Handle different expression types
        if isinstance(expression, sp.Expr):
            # SymPy expression
            return self._evaluate_sympy_expression(expression, var_substitutions)
        elif hasattr(expression, 'evaluate'):
            # Custom Expression object
            return expression.evaluate(var_substitutions)
        else:
            # Try to convert to SymPy
            try:
                sympy_expr = sp.sympify(str(expression))
                return self._evaluate_sympy_expression(sympy_expr, var_substitutions)
            except:
                raise ValueError(f"Cannot evaluate expression of type {type(expression)}")
    
    def _create_variable_substitutions(self,
                                     test_data: Dict[str, np.ndarray],
                                     variables: List[Any]) -> Dict[str, np.ndarray]:
        """Create substitution dictionary for variables based on test data."""
        
        substitutions = {}
        seq_len = test_data.get('sequence_length', test_data['input_ids'].shape[-1])
        batch_size = test_data['input_ids'].shape[0] if 'input_ids' in test_data else 1
        
        for var in variables:
            var_name = var.name if hasattr(var, 'name') else str(var)
            
            # Position-based variables
            if 'pos' in var_name or 'position' in var_name:
                if 'diff' in var_name:
                    # Position difference matrix
                    pos_diff = np.zeros((seq_len, seq_len))
                    for i in range(seq_len):
                        for j in range(seq_len):
                            pos_diff[i, j] = i - j
                    substitutions[var_name] = pos_diff
                elif 'ratio' in var_name:
                    # Position ratio matrix
                    pos_ratio = np.ones((seq_len, seq_len))
                    for i in range(seq_len):
                        for j in range(seq_len):
                            pos_ratio[i, j] = (i + 1) / (j + 1) if j < i else (j + 1) / (i + 1)
                    substitutions[var_name] = pos_ratio
                else:
                    # Absolute positions
                    substitutions[var_name] = np.arange(seq_len)
            
            # Token-based variables
            elif 'token' in var_name:
                if 'input_ids' in test_data:
                    substitutions[var_name] = test_data['input_ids']
                else:
                    substitutions[var_name] = np.random.randint(0, 1000, (batch_size, seq_len))
            
            # Distance-based variables
            elif 'dist' in var_name:
                dist_matrix = np.zeros((seq_len, seq_len))
                for i in range(seq_len):
                    for j in range(seq_len):
                        dist_matrix[i, j] = abs(i - j)
                substitutions[var_name] = dist_matrix
            
            # Default: zeros
            else:
                substitutions[var_name] = np.zeros((seq_len, seq_len))
        
        return substitutions
    
    def _evaluate_sympy_expression(self,
                                 expr: sp.Expr,
                                 var_subs: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate a SymPy expression with numpy arrays."""
        
        # Convert SymPy expression to a lambda function
        symbols = list(expr.free_symbols)
        symbol_names = [str(s) for s in symbols]
        
        # Create lambdify function
        try:
            func = sp.lambdify(symbols, expr, modules=['numpy'])
            
            # Get values in correct order
            values = []
            for sym_name in symbol_names:
                if sym_name in var_subs:
                    values.append(var_subs[sym_name])
                else:
                    # Default to zeros with appropriate shape
                    if var_subs:
                        shape = next(iter(var_subs.values())).shape
                        values.append(np.zeros(shape))
                    else:
                        values.append(0)
            
            # Evaluate
            result = func(*values)
            return np.array(result)
            
        except Exception as e:
            print(f"Expression evaluation failed: {e}")
            # Return zeros with appropriate shape
            if var_subs:
                shape = next(iter(var_subs.values())).shape
                return np.zeros(shape)
            return np.array([0.0])
    
    def _compute_correlation_fidelity(self,
                                    ground_truth: np.ndarray,
                                    predicted: np.ndarray) -> float:
        """Compute correlation-based fidelity score."""
        
        # Flatten arrays for correlation computation
        ground_truth = np.array(ground_truth).flatten()
        predicted = np.array(predicted).flatten()
        
        # Handle shape mismatches
        min_length = min(len(ground_truth), len(predicted))
        if min_length == 0:
            return 0.0
        
        gt_trimmed = ground_truth[:min_length]
        pred_trimmed = predicted[:min_length]
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(gt_trimmed) & np.isfinite(pred_trimmed)
        if not np.any(valid_mask):
            return 0.0
        
        gt_clean = gt_trimmed[valid_mask]
        pred_clean = pred_trimmed[valid_mask]
        
        # Check for constant arrays
        if np.std(gt_clean) < self.numerical_tolerance or np.std(pred_clean) < self.numerical_tolerance:
            # If one is constant, check if they're close
            return 1.0 if np.allclose(gt_clean, pred_clean, atol=self.numerical_tolerance) else 0.0
        
        # Compute multiple correlation metrics
        try:
            # Pearson correlation (linear relationships)
            pearson_r, _ = pearsonr(gt_clean, pred_clean)
            pearson_score = abs(pearson_r) if not np.isnan(pearson_r) else 0.0
            
            # Spearman correlation (monotonic relationships)
            spearman_r, _ = spearmanr(gt_clean, pred_clean)
            spearman_score = abs(spearman_r) if not np.isnan(spearman_r) else 0.0
            
            # R² score (explained variance)
            r2 = r2_score(gt_clean, pred_clean)
            r2_score_norm = max(0.0, r2)  # R² can be negative
            
            # Combine metrics (weighted average)
            fidelity = 0.5 * pearson_score + 0.3 * spearman_score + 0.2 * r2_score_norm
            
            return fidelity
            
        except Exception as e:
            print(f"Correlation calculation failed: {e}")
            return 0.0


# Integration for InterpretabilityReward class
def integrate_fidelity_into_interpretability_reward():
    """
    Complete implementation of _calculate_fidelity for InterpretabilityReward.
    This function should be called in the InterpretabilityReward.__init__ or
    added as a method to the InterpretabilityReward class.
    """
    
    def _calculate_fidelity(self,
                           expression: Any,
                           ai_model: Any,
                           test_data: Any) -> float:
        """
        Calculate how well the symbolic expression reproduces the AI model's attention behavior.
        Delegates to FidelityCalculator for robust, normalized fidelity scoring.
        """
        try:
            # Initialize calculator once
            if not hasattr(self, '_fidelity_calculator') or self._fidelity_calculator is None:
                self._fidelity_calculator = FidelityCalculator()

            # Normalize test_data into dict format expected by FidelityCalculator
            if hasattr(test_data, 'inputs') and hasattr(test_data, 'attention_weights'):
                data_dict = {
                    'input_ids': np.array(test_data.inputs),
                    'attention_mask': np.array(getattr(test_data, 'attention_mask', 
                                                     np.ones_like(test_data.inputs))),
                    'attention_weights': test_data.attention_weights,
                    'sequence_length': test_data.inputs.shape[-1],
                    'target_layer': getattr(test_data, 'target_layer', 0),
                    'target_head': getattr(test_data, 'target_head', None)
                }
            elif isinstance(test_data, dict):
                data_dict = test_data
            else:
                raise ValueError(f"Unrecognized test_data format: {type(test_data)}")

            # Use variables attribute if present
            variables = getattr(self, 'variables', [])

            return self._fidelity_calculator.calculate_fidelity(
                expression=expression,
                ai_model=ai_model,
                test_data=data_dict,
                variables=variables,
                target_behavior='attention'
            )
        except Exception as e:
            # Log error if logger available
            if hasattr(self, 'logger'):
                self.logger.error(f"Fidelity calculation error: {e}", exc_info=True)
            else:
                print(f"Error in _calculate_fidelity: {e}")
            # Return worst-case fidelity
            return 0.0
    
    return _calculate_fidelity


# Example usage for testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the FidelityCalculator
    calculator = FidelityCalculator()
    
    # Create dummy test data
    test_data = {
        'input_ids': np.random.randint(0, 1000, (2, 16)),
        'sequence_length': 16,
        'target_layer': 0,
        'target_head': 1,
        'attention_weights': np.random.rand(2, 12, 16, 16)  # [batch, heads, seq, seq]
    }
    
    # Create dummy variables
    from dataclasses import dataclass
    
    @dataclass
    class Variable:
        name: str
        index: int
    
    variables = [
        Variable('pos_diff', 0),
        Variable('pos_ratio', 1),
        Variable('token_distance', 2)
    ]
    
    # Test expression
    import sympy as sp
    expr = sp.Symbol('pos_diff') * 0.1 + sp.Symbol('pos_ratio') * 0.05
    
    # Dummy model (not used in this test since we have pre-computed attention)
    class DummyModel:
        device = 'cpu'
    
    dummy_model = DummyModel()
    
    try:
        fidelity = calculator.calculate_fidelity(
            expression=expr,
            ai_model=dummy_model,
            test_data=test_data,
            variables=variables,
            target_behavior='attention'
        )
        print(f"✓ Fidelity calculation successful: {fidelity:.4f}")
    except Exception as e:
        print(f"✗ Fidelity calculation failed: {e}")