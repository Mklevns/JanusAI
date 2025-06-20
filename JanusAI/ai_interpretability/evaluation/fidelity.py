# janus/ai_interpretability/evaluation/fidelity.py
"""
Complete implementation of the _calculate_fidelity method for AI interpretability.

This is the core function that measures how well our symbolic expressions
capture the behavior of attention heads in transformers.
"""

import numpy as np
import torch
import sympy as sp
from typing import Any, Dict, Optional, Union, Tuple
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
            # For attention heads, extract attention weights
            return self._extract_attention_patterns(ai_model, test_data)
        elif target_behavior == 'output':
            # For output prediction
            return self._extract_output_logits(ai_model, test_data)
        else:
            raise ValueError(f"Unknown target behavior: {target_behavior}")
    
    def _extract_attention_patterns(self, 
                                   ai_model: Any, 
                                   test_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract attention patterns from the transformer model."""
        
        # Ensure model is in eval mode
        ai_model.eval()
        
        attention_patterns = []
        
        with torch.no_grad():
            # Get input tokens
            input_ids = torch.tensor(test_data.get('input_ids', test_data.get('inputs')))
            
            # Create attention mask if needed
            attention_mask = torch.ones_like(input_ids) if 'attention_mask' not in test_data else torch.tensor(test_data['attention_mask'])
            
            # Forward pass with attention outputs
            outputs = ai_model(input_ids=input_ids, 
                              attention_mask=attention_mask, 
                              output_attentions=True)
            
            # Extract attention from specific layer/head (from test_data metadata)
            layer_idx = test_data.get('target_layer', 0)
            head_idx = test_data.get('target_head', None)
            
            layer_attention = outputs.attentions[layer_idx]  # (batch, heads, seq, seq)
            
            if head_idx is not None:
                # Specific head
                head_attention = layer_attention[:, head_idx, :, :]  # (batch, seq, seq)
            else:
                # Average across heads
                head_attention = layer_attention.mean(dim=1)  # (batch, seq, seq)
            
            # Flatten attention matrices for correlation analysis
            attention_patterns = head_attention.flatten().cpu().numpy()
        
        return attention_patterns
    
    def _evaluate_expression(self, 
                           expression: Any,
                           test_data: Dict[str, np.ndarray],
                           variables: list) -> np.ndarray:
        """Evaluate symbolic expression on test data."""
        
        if expression is None:
            return np.zeros(1)
        
        try:
            # Handle different expression types
            if hasattr(expression, 'symbolic'):
                symbolic_expr = expression.symbolic
            elif isinstance(expression, sp.Expr):
                symbolic_expr = expression
            else:
                # Try to convert string to sympy expression
                symbolic_expr = sp.sympify(str(expression))
            
            # Create variable substitutions
            var_subs = self._create_variable_substitutions(test_data, variables)
            
            # For attention patterns, create pairwise feature combinations
            if 'attention' in str(symbolic_expr).lower():
                return self._evaluate_attention_expression(symbolic_expr, test_data, var_subs)
            else:
                return self._evaluate_standard_expression(symbolic_expr, var_subs)
                
        except Exception as e:
            print(f"Expression evaluation failed: {e}")
            return np.zeros(1)
    
    def _create_variable_substitutions(self, 
                                     test_data: Dict[str, np.ndarray],
                                     variables: list) -> Dict[sp.Symbol, np.ndarray]:
        """Create substitutions for symbolic variables."""
        
        var_subs = {}
        
        for var in variables:
            var_symbol = sp.Symbol(var.name)
            
            if var.name == 'pos_diff':
                # Position differences for attention
                var_subs[var_symbol] = self._compute_position_differences(test_data)
            elif var.name == 'pos_ratio':
                # Position ratios
                var_subs[var_symbol] = self._compute_position_ratios(test_data)
            elif var.name == 'token_type_i' or var.name == 'token_type_j':
                # Token type information
                var_subs[var_symbol] = test_data.get('token_types', np.ones(100))
            elif var.name == 'relative_pos':
                # Relative position encoding
                var_subs[var_symbol] = self._compute_relative_positions(test_data)
            elif var.name in test_data:
                # Direct data mapping
                var_subs[var_symbol] = test_data[var.name]
            else:
                # Default to small random values
                var_subs[var_symbol] = np.random.randn(100) * 0.1
        
        return var_subs
    
    def _compute_position_differences(self, test_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute position differences for attention pattern analysis."""
        
        seq_len = test_data.get('sequence_length', 32)
        
        # Create all pairwise position differences
        positions = np.arange(seq_len)
        pos_i, pos_j = np.meshgrid(positions, positions, indexing='ij')
        pos_diff = pos_i - pos_j  # (seq_len, seq_len)
        
        return pos_diff.flatten()
    
    def _compute_position_ratios(self, test_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute position ratios, handling division by zero."""
        
        seq_len = test_data.get('sequence_length', 32)
        positions = np.arange(1, seq_len + 1)  # Start from 1 to avoid division by zero
        
        pos_i, pos_j = np.meshgrid(positions, positions, indexing='ij')
        pos_ratio = np.divide(pos_i, pos_j, out=np.ones_like(pos_i, dtype=float), where=pos_j!=0)
        
        return pos_ratio.flatten()
    
    def _compute_relative_positions(self, test_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute relative position encodings (simplified)."""
        
        seq_len = test_data.get('sequence_length', 32)
        positions = np.arange(seq_len)
        
        # Simple relative position encoding
        pos_i, pos_j = np.meshgrid(positions, positions, indexing='ij')
        rel_pos = np.abs(pos_i - pos_j) / seq_len  # Normalized relative distance
        
        return rel_pos.flatten()
    
    def _evaluate_attention_expression(self, 
                                     symbolic_expr: sp.Expr,
                                     test_data: Dict[str, np.ndarray],
                                     var_subs: Dict[sp.Symbol, np.ndarray]) -> np.ndarray:
        """Evaluate expressions specifically for attention patterns."""
        
        try:
            # Convert to numerical function
            symbols = list(var_subs.keys())
            values = list(var_subs.values())
            
            # Ensure all arrays have the same length
            min_length = min(len(v) for v in values)
            values = [v[:min_length] for v in values]
            
            # Use lambdify for fast numerical evaluation
            func = sp.lambdify(symbols, symbolic_expr, modules=['numpy'])
            result = func(*values)
            
            # Handle scalar results
            if np.isscalar(result):
                result = np.full(min_length, result)
            elif hasattr(result, '__len__') and len(result) != min_length:
                # Broadcast or truncate as needed
                result = np.broadcast_to(result, min_length)
            
            return np.array(result).flatten()
            
        except Exception as e:
            print(f"Attention expression evaluation failed: {e}")
            return np.zeros(min_length if 'min_length' in locals() else 100)
    
    def _evaluate_standard_expression(self, 
                                    symbolic_expr: sp.Expr,
                                    var_subs: Dict[sp.Symbol, np.ndarray]) -> np.ndarray:
        """Evaluate standard symbolic expressions."""
        
        try:
            # Substitute values and evaluate
            result = symbolic_expr.subs(var_subs)
            
            if hasattr(result, 'evalf'):
                result = float(result.evalf())
                return np.array([result])
            else:
                return np.array([float(result)])
                
        except Exception as e:
            print(f"Standard expression evaluation failed: {e}")
            return np.zeros(1)
    
    def _compute_correlation_fidelity(self, 
                                    ground_truth: np.ndarray,
                                    predicted: np.ndarray) -> float:
        """Compute correlation-based fidelity score."""
        
        # Handle shape mismatches
        min_length = min(len(ground_truth), len(predicted))
        if min_length == 0:
            return 0.0
        
        gt_trimmed = ground_truth[:min_length].flatten()
        pred_trimmed = predicted[:min_length].flatten()
        
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
            pearson_r, pearson_p = pearsonr(gt_clean, pred_clean)
            pearson_score = abs(pearson_r) if not np.isnan(pearson_r) else 0.0
            
            # Spearman correlation (monotonic relationships)
            spearman_r, spearman_p = spearmanr(gt_clean, pred_clean)
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


# Integration function for InterpretabilityReward class
def integrate_fidelity_calculator():
    """
    Function to integrate the FidelityCalculator into the existing 
    InterpretabilityReward class.
    """
    
    def _calculate_fidelity(self, expression: Any, ai_model: Any, test_data: Any) -> float:
        """
        Updated _calculate_fidelity method for InterpretabilityReward class.
        
        This replaces the placeholder implementation with a robust fidelity calculator.
        """
        
        if not hasattr(self, '_fidelity_calculator'):
            self._fidelity_calculator = FidelityCalculator()
        
        # Convert test_data to expected format
        if hasattr(test_data, 'inputs') and hasattr(test_data, 'outputs'):
            # AIBehaviorData format
            data_dict = {
                'input_ids': test_data.inputs,
                'outputs': test_data.outputs,
                'attention_weights': getattr(test_data, 'attention_weights', None),
                'sequence_length': test_data.inputs.shape[-1] if hasattr(test_data.inputs, 'shape') else 32
            }
        else:
            # Direct dictionary format
            data_dict = test_data
        
        # Use variables from self if available
        variables = getattr(self, 'variables', [])
        
        return self._fidelity_calculator.calculate_fidelity(
            expression=expression,
            ai_model=ai_model,
            test_data=data_dict,
            variables=variables,
            target_behavior='attention'
        )
    
    return _calculate_fidelity


# Example usage and testing
if __name__ == "__main__":
    # Test the fidelity calculator with dummy data
    calculator = FidelityCalculator()
    
    # Create dummy test data
    test_data = {
        'input_ids': np.random.randint(0, 1000, (2, 16)),
        'sequence_length': 16,
        'target_layer': 0,
        'target_head': 1
    }
    
    # Create dummy variables
    from dataclasses import dataclass
    
    @dataclass
    class DummyVariable:
        name: str
        index: int
        properties: dict
    
    variables = [
        DummyVariable('pos_diff', 0, {}),
        DummyVariable('pos_ratio', 1, {}),
    ]
    
    # Test expression evaluation
    expr = sp.Symbol('pos_diff') * 0.1 + sp.Symbol('pos_ratio') * 0.05
    
    try:
        predicted = calculator._evaluate_expression(expr, test_data, variables)
        print(f"✓ Expression evaluation successful: {predicted.shape}")
    except Exception as e:
        print(f"✗ Expression evaluation failed: {e}")
    
    # Test variable substitutions
    try:
        var_subs = calculator._create_variable_substitutions(test_data, variables)
        print(f"✓ Variable substitutions created: {list(var_subs.keys())}")
    except Exception as e:
        print(f"✗ Variable substitution failed: {e}")