# JanusAI/core/grammar/ai_grammar.py
"""
Unified AIGrammar implementation combining all scattered components
into a single, coherent attention-aware symbolic regression grammar.
"""

import numpy as np
import torch
import torch.nn.functional as F
import sympy as sp
from typing import Any, Dict, List, Optional, Callable, Union
import logging

from .base_grammar import ProgressiveGrammar, Expression, Variable


class AttentionPrimitives:
    """
    Attention-specific operations usable in both symbolic and numerical modes.
    This is the core engine for attention pattern discovery.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.numerical_tolerance = 1e-8

    def attention_score(self, query: Any, key: Any, scale: Optional[float] = None) -> Any:
        """Compute attention scores: Q @ K^T / sqrt(d_k)"""
        if self._is_symbolic(query, key):
            if scale is not None:
                return sp.Function('AttentionScore')(query, key) / scale
            return sp.Function('AttentionScore')(query, key)
        
        # Numerical computation
        q = self._to_tensor(query)
        k = self._to_tensor(key)
        
        if q.dim() == 1: q = q.unsqueeze(0)
        if k.dim() == 1: k = k.unsqueeze(0)
        
        scores = q @ k.transpose(-2, -1)
        if scale is None:
            scale = 1.0 / np.sqrt(q.shape[-1])
        
        return (scores * scale).cpu().numpy()

    def softmax_attention(self, scores: Any, mask: Optional[Any] = None) -> Any:
        """Apply softmax to attention scores with optional masking"""
        if self._is_symbolic(scores):
            return sp.Function('Softmax')(scores)
        
        t = self._to_tensor(scores)
        if mask is not None:
            m = self._to_tensor(mask, dtype=torch.bool)
            t = t.masked_fill(~m, float('-inf'))
        
        return F.softmax(t, dim=-1).cpu().numpy()

    def weighted_value(self, weights: Any, values: Any) -> Any:
        """Compute weighted combination: W @ V"""
        if self._is_symbolic(weights, values):
            return sp.Function('WeightedSum')(weights, values)
        
        w = self._to_tensor(weights)
        v = self._to_tensor(values)
        
        if w.dim() == 2 and v.dim() == 2:
            result = w @ v
        else:
            result = w.unsqueeze(-1) * v
        
        return result.cpu().numpy()

    def position_encoding(self, positions: Any, d_model: int = 512) -> Any:
        """Compute sinusoidal position encodings"""
        if self._is_symbolic(positions):
            return sp.Function('PosEnc')(positions)
        
        pos = np.array(positions, dtype=int)
        enc = np.zeros((len(pos), d_model), dtype=float)
        
        for idx, p in enumerate(pos):
            for i in range(0, d_model, 2):
                angle = p / (10000 ** (2 * i / d_model))
                enc[idx, i] = np.sin(angle)
                if i + 1 < d_model:
                    enc[idx, i + 1] = np.cos(angle)
        
        return enc

    def causal_mask(self, seq_len: int) -> np.ndarray:
        """Generate causal (lower triangular) mask"""
        return np.tril(np.ones((seq_len, seq_len), dtype=bool), k=0)

    def attention_pattern_detector(self, weights: Any, pattern: str = 'previous_token') -> Any:
        """Detect specific attention patterns"""
        if self._is_symbolic(weights):
            return sp.Function(f'Pattern_{pattern}')(weights)
        
        arr = np.array(weights)
        
        if pattern == 'previous_token':
            # Extract previous token attention pattern
            result = np.zeros_like(arr)
            for i in range(1, arr.shape[0]):
                result[i, i-1] = arr[i, i-1]
            return result
        
        elif pattern == 'copying':
            # Extract diagonal (self-attention) pattern
            result = np.zeros_like(arr)
            min_dim = min(arr.shape)
            for i in range(min_dim):
                result[i, i] = arr[i, i]
            return result
        
        elif pattern == 'positional_bias':
            # Extract position-based bias
            seq_len = arr.shape[0]
            bias = np.zeros_like(arr)
            for i in range(seq_len):
                for j in range(seq_len):
                    bias[i, j] = 1.0 / (1.0 + abs(i - j))
            return arr * bias
        
        else:
            return np.zeros_like(arr)

    def _is_symbolic(self, *args) -> bool:
        """Check if any argument is symbolic"""
        return any(isinstance(a, (sp.Expr, sp.Symbol)) for a in args)

    def _to_tensor(self, data: Any, dtype=torch.float32) -> torch.Tensor:
        """Convert data to tensor on the appropriate device"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, np.ndarray)):
            return torch.tensor(data, dtype=dtype, device=self.device)
        else:
            return torch.tensor([data], dtype=dtype, device=self.device)


class AIGrammar(ProgressiveGrammar):
    """
    Enhanced grammar for AI interpretability with complete attention primitive support.
    This is the unified implementation that consolidates all previous versions.
    """

    def __init__(self, load_defaults: bool = True):
        super().__init__(load_defaults=load_defaults)
        self.attention_primitives = AttentionPrimitives()
        self._init_ai_primitives()
        self._ai_operator_arities = self._define_ai_operator_arities()
        
        # Custom converters for SymPy integration
        self._custom_converters = {}

    def _init_ai_primitives(self):
        """Initialize AI-specific primitives and operators"""
        
        # Attention operations
        attention_ops = {
            'attention_score': self.attention_primitives.attention_score,
            'softmax_attention': self.attention_primitives.softmax_attention,
            'weighted_value': self.attention_primitives.weighted_value,
            'position_encoding': self.attention_primitives.position_encoding,
            'causal_mask': self.attention_primitives.causal_mask,
        }
        
        # Pattern detection operations
        pattern_ops = {
            'previous_token': lambda w: self.attention_primitives.attention_pattern_detector(w, 'previous_token'),
            'copying_pattern': lambda w: self.attention_primitives.attention_pattern_detector(w, 'copying'),
            'positional_bias': lambda w: self.attention_primitives.attention_pattern_detector(w, 'positional_bias'),
        }
        
        # Neural network operations
        nn_ops = {
            'relu': lambda x: np.maximum(0, x) if not self.attention_primitives._is_symbolic(x) else sp.Max(x, 0),
            'sigmoid': lambda x: 1/(1+np.exp(-np.clip(x, -500, 500))) if not self.attention_primitives._is_symbolic(x) else 1/(1+sp.exp(-x)),
            'tanh': lambda x: np.tanh(x) if not self.attention_primitives._is_symbolic(x) else sp.tanh(x),
            'gelu': self._gelu_function,
            'layer_norm': self._layer_norm_function,
        }
        
        # Logic operations
        logic_ops = {
            'if_then_else': lambda cond, true_val, false_val: true_val if cond else false_val,
            'threshold': lambda x, t: x > t,
            'weighted_sum': lambda weights, values: sum(w*v for w,v in zip(weights, values)),
            'max_pool': lambda values: max(values) if values else None,
        }
        
        # Add all operations to primitives
        if 'custom_callable_ops' not in self.primitives:
            self.primitives['custom_callable_ops'] = {}
        
        self.primitives['custom_callable_ops'].update(attention_ops)
        self.primitives['custom_callable_ops'].update(pattern_ops)
        self.primitives['custom_callable_ops'].update(nn_ops)
        self.primitives['custom_callable_ops'].update(logic_ops)
        
        # Add activation functions as unary operators too
        activation_names = {'relu', 'sigmoid', 'tanh', 'gelu'}
        if 'unary_ops' not in self.primitives:
            self.primitives['unary_ops'] = set()
        self.primitives['unary_ops'].update(activation_names)
        
        # Add custom sets for organization
        if 'custom_sets' not in self.primitives:
            self.primitives['custom_sets'] = {}
        
        self.primitives['custom_sets']['attention_types'] = list(attention_ops.keys())
        self.primitives['custom_sets']['pattern_types'] = list(pattern_ops.keys())
        self.primitives['custom_sets']['activation_types'] = list(activation_names)

    def _define_ai_operator_arities(self) -> Dict[str, int]:
        """Define the arity (number of arguments) for AI-specific operators"""
        return {
            # Attention operations
            'attention_score': 2,       # query, key
            'softmax_attention': 1,     # scores (mask is optional)
            'weighted_value': 2,        # weights, values
            'position_encoding': 1,     # positions
            'causal_mask': 1,          # sequence_length
            
            # Pattern operations
            'previous_token': 1,
            'copying_pattern': 1,
            'positional_bias': 1,
            
            # Neural operations
            'relu': 1,
            'sigmoid': 1,
            'tanh': 1,
            'gelu': 1,
            'layer_norm': 1,
            
            # Logic operations
            'if_then_else': 3,
            'threshold': 2,
            'weighted_sum': 2,
            'max_pool': 1,
        }

    def _gelu_function(self, x: Any) -> Any:
        """GELU activation function"""
        if self.attention_primitives._is_symbolic(x):
            return sp.Function('GELU')(x)
        x = np.array(x)
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def _layer_norm_function(self, x: Any) -> Any:
        """Layer normalization function"""
        if self.attention_primitives._is_symbolic(x):
            return sp.Function('LayerNorm')(x)
        x = np.array(x)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5)

    def get_arity(self, op_name: str) -> int:
        """Get the arity of a given operator, including AI-specific operators"""
        if op_name in self._ai_operator_arities:
            return self._ai_operator_arities[op_name]
        
        # Fall back to parent class
        return super().get_arity(op_name)

    def _validate_expression(self, operator: str, operands: List[Any]) -> bool:
        """Enhanced validation for AI-specific operators"""
        
        # First try standard validation
        try:
            if super()._validate_expression(operator, operands):
                return True
        except:
            pass
        
        # AI-specific validation
        if operator not in self._ai_operator_arities:
            return False
        
        expected_arity = self._ai_operator_arities[operator]
        if len(operands) != expected_arity:
            logging.debug(f"AIGrammar validation: Arity mismatch for {operator}. "
                         f"Expected {expected_arity}, got {len(operands)}")
            return False
        
        # Type-specific checks
        if operator in ['attention_score', 'weighted_value']:
            return all(self._is_tensor_compatible(op) for op in operands)
        
        return True

    def _is_tensor_compatible(self, operand: Any) -> bool:
        """Check if operand is compatible with tensor operations"""
        return (isinstance(operand, (np.ndarray, torch.Tensor, list)) or 
                isinstance(operand, (sp.Expr, sp.Symbol)) or
                isinstance(operand, (int, float)))

    def convert_ai_expression_to_sympy(self, expression: Expression) -> sp.Expr:
        """Convert AI-specific expressions to SymPy format"""
        if expression.operator in self._custom_converters:
            return self._custom_converters[expression.operator](expression)
        
        # Default conversion for AI operators
        if expression.operator in self.primitives.get('custom_callable_ops', {}):
            operand_symbols = []
            for i, operand in enumerate(expression.operands):
                if isinstance(operand, Expression):
                    operand_symbols.append(self.convert_ai_expression_to_sympy(operand))
                elif isinstance(operand, Variable):
                    operand_symbols.append(sp.Symbol(operand.name))
                else:
                    operand_symbols.append(operand)
            
            return sp.Function(expression.operator)(*operand_symbols)
        
        # Fall back to standard conversion
        raise NotImplementedError(f"No SymPy converter for {expression.operator}")

    def register_custom_operator(self, name: str, function: Callable, 
                                arity: int, sympy_converter: Optional[Callable] = None):
        """Register a custom operator with the grammar"""
        self.primitives['custom_callable_ops'][name] = function
        self._ai_operator_arities[name] = arity
        
        if sympy_converter:
            self._custom_converters[name] = sympy_converter

    def list_attention_primitives(self) -> Dict[str, List[str]]:
        """List all available attention-related primitives"""
        return {
            'attention_ops': self.primitives['custom_sets']['attention_types'],
            'pattern_ops': self.primitives['custom_sets']['pattern_types'],
            'activation_ops': self.primitives['custom_sets']['activation_types'],
        }


# Integration helper for backward compatibility
def create_enhanced_ai_grammar() -> AIGrammar:
    """Factory function to create a fully configured AIGrammar"""
    grammar = AIGrammar(load_defaults=True)
    
    # Add any additional configuration here
    logging.info(f"Created AIGrammar with {len(grammar.primitives['custom_callable_ops'])} AI operators")
    
    return grammar


if __name__ == "__main__":
    # Test the unified implementation
    grammar = create_enhanced_ai_grammar()
    
    print("Available attention primitives:")
    for category, ops in grammar.list_attention_primitives().items():
        print(f"  {category}: {ops}")
    
    # Test symbolic and numerical modes
    print("\nTesting attention primitives:")
    
    # Symbolic test
    q, k, v = sp.symbols('q k v')
    symbolic_score = grammar.attention_primitives.attention_score(q, k)
    print(f"Symbolic attention score: {symbolic_score}")
    
    # Numerical test
    Q = np.random.randn(2, 4)
    K = np.random.randn(2, 4)
    V = np.random.randn(2, 4)
    
    scores = grammar.attention_primitives.attention_score(Q, K)
    weights = grammar.attention_primitives.softmax_attention(scores)
    output = grammar.attention_primitives.weighted_value(weights, V)
    
    print(f"Numerical shapes - Scores: {scores.shape}, Weights: {weights.shape}, Output: {output.shape}")
    
    # Test pattern detection
    attention_matrix = np.random.rand(5, 5)
    prev_token_pattern = grammar.attention_primitives.attention_pattern_detector(
        attention_matrix, 'previous_token'
    )
    print(f"Previous token pattern detected: {np.sum(prev_token_pattern) > 0}")