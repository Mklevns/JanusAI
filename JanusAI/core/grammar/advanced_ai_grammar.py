# File: JanusAI/core/grammar/enhanced_ai_grammar.py
"""
Enhanced AIGrammar Implementation
=================================

Unified implementation that combines the best features from all AIGrammar variants
specifically optimized for transformer attention pattern discovery.
"""

import numpy as np
import torch
import torch.nn.functional as F
import sympy as sp
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass

# Import base grammar
from JanusAI.core.grammar.base_grammar import ProgressiveGrammar


class AttentionPrimitives:
    """
    Advanced attention-specific operations with both symbolic and numerical modes.
    Optimized for GPT-2 attention pattern discovery.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.numerical_tolerance = 1e-8

    def attention_score(self, query: Any, key: Any, scale: Optional[float] = None) -> Any:
        """Compute attention scores Q·K^T with optional scaling."""
        if self._is_symbolic(query, key):
            return sp.Function('AttentionScore')(query, key)
        
        q = self._to_tensor(query)
        k = self._to_tensor(key)
        
        # Handle different tensor dimensions
        if q.dim() == 1: q = q.unsqueeze(0)
        if k.dim() == 1: k = k.unsqueeze(0)
        
        scores = q @ k.transpose(-2, -1)
        
        if scale is None:
            scale = 1.0 / np.sqrt(q.shape[-1])
        
        return (scores * scale).cpu().numpy()

    def softmax_attention(self, scores: Any, mask: Optional[Any] = None, temperature: float = 1.0) -> Any:
        """Apply softmax to attention scores with optional masking and temperature."""
        if self._is_symbolic(scores):
            return sp.Function('Softmax')(scores)
        
        t = self._to_tensor(scores) / temperature
        
        if mask is not None:
            m = self._to_tensor(mask, dtype=torch.bool)
            t = t.masked_fill(~m, float('-inf'))
        
        return F.softmax(t, dim=-1).cpu().numpy()

    def weighted_value(self, weights: Any, values: Any) -> Any:
        """Compute attention-weighted values."""
        if self._is_symbolic(weights, values):
            return sp.Function('WeightedSum')(weights, values)
        
        w = self._to_tensor(weights)
        v = self._to_tensor(values)
        
        # Handle different tensor shapes for attention
        if w.dim() == 2 and v.dim() == 2:
            out = w @ v
        else:
            out = w.unsqueeze(-1) * v
        
        return out.cpu().numpy()

    def position_encoding(self, positions: Any, d_model: int = 512, max_len: int = 1000) -> Any:
        """Generate sinusoidal position encodings."""
        if self._is_symbolic(positions):
            return sp.Function('PosEnc')(positions)
        
        pos = np.array(positions, dtype=int) if not isinstance(positions, np.ndarray) else positions
        encoding = np.zeros((len(pos), d_model), dtype=np.float32)
        
        for idx, p in enumerate(pos):
            for i in range(0, d_model, 2):
                angle = p / (max_len ** (2 * i / d_model))
                encoding[idx, i] = np.sin(angle)
                if i + 1 < d_model:
                    encoding[idx, i + 1] = np.cos(angle)
        
        return encoding

    def causal_mask(self, seq_len: int, mask_future: bool = True) -> np.ndarray:
        """Generate causal attention mask."""
        if mask_future:
            mask = np.tril(np.ones((seq_len, seq_len), dtype=bool), k=0)
        else:
            mask = np.ones((seq_len, seq_len), dtype=bool)
        return mask

    def attention_pattern_detector(self, weights: Any, pattern: str = 'previous_token') -> Any:
        """Detect specific attention patterns in weight matrices."""
        if self._is_symbolic(weights):
            return sp.Function(f'Pattern_{pattern}')(weights)
        
        arr = np.array(weights)
        pattern_mask = np.zeros_like(arr)
        
        if pattern == 'previous_token':
            # Focus on i-1 positions (previous token attention)
            for i in range(1, arr.shape[0]):
                pattern_mask[i, i-1] = 1.0
        elif pattern == 'copying':
            # Diagonal attention (token attending to itself)
            min_dim = min(arr.shape)
            for i in range(min_dim):
                pattern_mask[i, i] = 1.0
        elif pattern == 'induction':
            # A[B] ... A -> B pattern (simplified)
            for i in range(2, arr.shape[0]):
                pattern_mask[i, i-2] = 1.0
        elif pattern == 'bos':
            # Beginning of sequence attention
            pattern_mask[:, 0] = 1.0
        
        return arr * pattern_mask

    def relative_position_bias(self, i: Any, j: Any, max_distance: int = 128) -> Any:
        """Compute relative position bias between positions i and j."""
        if self._is_symbolic(i, j):
            return sp.Function('RelPosBias')(i, j)
        
        distance = np.clip(np.array(i) - np.array(j), -max_distance, max_distance)
        # Simple learnable bias (in practice this would be learned)
        bias = np.exp(-np.abs(distance) / 10.0)
        return bias

    def attention_entropy(self, weights: Any, epsilon: float = 1e-8) -> Any:
        """Calculate attention entropy for measuring concentration."""
        if self._is_symbolic(weights):
            return sp.Function('AttentionEntropy')(weights)
        
        arr = np.array(weights) + epsilon
        arr = arr / np.sum(arr, axis=-1, keepdims=True)  # Normalize
        entropy = -np.sum(arr * np.log(arr), axis=-1)
        return entropy

    def _to_tensor(self, data: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert input to tensor with proper device placement."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        
        dtype = dtype or torch.float32
        return torch.tensor(data, dtype=dtype, device=self.device)

    def _is_symbolic(self, *args) -> bool:
        """Check if any argument is symbolic."""
        return any(isinstance(a, (sp.Expr, sp.Symbol)) for a in args)


@dataclass
class AttentionVariable:
    """Extended Variable class for attention-specific features."""
    name: str
    index: int
    var_type: str = 'positional'  # 'positional', 'token', 'pattern', 'activation'
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    @property
    def symbolic(self) -> sp.Symbol:
        return sp.Symbol(self.name)
    
    def __hash__(self):
        return hash((self.name, self.index, self.var_type))


class EnhancedAIGrammar(ProgressiveGrammar):
    """
    Enhanced AI Grammar with advanced attention primitives and 
    optimized symbolic regression capabilities.
    """
    
    def __init__(self, base_grammar: Optional[ProgressiveGrammar] = None):
        super().__init__(load_defaults=True)
        self.base = base_grammar
        self.attn = AttentionPrimitives()
        
        # Organized primitive categories
        self.primitive_categories = {
            'attention': {},
            'position': {},
            'pattern': {},
            'activation': {},
            'custom': {}
        }
        
        self._register_attention_primitives()
        self._register_position_primitives()
        self._register_pattern_primitives()
        self._register_activation_primitives()
        
        # AI-specific operator arities
        self._ai_operator_arities = self._define_ai_operator_arities()

    def _register_attention_primitives(self):
        """Register core attention mechanisms."""
        attention_ops = {
            'attention_score': self.attn.attention_score,
            'softmax_attention': self.attn.softmax_attention,
            'weighted_value': self.attn.weighted_value,
            'attention_entropy': self.attn.attention_entropy,
        }
        self.primitive_categories['attention'].update(attention_ops)

    def _register_position_primitives(self):
        """Register positional encoding and masking operations."""
        position_ops = {
            'position_encoding': self.attn.position_encoding,
            'causal_mask': self.attn.causal_mask,
            'relative_position_bias': self.attn.relative_position_bias,
        }
        self.primitive_categories['position'].update(position_ops)

    def _register_pattern_primitives(self):
        """Register attention pattern detection primitives."""
        pattern_ops = {
            'previous_token': lambda w: self.attn.attention_pattern_detector(w, 'previous_token'),
            'copying_pattern': lambda w: self.attn.attention_pattern_detector(w, 'copying'),
            'induction_pattern': lambda w: self.attn.attention_pattern_detector(w, 'induction'),
            'bos_pattern': lambda w: self.attn.attention_pattern_detector(w, 'bos'),
        }
        self.primitive_categories['pattern'].update(pattern_ops)

    def _register_activation_primitives(self):
        """Register neural network activation functions."""
        activation_ops = {
            'relu': lambda x: np.maximum(0, x) if not self._is_symbolic(x) else sp.Function('ReLU')(x),
            'gelu': self._gelu_activation,
            'sigmoid': lambda x: 1/(1+np.exp(-x)) if not self._is_symbolic(x) else sp.Function('Sigmoid')(x),
            'tanh': lambda x: np.tanh(x) if not self._is_symbolic(x) else sp.Function('Tanh')(x),
        }
        self.primitive_categories['activation'].update(activation_ops)

    def _gelu_activation(self, x: Any) -> Any:
        """GELU activation function used in transformers."""
        if self._is_symbolic(x):
            return sp.Function('GELU')(x)
        
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        x_np = np.array(x)
        return 0.5 * x_np * (1 + np.tanh(np.sqrt(2/np.pi) * (x_np + 0.044715 * x_np**3)))

    def _define_ai_operator_arities(self) -> Dict[str, int]:
        """Define arities for all AI-specific operators."""
        return {
            # Attention operations
            'attention_score': 2,
            'softmax_attention': 1,
            'weighted_value': 2,
            'attention_entropy': 1,
            
            # Position operations
            'position_encoding': 1,
            'causal_mask': 1,
            'relative_position_bias': 2,
            
            # Pattern operations
            'previous_token': 1,
            'copying_pattern': 1,
            'induction_pattern': 1,
            'bos_pattern': 1,
            
            # Activation functions
            'relu': 1,
            'gelu': 1,
            'sigmoid': 1,
            'tanh': 1,
        }

    def get_arity(self, op_name: str) -> int:
        """Get the arity of an operator, including AI-specific ones."""
        if op_name in self._ai_operator_arities:
            return self._ai_operator_arities[op_name]
        
        # Fall back to parent class
        return super().get_arity(op_name)

    def get_primitive(self, name: str) -> Optional[Callable]:
        """Get a primitive function by name from any category."""
        for category in self.primitive_categories.values():
            if name in category:
                return category[name]
        return None

    def list_primitives(self) -> Dict[str, List[str]]:
        """List all available primitives by category."""
        return {
            category: list(ops.keys()) 
            for category, ops in self.primitive_categories.items()
        }

    def add_custom_primitive(self, name: str, func: Callable, category: str = 'custom'):
        """Add a custom primitive to the grammar."""
        if category not in self.primitive_categories:
            self.primitive_categories[category] = {}
        
        self.primitive_categories[category][name] = func
        
        # Add to appropriate primitive set for base grammar compatibility
        if 'custom_callable_ops' not in self.primitives:
            self.primitives['custom_callable_ops'] = set()
        self.primitives['custom_callable_ops'].add(name)

    def create_attention_variables(self, sequence_length: int) -> List[AttentionVariable]:
        """Create standard variables for attention pattern discovery."""
        variables = []
        
        # Positional variables
        variables.extend([
            AttentionVariable('pos_i', 0, 'positional', {'description': 'Query position'}),
            AttentionVariable('pos_j', 1, 'positional', {'description': 'Key position'}),
            AttentionVariable('pos_diff', 2, 'positional', {'description': 'i - j'}),
            AttentionVariable('pos_ratio', 3, 'positional', {'description': 'i / max(j, 1)'}),
            AttentionVariable('relative_pos', 4, 'positional', {'description': 'abs(i - j)'}),
        ])
        
        # Pattern indicator variables
        variables.extend([
            AttentionVariable('is_previous', 5, 'pattern', {'description': '1 if j = i-1 else 0'}),
            AttentionVariable('is_diagonal', 6, 'pattern', {'description': '1 if i = j else 0'}),
            AttentionVariable('is_bos', 7, 'pattern', {'description': '1 if j = 0 else 0'}),
        ])
        
        # Token type variables (to be filled based on actual tokens)
        variables.extend([
            AttentionVariable('token_type_i', 8, 'token', {'description': 'Token type at position i'}),
            AttentionVariable('token_type_j', 9, 'token', {'description': 'Token type at position j'}),
        ])
        
        return variables

    def _is_symbolic(self, *args) -> bool:
        """Check if any argument is symbolic."""
        return any(isinstance(a, (sp.Expr, sp.Symbol)) for a in args)

    def validate_expression_for_attention(self, expression: Any) -> bool:
        """Validate that an expression is suitable for attention pattern modeling."""
        try:
            # Check if expression has reasonable complexity
            complexity = getattr(expression, 'complexity', len(str(expression)))
            if complexity > 50:
                return False
            
            # Check if expression uses appropriate variables
            if hasattr(expression, 'variables'):
                var_names = {v.name for v in expression.variables}
                required_vars = {'pos_i', 'pos_j', 'pos_diff'}
                if not any(var in var_names for var in required_vars):
                    return False
            
            return True
        except:
            return False


# Testing and validation functions
def test_enhanced_ai_grammar():
    """Test the enhanced AI grammar functionality."""
    print("Testing Enhanced AI Grammar...")
    
    grammar = EnhancedAIGrammar()
    
    # Test primitive registration
    primitives = grammar.list_primitives()
    print(f"Registered primitives: {primitives}")
    
    # Test attention variables creation
    variables = grammar.create_attention_variables(sequence_length=64)
    print(f"Created {len(variables)} attention variables")
    
    # Test symbolic operations
    q, k, v = sp.symbols('q k v')
    print(f"Attention score (symbolic): {grammar.attn.attention_score(q, k)}")
    
    # Test numerical operations
    Q = np.random.randn(4, 8)
    K = np.random.randn(4, 8)
    V = np.random.randn(4, 8)
    
    scores = grammar.attn.attention_score(Q, K)
    weights = grammar.attn.softmax_attention(scores)
    output = grammar.attn.weighted_value(weights, V)
    
    print(f"Numerical attention shapes: scores={scores.shape}, weights={weights.shape}, output={output.shape}")
    
    # Test pattern detection
    pattern_weights = grammar.attn.attention_pattern_detector(weights, 'previous_token')
    print(f"Previous token pattern detected: {np.sum(pattern_weights) > 0}")
    
    print("Enhanced AI Grammar test completed successfully!")


if __name__ == "__main__":
    test_enhanced_ai_grammar()