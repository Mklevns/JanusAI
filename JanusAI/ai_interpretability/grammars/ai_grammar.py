import numpy as np
import sympy as sp
from typing import List, Dict, Any, Callable

from janus_ai.core.grammar.base_grammar import ProgressiveGrammar, Primitive
from janus_ai.core.expressions.expression import Expression, Variable

# Helper function for softmax, assuming it might be used by Attention node
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class AIGrammar(ProgressiveGrammar):
    """
    Grammar for AI components, particularly focusing on attention mechanisms
    and related neural network operations.
    """
    def __init__(self,
                 primitives: Dict[str, Dict[str, Primitive]] = None,
                 variables: List[Variable] = None,
                 constants: Dict[str, float] = None,
                 max_depth: int = 5,
                 max_complexity: int = 50,
                 config: Dict[str, Any] = None):

        super().__init__(
            primitives=primitives if primitives is not None else self._get_default_primitives(),
            variables=variables,
            constants=constants,
            max_depth=max_depth,
            max_complexity=max_complexity,
            config=config
        )

    def _get_default_primitives(self) -> Dict[str, Dict[str, Primitive]]:
        """
        Defines the default set of primitives for AIGrammar, including
        attention-related operations.
        """
        primitives = {
            "binary_ops": {
                "add": Primitive("add", lambda x, y: x + y, arity=2, complexity=1, sympy_func=sp.Add),
                "mul": Primitive("mul", lambda x, y: x * y, arity=2, complexity=1, sympy_func=sp.Mul),
                "matmul": Primitive("matmul", lambda q, k: np.matmul(q, k), arity=2, complexity=5), # Q·K^T part of attention
            },
            "unary_ops": {
                "softmax_op": Primitive("softmax_op", lambda x: softmax(x), arity=1, complexity=3), # Softmax for attention
                # LinearProjection might be represented as a primitive that takes an input and applies a learned matrix
                # However, "learned projections" usually mean the weights are part of the model being discovered/optimized,
                # not fixed operations. For symbolic discovery, a LinearProjection node might need special handling
                # or be a placeholder for a matrix that gets optimized externally.
                # For now, a simple placeholder or a more complex node type might be needed.
                "linear_proj": Primitive("linear_proj", lambda x, w: np.matmul(x, w), arity=2, complexity=5, name="LinearProjection"), # x * W
            },
            "attention_ops": {
                # Q·K^T, then softmax, then multiply by V.
                # This could be a single complex primitive or built from smaller ones.
                # The problem statement asks for an "Attention node that computes Q·K^T".
                # This might mean Q·K^T is one part, and then it's combined.
                # Let's assume 'matmul' covers Q·K^T for now.
                # A full 'Attention' might look like: Attention(Q, K, V) -> softmax(Q @ K.T) @ V
                "attention_qk": Primitive(
                    "attention_qk",
                    lambda q, k: np.matmul(q, np.transpose(k, axes=(-1, -2) if k.ndim > 1 else (0,1))), # Q @ K.T
                    arity=2,
                    complexity=6,
                    name="AttentionQK"
                ),
                "attention_full": Primitive(
                    "attention_full",
                    lambda q, k, v: np.matmul(softmax(np.matmul(q, np.transpose(k, axes=(-1, -2) if k.ndim > 1 else (0,1)))), v),
                    arity=3,
                    complexity=10,
                    name="FullAttention"
                ),
            },
            "positional_encodings": {
                # Placeholder for Sinusoidal Positional Encoding.
                # This would typically take sequence length and embedding dimension.
                # For symbolic discovery, it might be a generator or a fixed matrix.
                "sin_pos_enc": Primitive(
                    "sin_pos_enc",
                    lambda seq_len, d_model: self._sinusoidal_positional_encoding(seq_len, d_model),
                    arity=2, # seq_len, d_model
                    complexity=4,
                    name="SinusoidalPositionalEncoding"
                ),
                # Placeholder for Learned Positional Embeddings
                # This would be a learnable parameter matrix (e.g., an embedding layer)
                # In symbolic context, it might be a special variable or placeholder.
                "learned_pos_emb": Primitive(
                    "learned_pos_emb",
                    lambda idx, table: table[idx], # Simplified: lookup from a table
                    arity=2, # index, embedding_table
                    complexity=3,
                    name="LearnedPositionalEmbedding"
                )
            },
            "other": {
                # Potentially other useful NN components
            }
        }
        # Ensure all categories are present, even if empty, for consistency
        for op_type in ['unary_ops', 'binary_ops', 'calculus_ops', 'logical_ops', 'comparison_ops', 'constants', 'variables_terminals']:
            primitives.setdefault(op_type, {})
        return primitives

    def _sinusoidal_positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """
        Generates sinusoidal positional encodings.
        Args:
            seq_len: Length of the sequence.
            d_model: Dimension of the model/embeddings.
        Returns:
            np.ndarray of shape (seq_len, d_model)
        """
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def add_primitive(self, name: str, func: Callable, arity: int, complexity: int, category: str, sympy_func: Any = None):
        """Adds a new primitive to the grammar."""
        if category not in self.primitives:
            self.primitives[category] = {}
        self.primitives[category][name] = Primitive(name, func, arity, complexity, sympy_func)

    # Example of how LinearProjection might be handled if weights are part of the expression:
    # It could be a node that holds its own weights, or references a global weight store.
    # For discovery, this is complex. A simpler approach is that 'LinearProjection'
    # is an operator that applies to an input and a 'WeightMatrix' variable.

    # The 'Attention' node computing Q·K^T is covered by 'attention_qk'.
    # 'LinearProjection' is covered by 'linear_proj' (takes input and weight matrix).
    # 'PositionalEncoding' primitives are 'sin_pos_enc' and 'learned_pos_emb'.

    def get_expression(self, max_depth: Optional[int] = None) -> Expression:
        # This would be inherited from ProgressiveGrammar or BaseGrammar
        # and used to generate expressions using the defined primitives.
        return super().get_expression(max_depth=max_depth)

if __name__ == '__main__':
    # Example Usage
    # Define some variables that might represent Q, K, V, or other inputs
    var_q = Variable("Q", 0, properties={"shape": (10, 64)}) # Batch_size=1, Seq_len=10, Dim=64
    var_k = Variable("K", 1, properties={"shape": (10, 64)})
    var_v = Variable("V", 2, properties={"shape": (10, 64)})
    var_w1 = Variable("W1", 3, properties={"shape": (64, 64)}) # Weight matrix for a linear projection
    var_seq_len = Variable("seq_len", 4, properties={"is_parameter": True}) # Parameter for positional encoding
    var_d_model = Variable("d_model", 5, properties={"is_parameter": True}) # Parameter for positional encoding
    var_pos_table = Variable("pos_table", 6, properties={"is_parameter": True, "shape": (20, 64)}) # Learned PE table

    variables = [var_q, var_k, var_v, var_w1, var_seq_len, var_d_model, var_pos_table]

    # Initialize AIGrammar
    ai_grammar = AIGrammar(variables=variables)
    print("AIGrammar initialized with primitives:")
    for category, prims in ai_grammar.primitives.items():
        print(f"  {category}:")
        for name, p_obj in prims.items():
            print(f"    {name} (arity {p_obj.arity}, complexity {p_obj.complexity})")

    # Test specific primitives (conceptual evaluation)
    # Note: Actual evaluation requires data and an evaluation engine.

    # Attention QK^T
    # expr_qk = Expression("attention_qk", [var_q, var_k])
    # print(f"\nSymbolic QK^T: {expr_qk}")
    # This would need actual numpy arrays for Q and K to evaluate the lambda

    # Linear Projection
    # expr_linear = Expression("linear_proj", [var_q, var_w1]) # Project Q using W1
    # print(f"Symbolic Linear Projection: {expr_linear}")

    # Sinusoidal Positional Encoding
    # This primitive generates the encoding matrix itself.
    # To use it, one might need to create constant nodes for seq_len and d_model
    # or have them as special kinds of variables/inputs.
    # expr_sin_pe = Expression("sin_pos_enc", [Expression("const", [10]), Expression("const", [64])]) # seq_len=10, d_model=64
    # print(f"Symbolic Sinusoidal PE: {expr_sin_pe}")
    # pe_matrix = ai_grammar.primitives['positional_encodings']['sin_pos_enc'].func(10, 64)
    # print(f"Generated Sinusoidal PE matrix shape: {pe_matrix.shape}")

    # Full Attention
    # expr_full_attn = Expression("attention_full", [var_q, var_k, var_v])
    # print(f"Symbolic Full Attention: {expr_full_attn}")

    # TODO: Add __init__.py to JanusAI/ai_interpretability/grammars/ if it doesn't exist
    # to make AIGrammar importable.
