from typing import Any, List, Dict # Added Dict for primitives
import sympy as sp # Assuming sp is sympy based on later methods

# Placeholder for a potential base class
class BaseGrammar:
    def __init__(self, primitives: Dict[str, Any] = None):
        # Initialize primitives if not provided
        self.primitives = primitives if primitives is not None else {
            'custom_values': {},
            'named_lists': {},
            'custom_sets': {},
            'custom_callable_ops': {}
        }

    def get_arity(self, op_name: str) -> int:
        # This would be implemented in the actual base class
        # For placeholder, raise ValueError if not a known basic op
        standard_ops = {'add': 2, 'subtract': 2, 'multiply': 2, 'divide': 2}
        if op_name in standard_ops:
            return standard_ops[op_name]
        raise ValueError(f"Unknown operator or function: '{op_name}' in BaseGrammar")

    def _validate_expression(self, operator: str, operands: List[Any]) -> bool:
        # Base validation logic
        # For this placeholder, let's assume it checks arity for some common ops
        try:
            expected_arity = self.get_arity(operator)
            return len(operands) == expected_arity
        except ValueError:
            return False # Operator not known to base

class AIGrammar(BaseGrammar):
    def __init__(self, primitives: Dict[str, Any] = None):
        super().__init__(primitives)
        # AI specific initializations can go here
        self._ai_op_arities = {
            'attention': 3, 'embedding_lookup': 2, 'if_then_else': 3,
            'threshold': 2, 'weighted_sum': 2, 'max_pool': 1,
            'relu': 1, 'sigmoid': 1, 'tanh': 1, 'gelu': 1,
            'softmax': 1, 'layer_norm': 1, 'residual': 2
        }
        # Adding custom_callable_ops to primitives if not already present for _validate_expression
        if 'custom_callable_ops' not in self.primitives:
            self.primitives['custom_callable_ops'] = {}


    def get_arity(self, op_name: str) -> int:
        """Get the arity (number of arguments) for an operator."""
        # _ai_op_arities is defined in __init__ to be instance specific
        # or could be a class variable if static for all AIGrammar instances

        # Check AI specific arities first
        if op_name in self._ai_op_arities:
            return self._ai_op_arities[op_name]

        # Fallback to parent class for standard operators
        try:
            return super().get_arity(op_name)
        except ValueError as e:
            # Python 3: use "raise ... from e" to chain exceptions
            raise ValueError(f"Unknown operator or function: '{op_name}' in AIGrammar") from e

    def is_operator_known(self, op_name: str) -> bool:
        """Check if an operator is known to this grammar."""
        try:
            self.get_arity(op_name) # This now checks both AI and base arities
            return True
        except ValueError:
            # Check other primitive categories
            # Ensure self.primitives is initialized (done in BaseGrammar placeholder)
            if op_name in self.primitives.get('custom_values', {}):
                return True
            if op_name in self.primitives.get('named_lists', {}):
                return True
            # Check custom sets values
            for value_list in self.primitives.get('custom_sets', {}).values():
                if op_name in value_list:
                    return True
            # Check custom callable ops, as per _validate_expression logic
            if op_name in self.primitives.get('custom_callable_ops', {}):
                return True
            return False

    def _validate_expression(self, operator: str, operands: List[Any]) -> bool:
        """Validate AI-specific expressions."""
        # First try parent validation (modified to call super's method correctly)
        if super()._validate_expression(operator, operands):
            return True

        # AI-specific validation
        # _ai_op_arities is an instance variable

        custom_callables = self.primitives.get('custom_callable_ops', {}).keys()

        if operator not in self._ai_op_arities and operator not in custom_callables:
            # If not in AI specific arities and not a custom callable,
            # and super validation failed, then it's false.
            # However, the original logic implies if super validation passes, it's true.
            # The current structure: if super says true, it's true.
            # If super says false, then we check AI specific rules.
            # If operator is not AI specific and not custom_callable, then it's False.
            return False

        expected_arity = self._ai_op_arities.get(operator)
        if expected_arity is not None: # It's an AI operator with a defined arity
            if len(operands) != expected_arity:
                return False
        # If it's a custom_callable, arity check might be handled differently or not at all here.
        # The original code does not explicitly check arity for custom_callables in this function.

        # Additional type checks for specific operators
        if operator == 'attention':
            # Ensure all operands are tensor compatible
            return all(self._is_tensor_compatible(op) for op in operands)

        # If it's a known AI operator (and passed arity check) or a custom_callable,
        # and no specific checks failed (like 'attention'), return True.
        if operator in self._ai_op_arities or operator in custom_callables:
            return True

        return False # Default if not caught by other conditions

    def _is_tensor_compatible(self, operand: Any) -> bool:
        """Check if operand is compatible with tensor operations."""
        # For now, accept most operand types - this can be refined
        return True

    def _attention_op(self, query: Any, key: Any, value: Any) -> Any:
        """Attention operation supporting both symbolic and numeric modes."""
        # Symbolic mode: return SymPy function
        if isinstance(query, (sp.Symbol, sp.Expr)):
            return sp.Function('Attention')(query, key, value)

        # Numeric mode: actual attention computation
        try:
            import torch
            import numpy as np

            # Convert to tensors if needed
            is_query_numpy = isinstance(query, np.ndarray)
            is_key_numpy = isinstance(key, np.ndarray)
            is_value_numpy = isinstance(value, np.ndarray)

            if is_query_numpy:
                query = torch.from_numpy(query).float()
            if is_key_numpy:
                key = torch.from_numpy(key).float()
            if is_value_numpy:
                value = torch.from_numpy(value).float()

            # Ensure inputs are torch tensors for computation
            if not (isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor)):
                # If not already tensors and not numpy arrays that were converted, this is an issue.
                # Or, one was numpy and others were not, leading to mixed types not handled by default.
                # For simplicity, this example assumes compatible types or successful conversion.
                # A more robust implementation might raise an error or handle type mismatches.
                pass # Assuming types are now torch.Tensor or this path leads to fallback.

            # Compute attention scores
            d_k = query.size(-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
            attention_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention_weights, value)

            # Convert back to numpy if all original inputs were numpy
            # This logic might need refinement based on desired output type consistency
            if is_query_numpy and is_key_numpy and is_value_numpy:
                return output.detach().numpy()
            return output

        except ImportError:
            # Fallback for symbolic representation
            return sp.Function('Attention')(query, key, value)
        except Exception as e: # Catch other numeric mode errors and fallback
            # print(f"Numeric attention failed: {e}, falling back to symbolic.") # Optional: log error
            return sp.Function('Attention')(query, key, value)

    def _embedding_op(self, indices: Any, embedding_matrix: Any) -> Any:
        """Embedding lookup operation."""
        if isinstance(indices, (sp.Symbol, sp.Expr)): # Check for symbolic indices
            # If embedding_matrix is also symbolic, represent it as such
            if isinstance(embedding_matrix, (sp.Symbol, sp.Expr)):
                 return sp.Function('Embedding')(indices, embedding_matrix)
            # If matrix is not symbolic (e.g. a string name or a placeholder), pass it as is
            return sp.Function('Embedding')(indices, sp.Symbol(str(embedding_matrix)))


        try:
            import torch
            import numpy as np

            is_indices_numpy = isinstance(indices, np.ndarray)
            is_matrix_numpy = isinstance(embedding_matrix, np.ndarray)

            if is_indices_numpy:
                indices = torch.from_numpy(indices).long()
            if is_matrix_numpy:
                embedding_matrix = torch.from_numpy(embedding_matrix).float()

            # Ensure inputs are torch tensors for computation
            if not (isinstance(indices, torch.Tensor) and isinstance(embedding_matrix, torch.Tensor)):
                # Fallback or error if types are not workable after potential conversion
                pass # Assuming types are now torch.Tensor or this path leads to fallback.

            # Simple embedding lookup
            output = embedding_matrix[indices]

            # Convert back to numpy if original inputs were numpy
            if is_indices_numpy and is_matrix_numpy: # Or just based on indices type, if matrix always tensor
                return output.detach().numpy()
            return output

        except ImportError:
            # Fallback for symbolic representation
            return sp.Function('Embedding')(indices, sp.Symbol(str(embedding_matrix))) # Ensure matrix is symbolic
        except Exception as e: # Catch other numeric mode errors and fallback
            # print(f"Numeric embedding failed: {e}, falling back to symbolic.") # Optional: log error
            return sp.Function('Embedding')(indices, sp.Symbol(str(embedding_matrix)))


# Example Usage (Optional)
if __name__ == '__main__':
    # Example primitives, including custom_callable_ops for testing
    prims = {
        'custom_values': {'my_val': 10},
        'named_lists': {'my_list': [1, 2, 3]},
        'custom_sets': {'my_set': {'a', 'b'}},
        'custom_callable_ops': {'custom_op': lambda x: x*x} # A custom operation
    }
    ai_grammar = AIGrammar(primitives=prims)

    # Test get_arity
    print(f"Arity of 'attention': {ai_grammar.get_arity('attention')}")  # Expected: 3
    print(f"Arity of 'relu': {ai_grammar.get_arity('relu')}")          # Expected: 1
    try:
        print(f"Arity of 'add' (from base): {ai_grammar.get_arity('add')}") # Expected: 2 (from placeholder BaseGrammar)
    except ValueError as e:
        print(e)
    try:
        ai_grammar.get_arity('unknown_op')
    except ValueError as e:
        print(e) # Expected: Unknown operator...

    # Test is_operator_known
    print(f"Is 'attention' known? {ai_grammar.is_operator_known('attention')}") # Expected: True
    print(f"Is 'my_val' known? {ai_grammar.is_operator_known('my_val')}")       # Expected: True
    print(f"Is 'my_list' known? {ai_grammar.is_operator_known('my_list')}")     # Expected: True
    print(f"Is 'a' (from my_set) known? {ai_grammar.is_operator_known('a')}")   # Expected: True
    print(f"Is 'custom_op' known? {ai_grammar.is_operator_known('custom_op')}") # Expected: True
    print(f"Is 'unknown_op' known? {ai_grammar.is_operator_known('unknown_op')}")# Expected: False

    # Test _validate_expression
    # BaseGrammar's _validate_expression for 'add' with 2 operands should be True
    print(f"Validate 'add', [1, 2]: {ai_grammar._validate_expression('add', [1, 2])}") # True (from BaseGrammar)
    # BaseGrammar's _validate_expression for 'add' with 1 operand should be False
    print(f"Validate 'add', [1]: {ai_grammar._validate_expression('add', [1])}") # False (from BaseGrammar)

    # AI specific: 'attention'
    print(f"Validate 'attention', [q, k, v]: {ai_grammar._validate_expression('attention', ['q', 'k', 'v'])}") # True
    print(f"Validate 'attention', [q, k]: {ai_grammar._validate_expression('attention', ['q', 'k'])}")       # False (arity)

    # AI specific: 'relu'
    print(f"Validate 'relu', [x]: {ai_grammar._validate_expression('relu', ['x'])}") # True
    print(f"Validate 'relu', [x, y]: {ai_grammar._validate_expression('relu', ['x', 'y'])}") # False (arity)

    # Custom callable op
    # The original _validate_expression doesn't explicitly check arity for custom_callables if they are not in _ai_op_arities
    # It returns True if operator is in custom_callables and not in _ai_op_arities (and super()._validate_expression was false)
    print(f"Validate 'custom_op', [data]: {ai_grammar._validate_expression('custom_op', ['data'])}") # True

    # Unknown operator
    print(f"Validate 'unknown_op', [data]: {ai_grammar._validate_expression('unknown_op', ['data'])}") # False

    # Test _is_tensor_compatible (always True for now)
    print(f"Is 'tensor_data' tensor compatible? {ai_grammar._is_tensor_compatible('tensor_data')}") # True
