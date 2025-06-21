# JanusAI/core/grammar/ai_grammar.py
"""AI-specific Grammar components, extending Progressive Grammar."""

import numpy as np
import sympy as sp
import torch # Added for _attention_op, _embedding_op
from typing import Dict, List, Tuple, Optional, Set, Any, Union # Added Union
import logging # Added for _validate_expression

from janus_ai.core.grammar.progressive_grammar import ProgressiveGrammar
from janus_ai.core.expressions.expression import Expression, Variable # Variable is used in _validate_expression

class AIGrammar(ProgressiveGrammar):
    """
    Extends ProgressiveGrammar with primitives and operations tailored for
    AI model interpretability, such as attention and activation functions.
    """
    def __init__(self):
        super().__init__(load_defaults=False)
        self.add_primitive_set('activation_types', ['relu', 'sigmoid', 'tanh', 'gelu'])
        self.add_primitive('attention', self._attention_op, category='custom_callable_ops')
        self.add_primitive('embedding_lookup', self._embedding_op, category='custom_callable_ops') # Renamed for clarity
        self.add_primitive('if_then_else', lambda cond, true_val, false_val: true_val if cond else false_val, category='custom_callable_ops')
        self.add_primitive('threshold', lambda x, t: x > t, category='custom_callable_ops')
        self.add_primitive('weighted_sum', lambda weights, values: sum(w*v for w,v in zip(weights, values)), category='custom_callable_ops')
        self.add_primitive('max_pool', lambda values: max(values) if values else None, category='custom_callable_ops')
        # Add common NN ops as placeholders, actual execution logic might be external
        self.primitives['unary_ops'].update(['relu', 'sigmoid', 'tanh', 'gelu', 'softmax', 'layer_norm'])
        self.primitives['binary_ops'].update(['residual']) # e.g. residual(x, y) = x + y


    def add_primitive_set(self, name: str, values: List[str]):
        if 'custom_sets' not in self.primitives: self.primitives['custom_sets'] = {}
        self.primitives['custom_sets'][name] = values

    def add_primitive(self, name: str, func_or_values: Any, category: Optional[str] = None):
        if callable(func_or_values):
            cat = category if category else 'custom_callable_ops'
            if cat not in self.primitives: self.primitives[cat] = {}
            self.primitives[cat][name] = func_or_values
        elif isinstance(func_or_values, list):
            if 'named_lists' not in self.primitives: self.primitives['named_lists'] = {}
            self.primitives['named_lists'][name] = func_or_values
        else:
            if 'custom_values' not in self.primitives: self.primitives['custom_values'] = {}
            self.primitives['custom_values'][name] = func_or_values

    def _attention_op(self, query: Any, key: Any, value: Any) -> Any:
        if isinstance(query, (sp.Symbol, sp.Expr)): # Symbolic mode
            return sp.Function('Attention')(query, key, value)
        # Numeric mode
        if isinstance(query, np.ndarray):
            q = torch.tensor(query, dtype=torch.float32) if not isinstance(query, torch.Tensor) else query
            k = torch.tensor(key, dtype=torch.float32) if not isinstance(key, torch.Tensor) else key
            v = torch.tensor(value, dtype=torch.float32) if not isinstance(value, torch.Tensor) else value
            d_k = q.shape[-1]
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
            attention_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention_weights, v)
            return output.numpy() if isinstance(query, np.ndarray) else output
        return f"Attention({query}, {key}, {value})"

    def _embedding_op(self, indices: Any, embedding_matrix: Any) -> Any: # Renamed from _embedding_lookup
        if isinstance(indices, (sp.Symbol, sp.Expr)): # Symbolic mode
            return sp.Function('Embedding')(indices, embedding_matrix)
        # Numeric mode
        if isinstance(indices, (np.ndarray, list)):
            indices_arr = np.array(indices, dtype=int) # Ensure it's an array for indexing
            if isinstance(embedding_matrix, np.ndarray): return embedding_matrix[indices_arr]
            elif isinstance(embedding_matrix, torch.Tensor):
                indices_tensor = torch.tensor(indices_arr, dtype=torch.long)
                return embedding_matrix[indices_tensor].numpy()
            elif isinstance(embedding_matrix, str): # Symbolic reference to matrix name
                return f"Embedding({indices_arr}, {embedding_matrix})"
        return f"EmbeddingLookup({indices}, {embedding_matrix})"

    def _is_tensor_compatible(self, operand: Any) -> bool:
        return isinstance(operand, (Expression, Variable, np.ndarray, list, sp.Expr, int, float))

    def _to_sympy(self, expr_node: Expression) -> sp.Expr: # Policy for Expression class to use
        operator = expr_node.operator
        # This method is a policy provider for Expression's .symbolic property.
        # It should only handle cases specific to AIGrammar.
        # Standard operations (+, -, sin, cos, etc.) should be handled by
        # Expression class itself or by ProgressiveGrammar's _to_sympy if it were providing a policy.

        # AI-specific operators that need custom Sympy representation
        if operator == 'if_then_else':
            # Ensure operands are converted to Sympy first
            sympy_operands = [op.symbolic if isinstance(op, Expression) else
                              op.symbolic if isinstance(op, Variable) else
                              sp.sympify(op) for op in expr_node.operands]
            return sp.Piecewise((sympy_operands[1], sympy_operands[0]), (sympy_operands[2], True))
        elif operator == 'threshold':
            sympy_operands = [op.symbolic if isinstance(op, Expression) else
                              op.symbolic if isinstance(op, Variable) else
                              sp.sympify(op) for op in expr_node.operands]
            return sympy_operands[0] > sympy_operands[1] # Results in a Sympy Boolean expression

        # For other AI ops like 'attention', 'embedding_lookup', 'relu', 'softmax', etc.,
        # represent them as uninterpreted functions in Sympy.
        # This relies on Expression class calling this policy for unknown operators.
        # Check if operator is one of AIGrammar's special custom callables or added unary/binary ops
        ai_ops_needing_func_representation = (
            set(self.primitives.get('custom_callable_ops', {}).keys()) |
            {'relu', 'sigmoid', 'tanh', 'gelu', 'softmax', 'layer_norm', 'residual'}
        )
        if operator in ai_ops_needing_func_representation and operator not in ['if_then_else', 'threshold']:
            sympy_operands = [op.symbolic if isinstance(op, Expression) else
                              op.symbolic if isinstance(op, Variable) else
                              sp.sympify(op) for op in expr_node.operands]
            capitalized_op = operator.capitalize() if not operator.isupper() else operator
            return sp.Function(capitalized_op)(*sympy_operands)

        # If the operator is not specifically handled by AIGrammar's policy,
        # it means it should be handled by the Expression class's default _to_sympy logic
        # (which covers standard math ops, variables, constants).
        # Thus, we should indicate that this policy doesn't apply by raising an error or returning a special value.
        # However, the design implies Expression calls grammar.get_sympy_equivalent(self) or similar.
        # For now, let's assume Expression.symbolic will try this policy, and if it fails (e.g. raises AttributeError
        # because AIGrammar doesn't handle it), Expression falls back to its own internal _to_sympy.
        # A cleaner way would be for Expression to call:
        # try: return self.grammar._to_sympy_policy(self) except NotHandledByPolicy: return self._internal_to_sympy()
        # For this refactoring, we keep the structure as close as possible.
        # The original AIGrammar._to_sympy also had a complex fallback to a temporary Expression.
        # This was problematic. AIGrammar should only define new symbolic forms.
        # If an op is not AI-specific, Expression itself should handle it.
        # So, if we reach here, it's an op AIGrammar doesn't know how to symbolize beyond default.
        # We should let Expression handle it. How to signal this?
        # The original code had "return super()._to_sympy(expr_node)" for var/const,
        # and then "temp_expr_for_std_op_symb = Expression(operator, sympy_operands)"
        # This implies Expression._to_sympy is the ultimate fallback.
        # Let's make this explicit: if AIGrammar doesn't have a rule, it doesn't handle it.
        # Expression.symbolic property must be structured to try grammar policy first, then its own.
        raise NotImplementedError(f"AIGrammar does not provide a specific Sympy conversion for operator '{operator}'. Expression class should handle.")


    def _validate_expression(self, operator: str, operands: List[Any]) -> bool:
        # First, try validation with ProgressiveGrammar's rules.
        # This covers standard ops, learned functions (if any were added to parent's primitives), var, const.
        if super()._validate_expression(operator, operands):
            return True

        # Now check AI-specific operators if parent validation failed.
        # Parent validation fails if operator is unknown to it, or arity/type is wrong.
        # We only proceed if operator is potentially an AI op.
        ai_operator_arity = {
            'attention': 3, 'embedding_lookup': 2, 'if_then_else': 3,
            'threshold': 2, 'weighted_sum': 2, 'max_pool': 1, # max_pool takes one list
            'relu': 1, 'sigmoid': 1, 'tanh': 1, 'gelu': 1, 'softmax': 1, 'layer_norm': 1,
            'residual': 2
        }
        # Also include custom callables that might not be in the fixed arity dict
        custom_callables = self.primitives.get('custom_callable_ops', {}).keys()

        if operator not in ai_operator_arity and operator not in custom_callables:
            # If it's not in parent's valid ops (checked by super call) AND not in AI specific ops,
            # then it's truly unknown or invalid according to this grammar.
            return False

        expected_arity = ai_operator_arity.get(operator)

        # For custom callables not in ai_operator_arity, we can't check arity here easily.
        # Assume they are valid if operator name matches. More robust: store arity with custom_callable.
        if expected_arity is not None:
            if len(operands) != expected_arity:
                logging.debug(f"AIGrammar validation: Arity mismatch for {operator}. Expected {expected_arity}, got {len(operands)}")
                return False
        elif operator in custom_callables:
            # Arity not specified in ai_operator_arity, assume valid for now if op name matches a custom callable.
            # This part might need refinement if custom callables have fixed arities not listed.
            pass
        else:
            # Operator is not in fixed arity list and not a registered custom callable.
            # This case should ideally be caught by the first check (operator not in ai_operator_arity and operator not in custom_callables)
            return False


        # Type checks for AI operators (example for attention)
        if operator == 'attention':
            return all(self._is_tensor_compatible(op) for op in operands)
        # Add more specific type checks for other AI ops if needed.
        # For now, allow general operand types if arity matches for other AI ops.
        return True


# The get_arity method was monkey-patched onto AIGrammar in the original base_grammar.py
# It should be a proper method of the AIGrammar class.
    def get_arity(self, op_name: str) -> int:
        _ai_op_arities = {
            'attention': 3, 'embedding_lookup': 2, 'if_then_else': 3,
            'threshold': 2, 'weighted_sum': 2, 'max_pool': 1,
            'relu':1, 'sigmoid':1, 'tanh':1, 'gelu':1, 'softmax':1, 'layer_norm':1, 'residual':2
            # Add other AI specific ops from self.primitives if they have fixed arity
        }
        # Check AI specific arities first
        if op_name in _ai_op_arities:
            return _ai_op_arities[op_name]

        # Check custom callable ops if not in the dict above (might be dynamically added)
        # This part is tricky if arity isn't stored with them. For now, assume covered by _ai_op_arities
        # or they fall through to parent.

        # Fallback to ProgressiveGrammar's get_arity for standard operators
        # or operators AIGrammar inherited but doesn't override arity for.
        try:
            return super().get_arity(op_name)
        except ValueError as e:
            # If super also doesn't know it, then it's truly unknown to AIGrammar too.
            raise ValueError(f"Unknown operator or function: '{op_name}' in AIGrammar") from e

# The is_operator_known method was also monkey-patched.
# It should rely on get_arity.
    def is_operator_known(self, op_name: str) -> bool:
        try:
            self.get_arity(op_name) # This will use AIGrammar's get_arity
            return True
        except ValueError:
             # AIGrammar might also have constants or other symbols defined in primitives
             # that are not "operators" with arity but are "known" symbols.
            if op_name in self.primitives.get('custom_values', {}): return True
            if op_name in self.primitives.get('named_lists', {}): return True
            # Check custom sets values
            for value_list in self.primitives.get('custom_sets', {}).values():
                if op_name in value_list: return True
            return False