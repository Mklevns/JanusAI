"""
Symbolic Mathematics Utilities for Janus Framework

This module serves as the central utility for all symbolic mathematics operations
within the Janus framework. It provides a clean, consistent, and robust interface
for symbolic computation tasks including evaluation, equivalence checking, and
symbolic manipulations.

Key Features:
- Fast numerical evaluation using sympy.lambdify
- Robust symbolic equivalence checking
- Integration with custom Expression and Variable classes
- Comprehensive error handling and caching
- Performance optimizations for large datasets
"""

import numpy as np
import sympy as sp
from typing import List, Union, Optional, Any, Dict, Tuple
from functools import lru_cache
import warnings
from dataclasses import dataclass

# Type aliases for better readability
ExpressionInput = Union[str, sp.Expr, Any]  # Any for custom Expression objects
VariablesList = List[str]
NumpyArray = np.ndarray


@dataclass
class EvaluationResult:
    """Container for expression evaluation results with metadata."""
    values: Optional[np.ndarray]
    success: bool
    error_message: Optional[str] = None
    expression_complexity: Optional[int] = None


class SymbolicMathError(Exception):
    """Custom exception for symbolic math operations."""
    pass


class ExpressionParsingError(SymbolicMathError):
    """Raised when expression string cannot be parsed."""
    pass


class VariableMismatchError(SymbolicMathError):
    """Raised when variables in expression don't match provided variables."""
    pass


class NumericalInstabilityError(SymbolicMathError):
    """Raised when numerical evaluation leads to instability."""
    pass


def evaluate_expression_on_data(
    expression_str: str, 
    variables: List[str], 
    data: np.ndarray,
    validate_variables: bool = True,
    handle_errors: bool = True
) -> np.ndarray:
    """
    Evaluate a symbolic expression numerically on provided data.
    
    Takes a string representation of a symbolic expression and evaluates it
    numerically on the provided data using fast lambdified functions.
    
    Args:
        expression_str: String representation of mathematical expression (e.g., "m * v**2 / 2")
        variables: List of variable names corresponding to data columns (e.g., ["m", "v"])
        data: NumPy array where columns correspond to variables
        validate_variables: Whether to validate that all expression variables are in variables list
        handle_errors: Whether to handle errors gracefully (return NaN) or raise exceptions
    
    Returns:
        NumPy array of evaluated expression values
        
    Raises:
        ExpressionParsingError: If expression cannot be parsed
        VariableMismatchError: If variables don't match expression variables
        NumericalInstabilityError: If evaluation leads to numerical issues
        
    Examples:
        >>> data = np.array([[1.0, 2.0], [2.0, 3.0]])  # mass, velocity
        >>> result = evaluate_expression_on_data("0.5 * m * v**2", ["m", "v"], data)
        >>> print(result)  # [2.0, 9.0] - kinetic energies
    """
    try:
        # Create SymPy expression
        sympy_expr = create_sympy_expression(expression_str, variables)
        
        # Get free symbols from the expression
        expr_variables = list(sympy_expr.free_symbols)
        expr_var_names = [str(var) for var in expr_variables]
        
        # Validate variable matching if requested
        if validate_variables and expr_var_names:
            missing_vars = set(expr_var_names) - set(variables)
            if missing_vars:
                raise VariableMismatchError(
                    f"Expression contains variables {missing_vars} not in provided variables {variables}"
                )
        
        # Handle constant expressions (no variables)
        if not expr_variables:
            try:
                constant_value = float(sympy_expr.evalf())
                return np.full(len(data), constant_value)
            except (ValueError, TypeError) as e:
                if handle_errors:
                    return np.full(len(data), np.nan)
                raise NumericalInstabilityError(f"Cannot evaluate constant expression: {e}")
        
        # Prepare data for evaluation
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Create fast lambdified function
        # Order variables to match the order they appear in the variables list
        ordered_vars = []
        ordered_data_indices = []
        
        for var_name in expr_var_names:
            if var_name in variables:
                var_symbol = sp.Symbol(var_name)
                ordered_vars.append(var_symbol)
                ordered_data_indices.append(variables.index(var_name))
        
        if not ordered_vars:
            if handle_errors:
                return np.full(len(data), np.nan)
            raise VariableMismatchError("No matching variables found between expression and data")
        
        # Create lambdified function for fast evaluation
        numerical_func = sp.lambdify(ordered_vars, sympy_expr, 'numpy')
        
        # Prepare arguments for the function
        if len(ordered_vars) == 1:
            # Single variable case
            col_index = ordered_data_indices[0]
            if col_index >= data.shape[1]:
                if handle_errors:
                    return np.full(len(data), np.nan)
                raise VariableMismatchError(f"Data column index {col_index} out of bounds")
            args = [data[:, col_index]]
        else:
            # Multiple variables case
            args = []
            for idx in ordered_data_indices:
                if idx >= data.shape[1]:
                    if handle_errors:
                        return np.full(len(data), np.nan)
                    raise VariableMismatchError(f"Data column index {idx} out of bounds")
                args.append(data[:, idx])
        
        # Evaluate the expression
        try:
            if len(args) == 1:
                result = numerical_func(args[0])
            else:
                result = numerical_func(*args)
            
            # Ensure result is a numpy array
            if np.isscalar(result):
                result = np.full(len(data), result)
            else:
                result = np.asarray(result)
            
            # Check for numerical issues
            if not np.all(np.isfinite(result)):
                if handle_errors:
                    # Replace infinite/NaN values with NaN
                    result = np.where(np.isfinite(result), result, np.nan)
                else:
                    raise NumericalInstabilityError("Expression evaluation resulted in infinite or NaN values")
            
            return result
            
        except (ZeroDivisionError, ValueError, RuntimeWarning) as e:
            if handle_errors:
                return np.full(len(data), np.nan)
            raise NumericalInstabilityError(f"Numerical evaluation failed: {e}")
            
    except Exception as e:
        if isinstance(e, (ExpressionParsingError, VariableMismatchError, NumericalInstabilityError)):
            raise
        if handle_errors:
            return np.full(len(data), np.nan)
        raise SymbolicMathError(f"Unexpected error in expression evaluation: {e}")


def are_expressions_equivalent_sympy(
    exp1_str: str, 
    exp2_str: str, 
    variables: List[str],
    tolerance: float = 1e-10,
    simplify_first: bool = True
) -> bool:
    """
    Check if two symbolic expressions are mathematically equivalent.
    
    Uses SymPy's symbolic manipulation capabilities to determine mathematical
    equivalence by checking if the difference between expressions simplifies to zero.
    
    Args:
        exp1_str: First expression string
        exp2_str: Second expression string  
        variables: List of variables present in the expressions
        tolerance: Numerical tolerance for equivalence checking
        simplify_first: Whether to simplify expressions before comparison
        
    Returns:
        True if expressions are equivalent, False otherwise
        
    Examples:
        >>> are_expressions_equivalent_sympy("x**2 + 2*x + 1", "(x + 1)**2", ["x"])
        True
        >>> are_expressions_equivalent_sympy("x + y", "y + x", ["x", "y"])
        True
    """
    try:
        # Parse both expressions
        expr1 = create_sympy_expression(exp1_str, variables)
        expr2 = create_sympy_expression(exp2_str, variables)
        
        # Simplify expressions if requested
        if simplify_first:
            expr1 = sp.simplify(expr1)
            expr2 = sp.simplify(expr2)
        
        # Calculate the difference
        difference = expr1 - expr2
        
        # Simplify the difference
        simplified_diff = sp.simplify(difference)
        
        # Check if difference is zero (symbolically)
        if simplified_diff.equals(0):
            return True
        
        # For expressions that might not simplify to exactly zero due to 
        # floating point representation, try numerical comparison
        if simplified_diff.is_number:
            try:
                numeric_diff = float(simplified_diff.evalf())
                return abs(numeric_diff) < tolerance
            except (ValueError, TypeError):
                pass
        
        # Try expanding and simplifying again
        expanded_diff = sp.expand(simplified_diff)
        if expanded_diff.equals(0):
            return True
            
        # Final numerical check if possible
        try:
            # If all symbols can be substituted with test values
            symbols = list(simplified_diff.free_symbols)
            if symbols:
                # Test with multiple random values
                test_points = 5
                for _ in range(test_points):
                    substitutions = {sym: np.random.uniform(-10, 10) for sym in symbols}
                    test_value = float(simplified_diff.subs(substitutions).evalf())
                    if abs(test_value) > tolerance:
                        return False
                return True
            
        except (ValueError, TypeError, AttributeError):
            pass
        
        return False
        
    except Exception as e:
        # If we can't determine equivalence due to parsing or other errors,
        # be conservative and return False
        warnings.warn(f"Could not determine expression equivalence: {e}")
        return False


@lru_cache(maxsize=256)
def create_sympy_expression(expression_str: str, variables_tuple: Tuple[str, ...]) -> sp.Expr:
    """
    Create a SymPy expression from string with proper variable handling.
    
    This is a cached helper function that parses string expressions into SymPy objects.
    Uses LRU cache to avoid redundant parsing of the same expressions.
    
    Args:
        expression_str: String representation of the expression
        variables_tuple: Tuple of variable names (tuple for hashability in cache)
        
    Returns:
        SymPy expression object
        
    Raises:
        ExpressionParsingError: If expression cannot be parsed
    """
    try:
        # Create symbol dictionary for consistent variable handling
        symbol_dict = {var: sp.Symbol(var) for var in variables_tuple}
        
        # Parse the expression with local dictionary
        try:
            # First try with sympify using the local symbol dictionary
            sympy_expr = sp.sympify(expression_str, locals=symbol_dict)
        except (sp.SympifyError, SyntaxError, TypeError) as e:
            # Try alternative parsing methods
            try:
                # Parse as string and substitute symbols
                sympy_expr = sp.parse_expr(expression_str, local_dict=symbol_dict)
            except Exception:
                raise ExpressionParsingError(f"Cannot parse expression '{expression_str}': {e}")
        
        return sympy_expr
        
    except Exception as e:
        if isinstance(e, ExpressionParsingError):
            raise
        raise ExpressionParsingError(f"Failed to create SymPy expression: {e}")


def create_sympy_expression(expression_str: str, variables: List[str]) -> sp.Expr:
    """
    Public interface for creating SymPy expressions.
    
    Converts variables list to tuple for caching and calls the cached version.
    """
    return create_sympy_expression(expression_str, tuple(variables))


def evaluate_custom_expression_on_data(
    expression: Any,
    data: np.ndarray,
    handle_errors: bool = True
) -> Optional[np.ndarray]:
    """
    Evaluate custom Expression objects (from janus.core.expressions) on data.
    
    This function handles evaluation of custom Expression objects that may have
    different interfaces than string expressions.
    
    Args:
        expression: Custom Expression object with symbolic representation
        data: NumPy array of input data
        handle_errors: Whether to handle errors gracefully
        
    Returns:
        NumPy array of results or None if evaluation fails
    """
    try:
        # Handle different expression types based on available attributes
        if hasattr(expression, 'symbolic') and expression.symbolic is not None:
            # Custom Expression object with symbolic representation
            sympy_expr = expression.symbolic
        elif hasattr(expression, 'to_sympy'):
            # Expression object with conversion method
            sympy_expr = expression.to_sympy()
        elif isinstance(expression, sp.Expr):
            # Already a SymPy expression
            sympy_expr = expression
        else:
            # Try to convert to string and parse
            sympy_expr = sp.sympify(str(expression))

        if sympy_expr is None:
            return None

        # Get free symbols from expression
        free_symbols = list(sympy_expr.free_symbols)
        
        if len(free_symbols) == 0:
            # Constant expression
            try:
                value = float(sympy_expr.evalf())
                return np.full(len(data), value)
            except (ValueError, TypeError):
                return None if handle_errors else None

        # Create lambdified function for fast evaluation
        if data.ndim == 1:
            # 1D input array
            if len(free_symbols) == 1:
                func = sp.lambdify(free_symbols[0], sympy_expr, 'numpy')
                result = func(data)
            else:
                # Multiple variables but 1D input - assume first variable
                func = sp.lambdify(free_symbols[0], sympy_expr, 'numpy')
                result = func(data)
        else:
            # Multi-dimensional input
            if len(free_symbols) <= data.shape[1]:
                func = sp.lambdify(free_symbols, sympy_expr, 'numpy')
                # Pass columns as separate arguments
                args = [data[:, i] for i in range(len(free_symbols))]
                if len(args) == 1:
                    result = func(args[0])
                else:
                    result = func(*args)
            else:
                # More variables than input dimensions
                return None

        # Ensure result is properly formatted
        if np.isscalar(result):
            result = np.full(len(data), result)
        else:
            result = np.asarray(result)

        # Handle numerical issues
        if not np.all(np.isfinite(result)):
            if handle_errors:
                result = np.where(np.isfinite(result), result, np.nan)
            else:
                return None

        return result

    except Exception as e:
        if handle_errors:
            warnings.warn(f"Expression evaluation failed: {e}")
            return None
        else:
            raise


def get_expression_complexity(expression: Union[str, sp.Expr, Any]) -> int:
    """
    Calculate the complexity of a symbolic expression.
    
    Complexity is defined as the number of operations (nodes) in the expression tree.
    
    Args:
        expression: Expression to analyze (string, SymPy, or custom object)
        
    Returns:
        Integer complexity score
    """
    try:
        # Convert to SymPy expression
        if isinstance(expression, str):
            sympy_expr = sp.sympify(expression)
        elif hasattr(expression, 'symbolic'):
            sympy_expr = expression.symbolic
        elif hasattr(expression, 'complexity'):
            return expression.complexity
        elif isinstance(expression, sp.Expr):
            sympy_expr = expression
        else:
            sympy_expr = sp.sympify(str(expression))
        
        # Count nodes in expression tree
        return _count_expression_nodes(sympy_expr)
        
    except Exception:
        # Default complexity if we can't determine it
        return 1


def _count_expression_nodes(expr: sp.Expr) -> int:
    """
    Recursively count nodes in a SymPy expression tree.
    
    Args:
        expr: SymPy expression
        
    Returns:
        Number of nodes in the expression tree
    """
    if expr.is_Atom:
        # Atoms (symbols, numbers) count as 1 node
        return 1
    else:
        # Non-atoms: count this node plus all children
        return 1 + sum(_count_expression_nodes(arg) for arg in expr.args)


def simplify_expression(expression_str: str, variables: List[str]) -> str:
    """
    Simplify a symbolic expression and return as string.
    
    Args:
        expression_str: Expression to simplify
        variables: List of variables in the expression
        
    Returns:
        Simplified expression as string
    """
    try:
        sympy_expr = create_sympy_expression(expression_str, variables)
        simplified = sp.simplify(sympy_expr)
        return str(simplified)
    except Exception:
        # Return original if simplification fails
        return expression_str


def validate_expression_syntax(expression_str: str, variables: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that an expression string has correct syntax.
    
    Args:
        expression_str: Expression string to validate
        variables: List of allowed variables
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        sympy_expr = create_sympy_expression(expression_str, variables)
        
        # Check if expression contains only allowed variables
        expr_vars = {str(sym) for sym in sympy_expr.free_symbols}
        allowed_vars = set(variables)
        
        unexpected_vars = expr_vars - allowed_vars
        if unexpected_vars:
            return False, f"Expression contains unexpected variables: {unexpected_vars}"
        
        return True, None
        
    except ExpressionParsingError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Validation error: {e}"


# Utility functions for integration with existing codebase
def expression_to_string(expression: Any) -> str:
    """
    Convert various expression types to string representation.
    
    Args:
        expression: Expression object (SymPy, custom, or string)
        
    Returns:
        String representation of the expression
    """
    if isinstance(expression, str):
        return expression
    elif isinstance(expression, sp.Expr):
        return str(expression)
    elif hasattr(expression, 'symbolic'):
        return str(expression.symbolic)
    else:
        return str(expression)


def create_evaluation_result(
    expression_str: str,
    variables: List[str], 
    data: np.ndarray,
    **kwargs
) -> EvaluationResult:
    """
    Create a comprehensive evaluation result with metadata.
    
    Args:
        expression_str: Expression to evaluate
        variables: Variable names
        data: Input data
        **kwargs: Additional arguments for evaluation
        
    Returns:
        EvaluationResult object with values and metadata
    """
    try:
        values = evaluate_expression_on_data(expression_str, variables, data, **kwargs)
        complexity = get_expression_complexity(expression_str)
        
        return EvaluationResult(
            values=values,
            success=True,
            expression_complexity=complexity
        )
        
    except Exception as e:
        return EvaluationResult(
            values=None,
            success=False,
            error_message=str(e)
        )
