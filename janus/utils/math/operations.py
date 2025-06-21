import numpy as np
import sympy as sp
from typing import Any, Dict, List, Union
import re

def calculate_expression_complexity(expression: Union[str, Any]) -> int:
    """Calculate complexity of mathematical expression."""
    expr_str = str(expression)

    # Count operations
    operations = _count_operations(expr_str)

    # Count variables
    variables = len(set(re.findall(r'[a-zA-Z_]\w*', expr_str)))

    # Count constants
    constants = len(re.findall(r'\d+\.?\d*', expr_str))

    return operations + variables + constants

def _count_operations(expr_str: str) -> int:
    """Count mathematical operations in expression string."""
    operators = ['+', '-', '*', '/', '**', '^', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
    count = 0
    for op in operators:
        count += expr_str.count(op)
    return count

def calculate_symbolic_accuracy(predicted_expr: Any, target_expr: Any, test_points: np.ndarray) -> float:
    """Calculate accuracy between symbolic expressions."""
    try:
        if isinstance(predicted_expr, str):
            predicted_expr = sp.sympify(predicted_expr)
        if isinstance(target_expr, str):
            target_expr = sp.sympify(target_expr)

        # Get free symbols
        symbols = list(predicted_expr.free_symbols.union(target_expr.free_symbols))

        if not symbols:
            return 1.0 if predicted_expr.equals(target_expr) else 0.0

        # Evaluate at test points
        pred_values = []
        target_values = []

        for point in test_points:
            subs_dict = {sym: val for sym, val in zip(symbols, point)}

            try:
                pred_val = float(predicted_expr.subs(subs_dict))
                target_val = float(target_expr.subs(subs_dict))
                pred_values.append(pred_val)
                target_values.append(target_val)
            except:
                continue

        if not pred_values:
            return 0.0

        # Calculate R² score
        from sklearn.metrics import r2_score
        return max(0.0, r2_score(target_values, pred_values))

    except Exception:
        return 0.0

def evaluate_expression_on_data(expression: Union[str, Any], data: Dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate symbolic expression on numerical data."""
    try:
        if isinstance(expression, str):
            expr = sp.sympify(expression)
        else:
            expr = expression

        # Get symbols in the expression
        symbols = list(expr.free_symbols)

        if not symbols:
            # Constant expression
            return np.full(len(next(iter(data.values()))), float(expr))

        # Determine data size
        data_size = len(next(iter(data.values())))
        results = np.zeros(data_size)

        for i in range(data_size):
            subs_dict = {}
            for sym in symbols:
                sym_name = str(sym)
                if sym_name in data:
                    subs_dict[sym] = data[sym_name][i]
                else:
                    # Try to find matching variable name
                    for key in data.keys():
                        if key.endswith(sym_name) or sym_name in key:
                            subs_dict[sym] = data[key][i]
                            break

            try:
                results[i] = float(expr.subs(subs_dict))
            except:
                results[i] = 0.0

        return results

    except Exception:
        return np.zeros(len(next(iter(data.values()))))

def are_expressions_equivalent_sympy(expr1: Any, expr2: Any, tolerance: float = 1e-10) -> bool:
    """Check if two SymPy expressions are equivalent."""
    try:
        if isinstance(expr1, str):
            expr1 = sp.sympify(expr1)
        if isinstance(expr2, str):
            expr2 = sp.sympify(expr2)

        # Direct equality check
        if expr1.equals(expr2):
            return True

        # Simplification check
        diff = sp.simplify(expr1 - expr2)
        return abs(float(diff)) < tolerance if diff.is_number else False

    except:
        return False
