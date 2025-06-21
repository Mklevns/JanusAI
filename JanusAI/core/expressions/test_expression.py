"""
Comprehensive tests for the Expression class and related functionality.
Tests cover Expression creation, complexity calculation, symbolic conversion,
and edge cases like division by zero.
"""
import pytest
import numpy as np
import sympy as sp
from unittest.mock import MagicMock, patch

from janus_ai.core.expressions.expression import Expression, Variable
from janus_ai.core.expressions.symbolic_math import evaluate_expression_on_data


class TestVariable:
    """Test the Variable class functionality."""
    
    def test_variable_creation(self):
        """Test basic variable creation and properties."""
        var = Variable(name="x", index=0, properties={"unit": "m", "type": "position"})
        
        assert var.name == "x"
        assert var.index == 0
        assert var.properties["unit"] == "m"
        assert var.complexity == 1
        assert str(var) == "x"
        assert "Variable(name='x'" in repr(var)
    
    def test_variable_hash_equality(self):
        """Test variable hashing and equality."""
        var1 = Variable("x", 0, {"unit": "m"})
        var2 = Variable("x", 0, {"unit": "s"})  # Different properties, same hash
        var3 = Variable("y", 0, {"unit": "m"})  # Different name
        
        # Same name and index should have same hash
        assert hash(var1) == hash(var2)
        # Different name should have different hash
        assert hash(var1) != hash(var3)
        
        # Equality includes properties
        assert var1 != var2  # Different properties
        assert var1 != var3  # Different name


class TestExpression:
    """Test the Expression class functionality."""
    
    @pytest.fixture
    def basic_variables(self):
        """Provide basic variables for testing."""
        return {
            'x': Variable("x", 0, {}),
            'y': Variable("y", 1, {}),
            'z': Variable("z", 2, {})
        }
    
    def test_expression_creation_basic(self, basic_variables):
        """Test basic expression creation."""
        x = basic_variables['x']
        
        # Test variable expression
        var_expr = Expression('var', [x])
        assert var_expr.operator == 'var'
        assert var_expr.operands[0] == x
        assert var_expr.complexity == 2  # 1 (operator) + 1 (variable)
        assert isinstance(var_expr.symbolic, sp.Symbol)
        assert str(var_expr.symbolic) == 'x'
        
        # Test constant expression
        const_expr = Expression('const', [3.14])
        assert const_expr.operator == 'const'
        assert const_expr.operands[0] == 3.14
        assert const_expr.complexity == 2  # 1 (operator) + 1 (constant)
        assert const_expr.symbolic == sp.Float(3.14)
    
    def test_expression_arithmetic_operators(self, basic_variables):
        """Test arithmetic operator expressions."""
        x, y = basic_variables['x'], basic_variables['y']
        
        # Addition
        add_expr = Expression('+', [
            Expression('var', [x]),
            Expression('var', [y])
        ])
        assert add_expr.complexity == 5  # 1 (+) + 2 (var x) + 2 (var y)
        assert add_expr.symbolic == sp.Symbol('x') + sp.Symbol('y')
        
        # Multiplication
        mul_expr = Expression('*', [
            Expression('var', [x]),
            Expression('const', [2.0])
        ])
        assert mul_expr.complexity == 5
        assert mul_expr.symbolic == 2.0 * sp.Symbol('x')
        
        # Power
        pow_expr = Expression('**', [
            Expression('var', [x]),
            Expression('const', [2])
        ])
        assert pow_expr.complexity == 5
        assert pow_expr.symbolic == sp.Symbol('x')**2
    
    def test_expression_unary_functions(self, basic_variables):
        """Test unary function expressions."""
        x = basic_variables['x']
        var_x = Expression('var', [x])
        
        # Sine
        sin_expr = Expression('sin', [var_x])
        assert sin_expr.complexity == 3  # 1 (sin) + 2 (var x)
        assert sin_expr.symbolic == sp.sin(sp.Symbol('x'))
        
        # Exponential
        exp_expr = Expression('exp', [var_x])
        assert exp_expr.symbolic == sp.exp(sp.Symbol('x'))
        
        # Logarithm
        log_expr = Expression('log', [var_x])
        assert log_expr.symbolic == sp.log(sp.Symbol('x'))
    
    def test_expression_division_by_zero(self, basic_variables):
        """Test division by zero handling."""
        x = basic_variables['x']
        
        # Numeric zero division
        div_by_zero = Expression('/', [
            Expression('var', [x]),
            Expression('const', [0])
        ])
        assert div_by_zero.symbolic == sp.nan
        
        # Symbolic zero division (0 * x)
        symbolic_zero = Expression('*', [
            Expression('const', [0]),
            Expression('var', [x])
        ])
        div_by_symbolic_zero = Expression('/', [
            Expression('const', [1]),
            symbolic_zero
        ])
        assert div_by_symbolic_zero.symbolic == sp.nan
    
    def test_expression_nested_complexity(self, basic_variables):
        """Test complexity calculation for nested expressions."""
        x, y, z = basic_variables['x'], basic_variables['y'], basic_variables['z']
        
        # ((x + y) * z) + sin(x)
        inner_add = Expression('+', [
            Expression('var', [x]),
            Expression('var', [y])
        ])
        mul_expr = Expression('*', [inner_add, Expression('var', [z])])
        sin_expr = Expression('sin', [Expression('var', [x])])
        complex_expr = Expression('+', [mul_expr, sin_expr])
        
        # Complexity breakdown:
        # inner_add: 1 + 2 + 2 = 5
        # mul_expr: 1 + 5 + 2 = 8
        # sin_expr: 1 + 2 = 3
        # complex_expr: 1 + 8 + 3 = 12
        assert complex_expr.complexity == 12
    
    def test_expression_clone(self, basic_variables):
        """Test expression cloning functionality."""
        x, y = basic_variables['x'], basic_variables['y']
        
        # Create a nested expression
        original = Expression('+', [
            Expression('*', [
                Expression('var', [x]),
                Expression('const', [2.0])
            ]),
            Expression('var', [y])
        ])
        
        # Clone it
        cloned = original.clone()
        
        # Test that clone is a different object
        assert cloned is not original
        assert cloned.operands is not original.operands
        
        # Test that clone has same structure
        assert cloned.operator == original.operator
        assert cloned.complexity == original.complexity
        assert cloned.symbolic == original.symbolic
        
        # Test deep cloning of nested expressions
        assert cloned.operands[0] is not original.operands[0]
        assert cloned.operands[0].operator == original.operands[0].operator
        
        # Variables should be shared (not cloned)
        assert cloned.operands[1].operands[0] is original.operands[1].operands[0]
    
    def test_expression_edge_cases(self, basic_variables):
        """Test edge cases and error conditions."""
        x = basic_variables['x']
        
        # Test invalid operators
        with pytest.raises(ValueError):
            invalid_expr = Expression('invalid_op', [Expression('var', [x])])
            _ = invalid_expr.symbolic
        
        # Test empty operands for operators that need them
        with pytest.raises(Exception):
            Expression('+', [])  # Addition needs operands
        
        # Test special cases
        # Derivative
        var_x = Expression('var', [x])
        diff_expr = Expression('diff', [var_x, x])
        assert diff_expr.symbolic == sp.Integer(1)  # d/dx(x) = 1
    
    def test_expression_evaluation(self, basic_variables):
        """Test expression evaluation on data."""
        x, y = basic_variables['x'], basic_variables['y']
        
        # Create expression: 2*x + y
        expr = Expression('+', [
            Expression('*', [
                Expression('const', [2]),
                Expression('var', [x])
            ]),
            Expression('var', [y])
        ])
        
        # Test data
        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        # Evaluate
        result = evaluate_expression_on_data(
            expr.symbolic,
            data,
            ['x', 'y']
        )
        
        expected = 2 * data[:, 0] + data[:, 1]  # 2*x + y
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_expression_with_nan_handling(self):
        """Test expression evaluation with NaN handling."""
        x_var = Variable("x", 0, {})
        
        # Create log(x) which will be undefined for negative x
        log_expr = Expression('log', [Expression('var', [x_var])])
        
        # Test data with negative values
        data = np.array([[-1.0], [1.0], [2.0]])
        
        # Evaluate with error handling
        result = evaluate_expression_on_data(
            log_expr.symbolic,
            data,
            ['x'],
            handle_errors=True
        )
        
        # First value should be NaN, others should be computed
        assert np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])


class TestExpressionSymbolicConversion:
    """Test symbolic conversion edge cases."""
    
    def test_all_operators(self):
        """Test that all supported operators convert correctly."""
        x = Variable("x", 0, {})
        y = Variable("y", 1, {})
        
        operators_unary = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']
        operators_binary = ['+', '-', '*', '/', '**']
        
        var_x = Expression('var', [x])
        var_y = Expression('var', [y])
        const_2 = Expression('const', [2])
        
        # Test unary operators
        for op in operators_unary:
            expr = Expression(op, [var_x])
            assert isinstance(expr.symbolic, sp.Expr)
            assert not expr.symbolic.is_number  # Should contain variable
        
        # Test binary operators
        for op in operators_binary:
            expr = Expression(op, [var_x, var_y])
            assert isinstance(expr.symbolic, sp.Expr)
            
            # Test with constants
            expr_const = Expression(op, [var_x, const_2])
            assert isinstance(expr_const.symbolic, sp.Expr)
    
    def test_complex_symbolic_expressions(self):
        """Test complex nested symbolic expressions."""
        x = Variable("x", 0, {})
        var_x = Expression('var', [x])
        
        # Build: sin(x) * exp(-x^2)
        x_squared = Expression('**', [var_x, Expression('const', [2])])
        neg_x_squared = Expression('*', [Expression('const', [-1]), x_squared])
        exp_term = Expression('exp', [neg_x_squared])
        sin_term = Expression('sin', [var_x])
        full_expr = Expression('*', [sin_term, exp_term])
        
        # Check symbolic representation
        expected = sp.sin(sp.Symbol('x')) * sp.exp(-sp.Symbol('x')**2)
        assert full_expr.symbolic == expected