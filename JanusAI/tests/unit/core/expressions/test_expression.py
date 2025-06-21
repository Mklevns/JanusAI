"""
Tests for core/expressions/expression.py: Variable and Expression classes.
"""
import pytest
import sympy as sp
from JanusAI.core.expressions.expression import Variable, Expression # Assuming Expression is also in this file

# Tests for Variable Class
class TestVariable:
    def test_variable_creation_basic(self):
        """Test basic Variable creation."""
        var = Variable(name="x", index=0)
        assert var.name == "x"
        assert var.index == 0
        assert var.properties == {}
        assert isinstance(var.symbolic, sp.Symbol)
        assert str(var.symbolic) == "x"

    def test_variable_creation_with_properties(self):
        """Test Variable creation with properties."""
        props = {"is_constant": True, "value": 5.0}
        var = Variable(name="y", index=1, properties=props)
        assert var.name == "y"
        assert var.index == 1
        assert var.properties == props
        assert isinstance(var.symbolic, sp.Symbol)
        assert str(var.symbolic) == "y"

    def test_variable_post_init_symbolic_creation(self):
        """Test that symbolic attribute is created correctly in __post_init__."""
        var = Variable(name="t", index=2)
        assert hasattr(var, 'symbolic')
        assert var.symbolic == sp.Symbol("t")

    def test_variable_invalid_name(self):
        """Test Variable creation with an invalid SymPy name."""
        with pytest.raises(ValueError, match="Variable name '123var' must be a valid Python identifier."):
            Variable(name="123var", index=0)
        with pytest.raises(ValueError, match="Variable name 'var with space' must be a valid Python identifier."):
            Variable(name="var with space", index=1)

    def test_variable_hash_and_eq(self):
        """Test hashing and equality for Variable objects."""
        var1 = Variable(name="x", index=0)
        var2 = Variable(name="x", index=0)
        var3 = Variable(name="x", index=1)
        var4 = Variable(name="y", index=0)
        var5 = Variable(name="x", index=0, properties={"unit": "m"})

        assert var1 == var2, "Variables with same name and index should be equal if properties are same."
        assert var1 is not var2
        assert hash(var1) == hash(var2), "Hashes should be equal for equal Variables (based on name and index)."

        assert var1 != var3, "Variables with different indices should not be equal."
        assert hash(var1) != hash(var3), "Hashes should differ for variables with different indices." # hash uses index

        assert var1 != var4, "Variables with different names should not be equal."

        # Dataclass eq=True by default compares all fields.
        # Our custom __hash__ only uses name and index.
        assert var1 != var5, "Variables with same name/index but different properties should not be equal by default dataclass __eq__."
        # However, their hashes based on (name, index) will be the same.
        assert hash(var1) == hash(var5), "Hashes based on (name, index) should be the same even if properties differ."

        var_set = {var1, var2, var3, var4, var5}
        # var1 and var2 are "equal" if properties are same (var2 has no different props).
        # var5 is different from var1 due to properties.
        # So, the set should contain var1 (or var2), var3, var4, var5. Expected size 4.
        assert len(var_set) == 4

    def test_variable_str_and_repr(self):
        """Test __str__ and __repr__ for Variable."""
        var = Variable(name="pos", index=3, properties={"unit": "m"})
        assert str(var) == "pos"
        assert repr(var) == "Variable(name='pos', index=3, properties={'unit': 'm'})"

    def test_variable_complexity(self):
        """Test the complexity property of Variable."""
        var = Variable(name="c", index=0)
        assert var.complexity == 1

# Tests for Expression Class
@pytest.fixture
def variables():
    """Provides common Variable instances for Expression tests."""
    return {
        "x": Variable(name="x", index=0),
        "y": Variable(name="y", index=1),
        "z": Variable(name="z", index=2)
    }

class TestExpression:
    def test_expression_creation_leaf_nodes(self, variables):
        """Test creation of leaf Expressions (variables and constants)."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        expr_y_direct_var_obj = Expression(operator='var', operands=[Variable("y_direct", 10)])
        expr_y_direct_name = Expression(operator='var', operands=["y_named"]) # Test with name string
        expr_const_5 = Expression(operator='const', operands=[5.0])
        expr_const_neg2 = Expression(operator='const', operands=[-2])

        assert str(expr_x.symbolic) == "x"
        assert expr_x.complexity == 2 # Op ('var') + Var ('x')

        assert str(expr_y_direct_var_obj.symbolic) == "y_direct"
        assert expr_y_direct_var_obj.complexity == 2

        assert str(expr_y_direct_name.symbolic) == "y_named"
        assert expr_y_direct_name.complexity == 2

        assert expr_const_5.symbolic == sp.Float(5.0)
        assert expr_const_5.complexity == 2 # Op ('const') + Value (5.0)

        assert expr_const_neg2.symbolic == sp.Float(-2)
        assert expr_const_neg2.complexity == 2

    def test_expression_creation_composite(self, variables):
        """Test creation of composite Expressions."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        expr_const_5 = Expression(operator='const', operands=[5.0])

        expr_add = Expression(operator='+', operands=[expr_x, expr_const_5]) # x + 5
        assert str(expr_add.symbolic) == "x + 5.0"
        # Complexity: 1 (for '+') + C(expr_x) (2) + C(expr_const_5) (2) = 5
        assert expr_add.complexity == 5

        expr_y = Expression(operator='var', operands=[variables["y"]])
        expr_mul = Expression(operator='*', operands=[expr_y, expr_add]) # y * (x + 5)
        assert str(expr_mul.symbolic) == "y*(x + 5.0)"
        # Complexity: 1 (for '*') + C(expr_y) (2) + C(expr_add) (5) = 8
        assert expr_mul.complexity == 8

    def test_expression_symbolic_and_complexity_properties(self, variables):
        """Test that .symbolic and .complexity properties work."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        expr_s = expr_x.symbolic # Access property
        comp = expr_x.complexity # Access property
        assert str(s) == "x"
        assert comp == 2

    def test_to_sympy_various_operators(self, variables):
        """Test _to_sympy conversion for various operators."""
        x, y = variables["x"], variables["y"]
        expr_x = Expression(operator='var', operands=[x])
        expr_y = Expression(operator='var', operands=[y])
        expr_2 = Expression(operator='const', operands=[2.0])
        expr_3 = Expression(operator='const', operands=[3.0])

        # Arithmetic
        assert str(Expression(operator='+', operands=[expr_x, expr_y]).symbolic) == "x + y"
        assert str(Expression(operator='-', operands=[expr_x, expr_y]).symbolic) == "x - y"
        assert str(Expression(operator='*', operands=[expr_x, expr_y]).symbolic) == "x*y"
        assert str(Expression(operator='/', operands=[expr_x, expr_y]).symbolic) == "x/y"
        assert str(Expression(operator='**', operands=[expr_x, expr_2]).symbolic) == "x**2.0"

        # Unary
        assert str(Expression(operator='neg', operands=[expr_x]).symbolic) == "-x"
        assert str(Expression(operator='inv', operands=[expr_x]).symbolic) == "1/x"
        assert str(Expression(operator='sqrt', operands=[expr_x]).symbolic) == "sqrt(x)"
        assert str(Expression(operator='log', operands=[expr_x]).symbolic) == "log(x)"
        assert str(Expression(operator='exp', operands=[expr_x]).symbolic) == "exp(x)"
        assert str(Expression(operator='sin', operands=[expr_x]).symbolic) == "sin(x)"
        assert str(Expression(operator='cos', operands=[expr_x]).symbolic) == "cos(x)"

        # Calculus
        assert str(Expression(operator='diff', operands=[Expression(operator='**', operands=[expr_x, expr_2]), x]).symbolic) == "2*x"
        assert str(Expression(operator='int', operands=[expr_x, x]).symbolic) == "x**2/2" # SymPy default indefinite

        # Special case: unary minus via '-' operator
        assert str(Expression(operator='-', operands=[expr_x]).symbolic) == "-x"


    def test_to_sympy_division_by_zero(self, variables):
        """Test _to_sympy for division by zero (should return sp.nan)."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        expr_zero = Expression(operator='const', operands=[0.0])
        expr_div_zero = Expression(operator='/', operands=[expr_x, expr_zero])
        assert expr_div_zero.symbolic == sp.nan

    def test_to_sympy_inv_zero(self, variables):
        """Test _to_sympy for inv(0) (should return sp.nan)."""
        expr_zero = Expression(operator='const', operands=[0.0])
        expr_inv_zero = Expression(operator='inv', operands=[expr_zero])
        assert expr_inv_zero.symbolic == sp.nan

    def test_to_sympy_arity_errors(self, variables):
        """Test _to_sympy for operator arity errors."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        with pytest.raises(ValueError, match="Division operator '/' expects 2 operands."):
            Expression(operator='/', operands=[expr_x])
        with pytest.raises(ValueError, match="'sin' operator expects 1 operand."):
            Expression(operator='sin', operands=[expr_x, expr_x])
        with pytest.raises(ValueError, match="'diff' operator expects 2 operands"):
            Expression(operator='diff', operands=[expr_x])
        with pytest.raises(ValueError, match="Second operand for 'diff' must be a Variable object or its SymPy symbol."):
            Expression(operator='diff', operands=[expr_x, Expression('const', [1])])


    def test_to_sympy_unknown_operator(self, variables, capsys):
        """Test _to_sympy fallback for unknown operators."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        expr_unknown = Expression(operator='myCustomFunc', operands=[expr_x])
        assert str(expr_unknown.symbolic) == "Mycustomfunc(x)" # SymPy capitalizes unknown functions
        captured = capsys.readouterr()
        assert "Warning: Unknown operator 'myCustomFunc'. Treating as SymPy Function." in captured.out


    def test_expression_evaluate(self, variables):
        """Test numerical evaluation of expressions."""
        x, y = variables["x"], variables["y"]
        expr_x = Expression(operator='var', operands=[x])
        expr_const_5 = Expression(operator='const', operands=[5.0])
        expr_add = Expression(operator='+', operands=[expr_x, expr_const_5]) # x + 5

        data = {"x": 2.0, "y": 3.0}
        assert expr_add.evaluate(data) == pytest.approx(7.0)

        expr_y = Expression(operator='var', operands=[y])
        expr_mul = Expression(operator='*', operands=[expr_y, expr_add]) # y * (x + 5)
        assert expr_mul.evaluate(data) == pytest.approx(21.0)

        expr_sin_x = Expression(operator='sin', operands=[expr_x])
        assert expr_sin_x.evaluate({"x": sp.pi/2}) == pytest.approx(1.0) # Using sp.pi

    def test_expression_evaluate_missing_data(self, variables):
        """Test evaluation with missing variable data."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        expr_add = Expression(operator='+', operands=[expr_x, Expression(operator='const', operands=[5.0])])
        with pytest.raises(ValueError, match="Missing data for variable 'x'"):
            expr_add.evaluate({"z": 1.0})

    def test_expression_evaluate_division_by_zero(self, variables):
        """Test evaluation resulting in division by zero (should be nan)."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        expr_y = Expression(operator='var', operands=[variables["y"]])
        expr_div = Expression(operator='/', operands=[expr_x, expr_y])
        result = expr_div.evaluate({"x": 5.0, "y": 0.0})
        assert result != result # Standard way to check for NaN

    def test_expression_evaluate_domain_error(self, variables):
        """Test evaluation with domain error (e.g. log(-1)) (should be nan)."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        expr_log = Expression(operator='log', operands=[expr_x])
        result = expr_log.evaluate({"x": -1.0})
        assert result != result # Check for NaN

    def test_expression_clone(self, variables):
        """Test cloning of expressions."""
        expr_x = Expression(operator='var', operands=[variables["x"]])
        expr_const_5 = Expression(operator='const', operands=[5.0])
        original_expr = Expression(operator='+', operands=[expr_x, expr_const_5]) # x + 5

        cloned_expr = original_expr.clone()

        assert cloned_expr is not original_expr, "Cloned expression should be a new object."
        assert cloned_expr == original_expr, "Cloned expression should be structurally equal to original."
        assert cloned_expr.symbolic == original_expr.symbolic
        assert cloned_expr.complexity == original_expr.complexity

        # Check that operands are also cloned (for Expression operands) or shared (for Variable/primitives)
        assert cloned_expr.operands[0] is original_expr.operands[0] # Variable 'x' should be shared
        assert cloned_expr.operands[1] is not original_expr.operands[1] # const_5 is an Expression, should be cloned
        assert cloned_expr.operands[1] == original_expr.operands[1] # Cloned const_5 should be equal

        # Modify cloned to ensure original is not affected (deep copy check for nested Expressions)
        # original_expr was x + (const 5.0). cloned_expr.operands[1] is the cloned (const 5.0) Expression.
        # Let's change the constant value in the cloned expression.
        # This requires operands of 'const' to be mutable if we change them directly.
        # The Expression class design makes operands a list, so we can replace elements.
        cloned_expr.operands[1] = Expression(operator='const', operands=[10.0])
        # Must re-initialize post-init fields for the clone if we manually change operands list like this
        # A better way for testing would be to modify a *nested* Expression that was cloned.
        # Let's try that.

        nested_original = Expression(operator='*', operands=[
            variables["y"],
            Expression(operator='+', operands=[expr_x, Expression(operator='const', operands=[3.0])])
        ]) # y * (x + 3.0)

        cloned_nested = nested_original.clone()
        # cloned_nested.operands[1] is the Expression for (x + 3.0)
        # cloned_nested.operands[1].operands[1] is the Expression for (const 3.0)
        cloned_nested.operands[1].operands[1] = Expression(operator='const', operands=[30.0])
        # Re-trigger __post_init__ for the modified parts of the clone.
        # The current .clone() creates new Expression objects, so their __post_init__ runs.
        # If we modify operands of a *cloned* Expression, its symbolic/complexity might become stale
        # UNLESS the modification itself is done via creating new Expression objects or the setter
        # for operands triggers a re-calculation.
        # The current Expression class calculates symbolic/complexity in __post_init__.
        # Modifying `cloned_nested.operands[1].operands[1] = ...` replaces an Expression object.
        # The parent, `cloned_nested.operands[1]`, still holds its old symbolic form.
        # This highlights a potential issue if Expressions are mutated after creation without care.
        # However, clone() itself should produce a valid, independent new Expression.
        # The test here is more about whether the clone is independent.

        # To properly test independence after modification, we need to ensure the clone's state is updated.
        # One way: re-run post_init on the modified part, or reconstruct the part.
        # For this test, let's verify the original is unchanged.
        # The critical part of clone is that it creates *new* Expression objects for nested Expr.

        assert str(nested_original.symbolic) == "y*(x + 3.0)", "Original should not change."
        # If clone was perfect and we could easily recalculate, cloned_nested symbolic would be y*(x + 30.0)
        # The issue is modifying parts of a cloned expression and expecting parent parts to auto-update.
        # The clone() method as written *does* create new instances, so initial clone is fine.
        # The problem is if we then mutate the *structure* of the clone by replacing its operand objects.
        # Let's test if the original's operand (an Expression) is not the same object as the clone's operand.
        assert nested_original.operands[1] is not cloned_nested.operands[1], "Nested expression should be cloned, not shared."


    def test_expression_hash_and_eq(self, variables):
        """Test hashing and equality for Expressions."""
        x = variables["x"]
        expr_x = Expression(operator='var', operands=[x])
        expr_const_5_A = Expression(operator='const', operands=[5.0])
        expr_const_5_B = Expression(operator='const', operands=[5.0])

        expr1_A = Expression(operator='+', operands=[expr_x, expr_const_5_A]) # x + 5.0
        expr1_B = Expression(operator='+', operands=[expr_x, expr_const_5_B]) # x + 5.0 (structurally same)
        expr2   = Expression(operator='+', operands=[expr_x, Expression(operator='const', operands=[6.0])]) # x + 6.0
        expr3   = Expression(operator='*', operands=[expr_x, expr_const_5_A]) # x * 5.0

        assert expr_const_5_A == expr_const_5_B
        assert hash(expr_const_5_A) == hash(expr_const_5_B)

        assert expr1_A == expr1_B
        assert hash(expr1_A) == hash(expr1_B)

        assert expr1_A != expr2
        assert expr1_A != expr3

        # Test with different variable instance but same name/index
        x_prime = Variable(name="x", index=0) # Same as variables["x"] effectively
        expr_x_prime = Expression(operator='var', operands=[x_prime])
        expr1_C = Expression(operator='+', operands=[expr_x_prime, expr_const_5_A])
        assert expr1_A == expr1_C # Should be equal due to Variable equality
        assert hash(expr1_A) == hash(expr1_C)


    def test_expression_str_and_repr(self, variables):
        """Test __str__ and __repr__ for Expressions."""
        x = variables["x"]
        expr_x = Expression(operator='var', operands=[x])
        expr_const_5 = Expression(operator='const', operands=[5.0])
        expr_add = Expression(operator='+', operands=[expr_x, expr_const_5]) # x + 5.0

        assert str(expr_add) == "x + 5.0"
        # Repr will be long. Check key parts.
        # Expression(operator='var', operands=[Variable(name='x', index=0, properties={})])
        # Expression(operator='const', operands=[5.0])
        # Expression(operator='+', operands=[Expression(operator='var', operands=[Variable(name='x', index=0, properties={})]), Expression(operator='const', operands=[5.0])])

        expected_repr_x = "Expression(operator='var', operands=[Variable(name='x', index=0, properties={})])"
        expected_repr_const_5 = "Expression(operator='const', operands=[5.0])"
        expected_repr_add = f"Expression(operator='+', operands=[{expected_repr_x}, {expected_repr_const_5}])"

        assert repr(expr_x) == expected_repr_x
        assert repr(expr_const_5) == expected_repr_const_5
        assert repr(expr_add) == expected_repr_add

    def test_expression_complexity_nested(self, variables):
        """Test complexity calculation for deeply nested expressions."""
        x = variables["x"]
        # expr = sin(x + (const 1))
        # C(const 1) = 1(op) + 1(val) = 2
        # C(var x) = 1(op) + 1(var) = 2
        # C(x + (const 1)) = 1(op '+') + C(var x) + C(const 1) = 1 + 2 + 2 = 5
        # C(sin(x + (const 1))) = 1(op 'sin') + C(x + (const 1)) = 1 + 5 = 6
        expr = Expression(operator='sin', operands=[
            Expression(operator='+', operands=[
                Expression(operator='var', operands=[x]),
                Expression(operator='const', operands=[1.0])
            ])
        ])
        assert expr.complexity == 6

    def test_var_operand_types_in_to_sympy(self, variables):
        """Test 'var' operator with Variable object and string name."""
        var_obj = variables["x"]
        expr_from_obj = Expression(operator='var', operands=[var_obj])
        expr_from_name = Expression(operator='var', operands=["x_name"])
        expr_from_symbol = Expression(operator='var', operands=[sp.Symbol("x_sym")])

        assert str(expr_from_obj.symbolic) == "x"
        assert str(expr_from_name.symbolic) == "x_name"
        assert str(expr_from_symbol.symbolic) == "x_sym"

        with pytest.raises(ValueError, match="Invalid operand for 'var' operator"):
            Expression(operator='var', operands=[123])

    def test_calculus_operand_validation(self, variables):
        """Test that calculus operators require a Variable or its symbol as the second operand."""
        x_expr = Expression(operator='var', operands=[variables["x"]])

        # Valid: Variable object
        expr_diff_var_obj = Expression(operator='diff', operands=[x_expr, variables["x"]])
        assert str(expr_diff_var_obj.symbolic) == "1" # d(x)/dx = 1

        # Valid: Sympy Symbol (from variable.symbolic)
        expr_diff_var_sym = Expression(operator='diff', operands=[x_expr, variables["x"].symbolic])
        assert str(expr_diff_var_sym.symbolic) == "1"

        # Invalid: Non-variable/symbol
        with pytest.raises(ValueError, match="Second operand for 'diff' must be a Variable object or its SymPy symbol."):
            Expression(operator='diff', operands=[x_expr, Expression(operator='const', operands=[1.0])])

        with pytest.raises(ValueError, match="Second operand for 'int' must be a Variable object or its SymPy symbol."):
            Expression(operator='int', operands=[x_expr, "string_not_var_symbol"])
