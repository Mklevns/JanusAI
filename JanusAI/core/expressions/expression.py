"""
Expression and Variable Classes
===============================

Defines the fundamental building blocks for symbolic expressions within the
Janus framework: `Variable` for representing independent variables, and
`Expression` for representing mathematical operations and their operands.
"""

import sympy as sp
from typing import List, Any, Optional, Dict
from dataclasses import dataclass, field

# Forward declaration for type hinting (needed if classes reference each other
# before full definition, though Python 3.7+ often handles this with string literals)
# Using these stubs for clarity as per previous prompt response.
# class Variable: pass
# class Expression: pass


@dataclass(eq=True, frozen=False)
class Variable:
    """
    Represents a fundamental variable in a symbolic expression or environment.

    Each variable has a name, an index (e.g., its column in an observation matrix),
    and can carry semantic properties (e.g., 'is_constant', 'value').
    It also holds a SymPy Symbol object for direct symbolic manipulation.

    Attributes:
        name (str): The string name of the variable (e.g., "x", "t", "mass").
        index (int): The integer index associated with the variable, often
                     corresponding to its position in a data array or observation.
        properties (Dict[str, float]): A dictionary of arbitrary semantic properties,
                                       e.g., {'is_constant': True, 'value': 9.81}.
        symbolic (sp.Symbol): A SymPy Symbol instance derived from the name.
                              This is initialized after object creation.
    """
    name: str
    index: int  # Column in observation matrix
    properties: Dict[str, float] = field(default_factory=dict)
    symbolic: sp.Symbol = field(init=False) # SymPy symbol for this variable

    def __post_init__(self):
        """Initializes the SymPy symbolic representation after object creation."""
        # Ensure the name is valid for a SymPy Symbol
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"Variable name '{self.name}' must be a valid Python identifier.")
        self.symbolic = sp.Symbol(self.name)

    def __hash__(self):
        """
        Custom hash method to allow Variable objects to be used in sets or as
        dictionary keys. Hashing is based on `name` and `index`.
        """
        # Note: If `eq=True` by dataclass default compares all fields,
        # but `__hash__` only uses `name` and `index`, then `Variable("x",0,{})`
        # and `Variable("x",0,{"unit":"m"})` will have the same hash but be
        # considered unequal by default `__eq__`. This is generally fine for sets,
        # but one should be aware of `__eq__` behavior if properties are critical for equality.
        return hash((self.name, self.index))

    def __str__(self):
        """Returns the string representation of the variable (its name)."""
        return self.name

    def __repr__(self):
        """Returns a detailed string representation for debugging."""
        return f"Variable(name='{self.name}', index={self.index}, properties={self.properties})"

    @property
    def complexity(self) -> int:
        """
        Defines the inherent complexity of a variable.
        For a base variable, complexity is typically minimal (e.g., 1).
        """
        return 1

# Using @dataclass for Expression, as provided in the user's latest query
@dataclass(eq=False, frozen=False) # eq=False because we'll implement custom __eq__ based on symbolic form or deeper structural comparison
class Expression:
    """
    Represents a mathematical expression as a tree structure.

    This dataclass handles various operators and their operands, computes its
    complexity, and converts itself into a SymPy expression for advanced
    mathematical operations and evaluation.

    Attributes:
        operator (str): The string representing the mathematical operator
                        (e.g., '+', '-', '*', '/', 'sin', 'cos', 'var', 'const').
        operands (List[Any]): A list of operands for the operator. These can be
                              other `Expression` objects, `Variable` objects, or
                              primitive types like `int` or `float` (for constants).
        complexity (int): The calculated complexity of the expression (computed in post_init).
        symbolic (Optional[sp.Expr]): The SymPy representation of the expression (computed in post_init).
    """
    operator: str
    operands: List[Any]  # Can be Expression, Variable, or Constant
    _complexity: int = field(init=False, repr=False) # Internal, calculated field
    _symbolic: Optional[sp.Expr] = field(init=False, repr=False) # Internal, calculated field

    def __post_init__(self):
        """
        Initializes calculated fields after the dataclass is constructed.
        Computes complexity and the SymPy representation of the expression.
        """
        # It's important that _to_sympy can handle partially initialized operands
        # if operands are also Expression objects that rely on _to_sympy.
        # This typically means _to_sympy should recurse.
        self._grammar_context = None # Initialize grammar context
        
        # Calculate symbolic representation first, as complexity might depend on its structure
        self._symbolic = self._to_sympy()
        # Calculate complexity
        self._complexity = self._compute_complexity()

    @property
    def symbolic(self) -> sp.Expr:
        """Returns the cached SymPy expression."""
        return self._symbolic

    @property
    def complexity(self) -> int:
        """Returns the cached complexity of the expression."""
        return self._complexity

    def _compute_complexity(self) -> int:
        """
        Calculates the complexity of the expression using an MDL-inspired approach
        (Minimum Description Length) by counting nodes in the expression tree.
        """
        # The complexity of the operator node itself is 1.
        current_complexity = 1
        
        # Recursively sum complexities of operands.
        for op in self.operands:
            if isinstance(op, Expression):
                current_complexity += op.complexity # Recurse for nested expressions
            elif isinstance(op, Variable):
                current_complexity += op.complexity # Use Variable's complexity (typically 1)
            else:
                # Assumed to be a primitive constant (int, float). Its complexity is 1.
                current_complexity += 1
        return current_complexity

    def _to_sympy(self) -> sp.Expr:
        """
        Converts the Expression object into its corresponding SymPy expression.
        Handles various operators including arithmetic, unary functions, and
        special cases like variables ('var') and constants ('const').
        """
        # Check if we have a grammar context that can handle AI operators
        if hasattr(self, '_grammar_context') and self._grammar_context is not None and \
           hasattr(self._grammar_context, 'convert_ai_expression_to_sympy'):
            try:
                return self._grammar_context.convert_ai_expression_to_sympy(self)
            except (NotImplementedError, AttributeError):
                pass  # Fall back to original method

        # Original _to_sympy implementation
        # Handle 'var' and 'const' special operators (leaf nodes or direct values)
        if self.operator == 'var':
            # Operand should be a variable name string or a Variable object
            op_val = self.operands[0]
            if isinstance(op_val, sp.Symbol): # Already a sympy symbol
                return op_val
            elif isinstance(op_val, Variable): # If Variable object is passed
                return op_val.symbolic
            elif isinstance(op_val, str): # If name string is passed
                return sp.Symbol(op_val)
            else:
                raise ValueError(f"Invalid operand for 'var' operator: {op_val} (type: {type(op_val)})")
        elif self.operator == 'const':
            return sp.Float(self.operands[0]) # Convert Python number to SymPy Float

        # Process operands recursively to get their SymPy representations
        # `arg` will hold the SymPy expression for each operand
        args = []
        for op in self.operands:
            if isinstance(op, Expression):
                args.append(op.symbolic) # Get symbolic form of nested expression
            elif isinstance(op, Variable):
                args.append(op.symbolic) # Get symbolic form of Variable
            else:
                # Assume raw constants (int/float)
                args.append(sp.Float(op) if isinstance(op, (float, int)) else op)
        
        # Handle binary and N-ary operators
        if self.operator == '+':
            return sp.Add(*args)
        elif self.operator == '-':
            if len(args) == 1: # Unary minus (e.g., -x)
                return -args[0]
            return sp.Add(args[0], sp.Mul(-1, *args[1:])) # a - b - c => a + (-b) + (-c)
        elif self.operator == '*':
            return sp.Mul(*args)
        elif self.operator == '/':
            if len(args) != 2:
                raise ValueError("Division operator '/' expects 2 operands.")
            numerator = args[0]
            denominator = args[1]
            # Handle division by zero explicitly using SymPy's nan
            if denominator == 0 or (hasattr(denominator, 'is_zero') and denominator.is_zero is True):
                return sp.nan
            return numerator / denominator
        elif self.operator == '**':
            if len(args) != 2:
                raise ValueError("Power operator '**' expects 2 operands.")
            return args[0] ** args[1]

        # Handle unary functions
        elif self.operator == 'neg':
            if len(args) != 1: raise ValueError("'neg' operator expects 1 operand.")
            return -args[0]
        elif self.operator == 'inv':
            if len(args) != 1: raise ValueError("'inv' operator expects 1 operand.")
            arg = args[0]
            if arg == 0 or (hasattr(arg, 'is_zero') and arg.is_zero is True):
                return sp.nan
            return 1 / arg
        elif self.operator == 'sqrt':
            if len(args) != 1: raise ValueError("'sqrt' operator expects 1 operand.")
            return sp.sqrt(args[0])
        elif self.operator == 'log':
            if len(args) != 1: raise ValueError("'log' operator expects 1 operand.")
            return sp.log(args[0])
        elif self.operator == 'exp':
            if len(args) != 1: raise ValueError("'exp' operator expects 1 operand.")
            return sp.exp(args[0])
        elif self.operator == 'sin':
            if len(args) != 1: raise ValueError("'sin' operator expects 1 operand.")
            return sp.sin(args[0])
        elif self.operator == 'cos':
            if len(args) != 1: raise ValueError("'cos' operator expects 1 operand.")
            return sp.cos(args[0])

        # Handle calculus operators
        elif self.operator == 'diff':
            if len(args) != 2: raise ValueError("'diff' operator expects 2 operands: (expression, variable_to_differentiate_with_respect_to).")
            expr_sym, var_sym = args
            # Ensure the second operand is a SymPy Symbol (from a Variable object)
            if not isinstance(var_sym, sp.Symbol):
                raise ValueError("Second operand for 'diff' must be a Variable object or its SymPy symbol.")
            return sp.diff(expr_sym, var_sym)
        elif self.operator == 'int':
            if len(args) != 2: raise ValueError("'int' operator expects 2 operands: (expression, variable_to_integrate_with_respect_to).")
            expr_sym, var_sym = args
            # Ensure the second operand is a SymPy Symbol (from a Variable object)
            if not isinstance(var_sym, sp.Symbol):
                raise ValueError("Second operand for 'int' must be a Variable object or its SymPy symbol.")
            return sp.integrate(expr_sym, var_sym)

        # Fallback for unknown operators: treat as an undefined SymPy Function
        # This allows handling custom functions not explicitly defined above.
        print(f"Warning: Unknown operator '{self.operator}'. Treating as SymPy Function.")
        return sp.Function(self.operator.capitalize())(*args)

    def evaluate(self, data: Dict[str, Any]) -> Any:
        """
        Numerically evaluates the expression given a dictionary of variable values.
        Uses the cached SymPy representation.

        Args:
            data (Dict[str, Any]): A dictionary where keys are variable names (strings)
                                   and values are their numerical assignments.

        Returns:
            Any: The numerical result of the evaluation. Returns `float('nan')`
                 if evaluation fails (e.g., division by zero, domain errors).

        Raises:
            ValueError: If data for a required variable is missing.
        """
        # Collect all free symbols (variables) in the expression
        free_symbols = list(self.symbolic.free_symbols)
        
        # Create a mapping from SymPy symbols to their numerical values from the input data
        subs_dict = {}
        for sym in free_symbols:
            if str(sym) in data:
                subs_dict[sym] = data[str(sym)]
            else:
                raise ValueError(f"Missing data for variable '{sym.name}' in evaluation of expression '{self}'.")
        
        # Substitute values and numerically evaluate
        try:
            # Use `evalf` for numerical evaluation after substitution
            result = self.symbolic.evalf(subs=subs_dict)
            # Convert SymPy number to Python float. Handle complex results or NaNs.
            if isinstance(result, (sp.Number, int, float)):
                return float(result)
            elif result.is_comparable is False or result.is_finite is False: # e.g. infinity, NaN, unevaluated complex
                 return float('nan')
            else: # Other SymPy types (e.g. complex, but also could be custom functions if not evalf'd)
                 return float(result.as_real_imag()[0]) # Take real part if complex, or handle as NaN
        except (TypeError, ValueError, sp.SympifyError, AttributeError) as e:
            # Handle cases where evaluation fails (e.g., division by zero, invalid args, non-numeric result)
            print(f"Warning: Numerical evaluation failed for expression '{self}' with data {data}: {e}")
            return float('nan') # Return NaN for failed evaluations


    def clone(self) -> 'Expression':
        """
        Creates a deep copy of the expression tree.
        Ensures that nested expressions are also cloned, while Variables are shared
        (as they are typically immutable or globally unique references).
        """
        cloned_operands = []
        for op in self.operands:
            if isinstance(op, Expression):
                cloned_operands.append(op.clone()) # Recursively clone nested expressions
            elif isinstance(op, Variable):
                cloned_operands.append(op) # Variables are typically immutable; share reference
            else:
                # Primitive types like numbers or strings are immutable; share reference
                cloned_operands.append(op)
        # Create a new Expression instance with cloned operands.
        # __post_init__ will automatically re-calculate complexity and symbolic representation.
        return Expression(operator=self.operator, operands=cloned_operands)

    def __hash__(self):
        """
        Computes a hash for the expression, enabling its use in sets and as dictionary keys.
        The hash is based on the operator and the hashes of its operands.
        """
        operand_hashes = []
        for op in self.operands:
            if isinstance(op, (Expression, Variable)):
                operand_hashes.append(hash(op)) # Rely on child objects' __hash__
            elif isinstance(op, (int, float, str, bool, sp.Number)): # Commonly hashable primitives and SymPy numbers
                operand_hashes.append(hash(op))
            else:
                # Fallback for other potentially unhashable types, converting to string representation.
                # Be cautious with this, as different objects might have the same string representation.
                operand_hashes.append(hash(str(op)))
        return hash((self.operator, tuple(operand_hashes)))

    def __eq__(self, other: Any) -> bool:
        """
        Compares two Expression objects for structural equality.
        Two expressions are equal if they have the same operator and
        their operands are pairwise equal.
        """
        if not isinstance(other, Expression):
            return False
        return self.operator == other.operator and self.operands == other.operands

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the expression.
        Uses SymPy's string representation for the underlying symbolic form.
        """
        return str(self.symbolic)

    def __repr__(self) -> str:
        """
        Returns a detailed string representation for debugging, including operator and operands.
        """
        operands_repr = ", ".join([repr(op) for op in self.operands])
        return f"Expression(operator='{self.operator}', operands=[{operands_repr}])"

    def set_grammar_context(self, grammar):
        """Set grammar context for enhanced symbolic conversion."""
        self._grammar_context = grammar
        # Optionally, re-calculate symbolic representation if context changes
        # self._symbolic = self._to_sympy()
        # self._complexity = self._compute_complexity() # And complexity if it depends on symbolic form


if __name__ == "__main__":
    # --- Variable Class Tests (retained from previous iteration, ensuring consistency) ---
    print("--- Testing Variable Class ---")
    var_x = Variable(name="x", index=0)
    var_y_const = Variable(name="y", index=1, properties={"is_constant": True, "value": 5.0})
    var_z = Variable("z", 2)

    print(f"var_x: {var_x}")
    print(f"var_y_const: {var_y_const}")
    print(f"var_z: {var_z}")
    print(f"Type of var_x.symbolic: {type(var_x.symbolic)}")
    print(f"var_y_const properties: {var_y_const.properties}")
    print(f"Complexity of var_x: {var_x.complexity}")

    # Test equality and hashing for Variable
    var_x_2 = Variable("x", 0)
    var_x_diff_prop = Variable("x", 0, properties={"unit": "m"})
    var_x_diff_idx = Variable("x", 1)

    print(f"var_x == var_x_2: {var_x == var_x_2}") # True
    print(f"var_x is var_x_2: {var_x is var_x_2}") # False
    # As per dataclass `eq=True`, properties are compared unless custom __eq__ on Variable.
    # Our custom __hash__ for Variable only uses name and index.
    # Default dataclass __eq__ compares all fields. So var_x != var_x_diff_prop
    print(f"var_x == var_x_diff_prop: {var_x == var_x_diff_prop}") # False (due to differing 'properties' field)
    print(f"var_x == var_x_diff_idx: {var_x == var_x_diff_idx}") # False

    var_set = {var_x, var_y_const, var_x_2, var_x_diff_prop}
    # Expected set size: var_x (differs from var_x_diff_prop by props), var_y_const.
    # var_x_2 is equal to var_x. So: {var_x, var_y_const, var_x_diff_prop} -> size 3
    print(f"Set of variables (size should be 3, for x, y, x_diff_prop): {len(var_set)}")
    assert len(var_set) == 3, f"Expected set size 3, got {len(var_set)}"
    print(f"Confirmed set size: {len(var_set)}")


    # --- Expression Class Tests (using the new dataclass implementation) ---
    print("\n--- Testing Expression Class (Dataclass Implementation) ---")

    # Creating leaf nodes
    expr_x = Expression(operator='var', operands=[var_x])
    expr_y = Expression(operator='var', operands=[Variable("y", 1)]) # Can instantiate Variable directly
    expr_const_5 = Expression(operator='const', operands=[5.0])
    expr_const_neg2 = Expression(operator='const', operands=[-2])

    print(f"Expr 'x': {expr_x} (Complexity: {expr_x.complexity})")
    print(f"Expr 'y': {expr_y} (Complexity: {expr_y.complexity})")
    print(f"Expr '5.0': {expr_const_5} (Complexity: {expr_const_5.complexity})")
    print(f"Expr '-2': {expr_const_neg2} (Complexity: {expr_const_neg2.complexity})")

    # Creating composite expressions
    expr_add = Expression(operator='+', operands=[expr_x, expr_const_5]) # x + 5
    expr_mul = Expression(operator='*', operands=[expr_y, expr_add]) # y * (x + 5)
    expr_sub = Expression(operator='-', operands=[expr_mul, expr_const_neg2]) # y * (x + 5) - (-2)

    expr_sin_x = Expression(operator='sin', operands=[expr_x]) # sin(x)
    expr_pow = Expression(operator='**', operands=[expr_x, Expression(operator='const', operands=[2])]) # x**2
    expr_inv = Expression(operator='inv', operands=[expr_y]) # 1/y

    print(f"\nExpr 'x + 5': {expr_add} (Complexity: {expr_add.complexity})")
    print(f"Expr 'y * (x + 5)': {expr_mul} (Complexity: {expr_mul.complexity})")
    print(f"Expr 'y * (x + 5) - (-2)': {expr_sub} (Complexity: {expr_sub.complexity})")
    print(f"Expr 'sin(x)': {expr_sin_x} (Complexity: {expr_sin_x.complexity})")
    print(f"Expr 'x**2': {expr_pow} (Complexity: {expr_pow.complexity})")
    print(f"Expr '1/y': {expr_inv} (Complexity: {expr_inv.complexity})")


    # Test symbolic differentiation and integration
    # Create a variable to differentiate with respect to
    var_to_diff = Variable("x", 0) # Use the same x as in expr_x

    expr_diff = Expression(operator='diff', operands=[expr_pow, var_to_diff]) # diff(x**2, x)
    print(f"Expr 'diff(x**2, x)': {expr_diff} (Complexity: {expr_diff.complexity})")
    assert str(expr_diff.symbolic) == "2*x"

    expr_int = Expression(operator='int', operands=[expr_x, var_to_diff]) # int(x, x)
    print(f"Expr 'int(x, x)': {expr_int} (Complexity: {expr_int.complexity})")
    assert str(expr_int.symbolic) == "x**2/2" # SymPy default for indefinite integral


    # Test cloning
    cloned_expr_mul = expr_mul.clone()
    print(f"\nOriginal expr_mul: {expr_mul}")
    print(f"Cloned expr_mul: {cloned_expr_mul}")
    print(f"expr_mul == cloned_expr_mul: {expr_mul == cloned_expr_mul}") # Should be True (structural equality)
    print(f"expr_mul is cloned_expr_mul: {expr_mul is cloned_expr_mul}") # Should be False (different objects)
    # Modify cloned and check if original changes (it shouldn't)
    if isinstance(cloned_expr_mul.operands[1], Expression):
        cloned_expr_mul.operands[1].operands.append(Expression(operator='const', operands=[100])) # Modifies (x+5) to (x+5+100)
    print(f"Original expr_mul after cloning and modifying clone: {expr_mul}")
    print(f"Cloned expr_mul after modification: {cloned_expr_mul}")
    print(f"expr_mul == cloned_expr_mul after modification: {expr_mul == cloned_expr_mul}") # Should be False


    # Test evaluation
    data_for_eval = {"x": 2.0, "y": 3.0}
    
    eval_add = expr_add.evaluate(data_for_eval) # x + 5 = 2 + 5 = 7
    eval_mul = expr_mul.evaluate(data_for_eval) # y * (x + 5) = 3 * (2 + 5) = 21
    eval_sub = expr_sub.evaluate(data_for_eval) # y * (x + 5) - (-2) = 21 - (-2) = 23

    print(f"\nEvaluation of (x + 5) with x=2: {eval_add}")
    print(f"Evaluation of (y * (x + 5)) with x=2, y=3: {eval_mul}")
    print(f"Evaluation of (y * (x + 5) - (-2)) with x=2, y=3: {eval_sub}")
    print(f"Evaluation of sin(x) with x=2: {expr_sin_x.evaluate({'x': np.pi/2})}") # Should be ~1.0

    # Test division by zero
    expr_div_by_y = Expression(operator='/', operands=[expr_x, expr_y])
    eval_div_by_zero = expr_div_by_y.evaluate({"x": 5.0, "y": 0.0})
    print(f"Evaluation of (x / y) with y=0: {eval_div_by_zero} (expected NaN)")
    assert np.isnan(eval_div_by_zero)


    print("\nAll tests for Expression and Variable classes completed using the dataclass implementation.")

