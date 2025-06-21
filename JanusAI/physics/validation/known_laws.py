"""
Known Physics Laws
==================

Defines a library of known physical laws, represented symbolically and
with associated metadata, for the purpose of validating discovered hypotheses.
"""

import sympy as sp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Import Expression and Variable for comparison with discovered laws
from janus_ai.core.expressions.expression import Expression, Variable
from janus_ai.core.expressions.symbolic_math import are_expressions_equivalent_sympy


@dataclass
class KnownLaw:
    """
    Represents a known physical law with its symbolic form and metadata.
    """
    name: str
    formula_str: str # The standard string representation of the law (e.g., "m*a")
    variables: List[str] # Expected variables in the formula (e.g., ["m", "a"])
    domain: str # E.g., "mechanics", "thermodynamics"
    description: str = ""
    # SymPy representation of the formula for robust comparison
    formula_sympy: sp.Expr = field(init=False)

    def __post_init__(self):
        self.formula_sympy = sp.sympify(self.formula_str)

    def is_equivalent(self, discovered_expression: Union[str, Expression, sp.Expr],
                     tolerance: float = 1e-6) -> bool:
        """
        Checks if a discovered expression is mathematically equivalent to this known law.

        Args:
            discovered_expression: The expression discovered by the agent. Can be a string,
                                   an `Expression` object, or a SymPy expression.
            tolerance: Numerical tolerance for equivalence checking.

        Returns:
            True if the expressions are equivalent, False otherwise.
        """
        if isinstance(discovered_expression, Expression):
            discovered_sympy = discovered_expression.symbolic
        elif isinstance(discovered_expression, str):
            discovered_sympy = sp.sympify(discovered_expression)
        elif isinstance(discovered_expression, sp.Expr):
            discovered_sympy = discovered_expression
        else:
            raise TypeError("Discovered expression must be a string, Expression, or SymPy object.")
        
        # Ensure the comparison is done with respect to the relevant variables
        # Extract variables involved in both known and discovered laws
        known_symbols = self.formula_sympy.free_symbols
        discovered_symbols = discovered_sympy.free_symbols
        
        # Combine all relevant symbols for the equivalence check
        all_symbols = list(known_symbols.union(discovered_symbols))

        return are_expressions_equivalent_sympy(self.formula_sympy, discovered_sympy, all_symbols, tolerance)


class KnownLawLibrary:
    """
    A collection of predefined physical laws for validation and benchmarking.
    """
    def __init__(self):
        self._laws: List[KnownLaw] = self._create_default_laws()
        self._laws_by_name: Dict[str, KnownLaw] = {law.name: law for law in self._laws}

    def _create_default_laws(self) -> List[KnownLaw]:
        """
        Populates the library with common known physical laws.
        """
        laws = []
        
        # Mechanics
        laws.append(KnownLaw(
            name="Newton's Second Law",
            formula_str="m*a",
            variables=["m", "a"],
            domain="mechanics",
            description="Force equals mass times acceleration."
        ))
        laws.append(KnownLaw(
            name="Kinetic Energy",
            formula_str="0.5 * m * v**2",
            variables=["m", "v"],
            domain="mechanics",
            description="Energy of motion."
        ))
        laws.append(KnownLaw(
            name="Potential Energy (Gravity near Earth)",
            formula_str="m * g * h",
            variables=["m", "g", "h"],
            domain="mechanics",
            description="Gravitational potential energy."
        ))
        laws.append(KnownLaw(
            name="Hooke's Law (Spring Force)",
            formula_str="-k * x",
            variables=["k", "x"],
            domain="mechanics",
            description="Force exerted by a spring."
        ))
        laws.append(KnownLaw(
            name="Harmonic Oscillator Energy",
            formula_str="0.5 * m * v**2 + 0.5 * k * x**2",
            variables=["m", "v", "k", "x"],
            domain="mechanics",
            description="Total energy of a simple harmonic oscillator."
        ))
        laws.append(KnownLaw(
            name="Momentum (Linear)",
            formula_str="m * v",
            variables=["m", "v"],
            domain="mechanics",
            description="Linear momentum of a body."
        ))
        laws.append(KnownLaw(
            name="Coulomb's Law (Force Magnitude)",
            formula_str="k_e * q1 * q2 / r**2",
            variables=["k_e", "q1", "q2", "r"],
            domain="electromagnetism",
            description="Magnitude of electrostatic force between two point charges."
        ))
        
        # Thermodynamics
        laws.append(KnownLaw(
            name="Ideal Gas Law",
            formula_str="P * V / (n * T)",
            variables=["P", "V", "n", "T"],
            domain="thermodynamics",
            description="Relates pressure, volume, temperature, and amount of gas."
        ))
        
        return laws

    def get_law_by_name(self, name: str) -> Optional[KnownLaw]:
        """Retrieves a known law by its name."""
        return self._laws_by_name.get(name)

    def get_all_laws(self, domain: Optional[str] = None) -> List[KnownLaw]:
        """Retrieves all known laws, optionally filtered by domain."""
        if domain:
            return [law for law in self._laws if law.domain == domain]
        return list(self._laws)

    def describe_laws(self) -> str:
        """Returns a string summary of the laws in the library."""
        summary = [f"Known Law Library ({len(self._laws)} laws):"]
        for law in self._laws:
            summary.append(f"- {law.name} ({law.domain}): {law.formula_str}")
        return "\n".join(summary)


if __name__ == "__main__":
    # Mock evaluate_expression_on_data and are_expressions_equivalent_sympy if not fully available
    try:
        pass
        # from janus.core.expressions.symbolic_math import are_expressions_equivalent_sympy as real_are_eq
    except ImportError:
        print("Using mock are_expressions_equivalent_sympy for known_laws.py test.")
        def are_expressions_equivalent_sympy(expr1: sp.Expr, expr2: sp.Expr, symbols: List[sp.Symbol], tolerance: float) -> bool:
            """Simple mock: considers expressions equivalent if their string forms are identical."""
            return str(expr1) == str(expr2)

    # Mock Expression and Variable classes if needed for `KnownLaw.is_equivalent` tests
    try:
        from janus.core.expressions.expression import Expression as RealExpression, Variable as RealVariable
    except ImportError:
        print("Using mock Expression and Variable for known_laws.py test.")
        # Minimal mocks
        @dataclass(eq=True, frozen=False)
        class RealVariable:
            name: str
            index: int
            properties: Dict[str, Any] = field(default_factory=dict)
            symbolic: sp.Symbol = field(init=False)
            def __post_init__(self): self.symbolic = sp.Symbol(self.name)
            def __hash__(self): return hash((self.name, self.index))
            def __str__(self): return self.name

        @dataclass(eq=False, frozen=False)
        class RealExpression:
            operator: str
            operands: List[Any]
            _symbolic: Optional[sp.Expr] = field(init=False, repr=False)
            
            def __post_init__(self):
                # Attempt to create a symbolic representation
                if self.operator == 'var' and isinstance(self.operands[0], RealVariable):
                    self._symbolic = self.operands[0].symbolic
                elif self.operator == 'const':
                    self._symbolic = sp.Float(self.operands[0])
                else:
                    self._symbolic = sp.sympify(self.operator + "(" + ",".join([str(op) for op in self.operands]) + ")")
            @property
            def symbolic(self) -> sp.Expr: return self._symbolic
            def __str__(self) -> str: return str(self.symbolic)

    Expression = RealExpression
    Variable = RealVariable


    print("--- Testing KnownLaw Library ---")
    law_library = KnownLawLibrary()

    print("\n--- All Known Laws ---")
    print(law_library.describe_laws())

    # Test retrieving a specific law
    newton_law = law_library.get_law_by_name("Newton's Second Law")
    print(f"\nRetrieved law by name 'Newton's Second Law': {newton_law.name}, Formula: {newton_law.formula_str}")
    assert newton_law is not None
    assert str(newton_law.formula_sympy) == "a*m" # SymPy reorders alphabetically

    # Test equivalence check
    print("\n--- Testing Law Equivalence ---")
    
    # Create variables for testing discovered expressions
    m_sym, a_sym, v_sym, k_sym, x_sym, h_sym, g_sym, P_sym, V_sym, n_sym, T_sym = \
        sp.symbols('m a v k x h g P V n T')

    # Scenario 1: Correctly discovered Newton's Second Law
    discovered_newton_str = "a * m"
    is_equivalent_newton = newton_law.is_equivalent(discovered_newton_str)
    print(f"Is '{discovered_newton_str}' equivalent to Newton's Second Law? {is_equivalent_newton}")
    assert is_equivalent_newton == True

    # Scenario 2: Equivalent but rearranged form
    discovered_newton_rearranged_str = "m * a"
    is_equivalent_rearranged = newton_law.is_equivalent(discovered_newton_rearranged_str)
    print(f"Is '{discovered_newton_rearranged_str}' equivalent to Newton's Second Law (rearranged)? {is_equivalent_rearranged}")
    assert is_equivalent_rearranged == True

    # Scenario 3: Incorrect law
    discovered_incorrect_str = "m + a"
    is_not_equivalent = newton_law.is_equivalent(discovered_incorrect_str)
    print(f"Is '{discovered_incorrect_str}' equivalent to Newton's Second Law? {is_not_equivalent}")
    assert is_not_equivalent == False

    # Test with an Expression object
    # Representing "0.5 * m * v**2" as an Expression object
    var_m_obj = Variable("m", 0)
    var_v_obj = Variable("v", 1)
    
    discovered_ke_expr_obj = Expression(operator='*', operands=[
        Expression(operator='const', operands=[0.5]),
        var_m_obj,
        Expression(operator='**', operands=[var_v_obj, Expression(operator='const', operands=[2])])
    ])
    
    kinetic_energy_law = law_library.get_law_by_name("Kinetic Energy")
    is_equivalent_ke_obj = kinetic_energy_law.is_equivalent(discovered_ke_expr_obj)
    print(f"Is '{discovered_ke_expr_obj}' equivalent to Kinetic Energy (as Expression obj)? {is_equivalent_ke_obj}")
    assert is_equivalent_ke_obj == True

    # Test a complex law (Harmonic Oscillator Energy)
    ho_energy_law = law_library.get_law_by_name("Harmonic Oscillator Energy")
    discovered_ho_energy_str = "0.5*m*v**2 + 0.5*k*x**2"
    is_equivalent_ho = ho_energy_law.is_equivalent(discovered_ho_energy_str)
    print(f"Is '{discovered_ho_energy_str}' equivalent to Harmonic Oscillator Energy? {is_equivalent_ho}")
    assert is_equivalent_ho == True

    print("\nAll KnownLawLibrary tests completed.")

