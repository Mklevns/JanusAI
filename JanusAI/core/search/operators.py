# janus/core/search/operators.py
"""
Genetic operators for symbolic expression evolution.

This module provides genetic operators (crossover, mutation, initialization)
for evolving populations of symbolic expressions. All operators respect
grammar constraints and maintain expression validity.
"""

import copy
import random
import numpy as np
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING
from abc import ABC, abstractmethod
import logging

if TYPE_CHECKING:
    from janus.core.search.config import ExpressionConfig

from janus_ai.core.grammar.progressive_grammar import ProgressiveGrammar as BaseGrammar # Updated import
from janus_ai.core.expressions.expression import Expression, Variable
from janus_ai.core.expressions.symbolic_math import get_expression_complexity, expression_to_string
from janus_ai.utils.exceptions import GrammarError


class ExpressionGenerator:
    """
    Generates random expressions respecting grammar constraints.
    
    This class handles the creation of random expressions for initialization
    and mutation operations, ensuring all generated expressions are valid
    according to the provided grammar.
    """
    
    def __init__(
        self, 
        grammar: BaseGrammar,
        expression_config: Optional['ExpressionConfig'] = None,
        max_depth: int = 6
    ):
        """
        Initialize expression generator.
        
        Args:
            grammar: Grammar defining valid operations and structure
            expression_config: Configuration for expression generation
            max_depth: Maximum tree depth for generated expressions (fallback)
        """
        self.grammar = grammar
        
        # Use expression config if provided, otherwise use defaults
        if expression_config is not None:
            self.terminal_probability = expression_config.terminal_probability
            self.constant_range = expression_config.constant_range
            self.variable_probability = expression_config.variable_probability
            self.constant_precision = expression_config.constant_precision
            self.use_grammar_probabilities = expression_config.use_grammar_probabilities
            self.fallback_operators = expression_config.fallback_operators
            self.max_depth = max_depth  # Still use parameter for compatibility
        else:
            # Default values
            self.terminal_probability = 0.3
            self.constant_range = (-5.0, 5.0)
            self.variable_probability = 0.7
            self.constant_precision = 3
            self.use_grammar_probabilities = True
            self.fallback_operators = ['+', '-', '*', '/', 'sin', 'cos']
            self.max_depth = max_depth
            
        self.logger = logging.getLogger(__name__)
    
    def generate_random_expression(
        self, 
        variables: List[Variable], 
        max_complexity: Optional[int] = None,
        current_depth: int = 0
    ) -> Optional[Expression]:
        """
        Generate a single random expression.
        
        Args:
            variables: Available variables for the expression
            max_complexity: Maximum allowed complexity (None for no limit)
            current_depth: Current tree depth (for recursion)
            
        Returns:
            Generated Expression or None if generation failed
        """
        if current_depth >= self.max_depth:
            return self._generate_terminal(variables)
        
        # Calculate terminal probability based on depth
        depth_factor = current_depth / self.max_depth
        terminal_prob = self.terminal_probability + (depth_factor * 0.5)
        
        if np.random.random() < terminal_prob:
            return self._generate_terminal(variables)
        else:
            return self._generate_non_terminal(variables, max_complexity, current_depth)
    
    def generate_population(
        self, 
        variables: List[Variable], 
        population_size: int,
        max_complexity: Optional[int] = None
    ) -> List[Expression]:
        """
        Generate a population of random expressions.
        
        Args:
            variables: Available variables
            population_size: Size of population to generate
            max_complexity: Maximum allowed complexity
            
        Returns:
            List of generated expressions
        """
        population = []
        max_attempts = population_size * 10
        attempts = 0
        
        while len(population) < population_size and attempts < max_attempts:
            try:
                expr = self.generate_random_expression(variables, max_complexity)
                if expr is not None:
                    # Validate complexity if specified
                    if max_complexity is None or get_expression_complexity(expr) <= max_complexity:
                        population.append(expr)
            except Exception as e:
                self.logger.debug(f"Failed to generate expression: {e}")
            
            attempts += 1
        
        # Fill remaining slots with simple variables if needed
        while len(population) < population_size:
            var = np.random.choice(variables)
            expr = Expression('var', [var])
            population.append(expr)
        
        return population
    
    def _generate_terminal(self, variables: List[Variable]) -> Expression:
        """Generate a terminal expression (variable or constant)."""
        if np.random.random() < self.variable_probability:
            # Generate variable
            var = np.random.choice(variables)
            return Expression('var', [var])
        else:
            # Generate constant
            constant = np.random.uniform(*self.constant_range)
            # Round to specified precision
            if self.constant_precision >= 0:
                constant = round(constant, self.constant_precision)
            return Expression('const', [constant])
    
    def _generate_non_terminal(
        self, 
        variables: List[Variable], 
        max_complexity: Optional[int],
        current_depth: int
    ) -> Optional[Expression]:
        """Generate a non-terminal expression (operator with operands)."""
        try:
            # Get available operators from grammar
            available_ops = self._get_available_operators()
            
            if not available_ops:
                return self._generate_terminal(variables)
            
            # Select random operator
            operator, arity = random.choice(available_ops)
            
            # Generate operands
            operands = []
            for _ in range(arity):
                operand = self.generate_random_expression(
                    variables, max_complexity, current_depth + 1
                )
                if operand is None:
                    operand = self._generate_terminal(variables)
                operands.append(operand)
            
            expr = Expression(operator, operands)
            
            # Check complexity constraint
            if max_complexity is not None:
                complexity = get_expression_complexity(expr)
                if complexity > max_complexity:
                    return self._generate_terminal(variables)
            
            return expr
            
        except Exception as e:
            self.logger.debug(f"Failed to generate non-terminal: {e}")
            return self._generate_terminal(variables)
    
    def _get_available_operators(self) -> List[Tuple[str, int]]:
        """Get list of available operators and their arities from grammar."""
        operators = []
        
        try:
            # Try to get operators from grammar if it supports the interface
            if hasattr(self.grammar, 'get_operators'):
                # Grammar provides operator list
                grammar_operators = self.grammar.get_operators()
                for op in grammar_operators:
                    if hasattr(self.grammar, 'get_arity'):
                        arity = self.grammar.get_arity(op)
                        operators.append((op, arity))
                    else:
                        # Guess arity based on common operators
                        arity = self._guess_operator_arity(op)
                        operators.append((op, arity))
            
            elif hasattr(self.grammar, 'operators'):
                # Grammar has operators attribute
                for op in self.grammar.operators:
                    arity = self._guess_operator_arity(op)
                    if hasattr(self.grammar, 'get_arity'):
                        arity = self.grammar.get_arity(op)
                    operators.append((op, arity))
            
            else:
                # Fall back to checking if operators are available
                if self.use_grammar_probabilities:
                    # Try common operators and check availability
                    candidate_ops = [
                        ('+', 2), ('-', 2), ('*', 2), ('/', 2), ('**', 2),
                        ('sin', 1), ('cos', 1), ('exp', 1), ('log', 1), ('sqrt', 1), ('neg', 1)
                    ]
                    
                    for op, arity in candidate_ops:
                        if self._is_operator_available(op):
                            operators.append((op, arity))
                else:
                    # Use fallback operators
                    for op in self.fallback_operators:
                        arity = self._guess_operator_arity(op)
                        operators.append((op, arity))
        
        except Exception as e:
            self.logger.warning(f"Failed to get operators from grammar: {e}")
            # Ultimate fallback to basic operators
            operators = [('+', 2), ('-', 2), ('*', 2), ('/', 2)]
        
        # Ensure we have at least some operators
        if not operators:
            self.logger.warning("No operators found, using minimal fallback set")
            operators = [('+', 2), ('-', 2), ('*', 2), ('/', 2)]
        
        return operators
    
    def _guess_operator_arity(self, operator: str) -> int:
        """Guess the arity of an operator based on common conventions."""
        binary_ops = ['+', '-', '*', '/', '//', '%', '**', '^', '&', '|', 
                     'and', 'or', 'max', 'min', 'atan2', 'pow']
        unary_ops = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                    'sinh', 'cosh', 'tanh', 'exp', 'log', 'log10', 'log2',
                    'sqrt', 'abs', 'neg', 'not', '~', 'ceil', 'floor']
        
        if operator in unary_ops:
            return 1
        elif operator in binary_ops:
            return 2
        else:
            # Default guess: if it's a short symbol, likely binary; if word, likely unary
            return 2 if len(operator) <= 2 else 1
    
    def _is_operator_available(self, operator: str) -> bool:
        """Check if an operator is available in the grammar."""
        if hasattr(self.grammar, 'is_operator_known'):
            return self.grammar.is_operator_known(operator)
        else:
            # Fallback: assume basic operators are available
            return operator in ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log', 'sqrt']


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators."""
    
    @abstractmethod
    def crossover(
        self, 
        parent1: Expression, 
        parent2: Expression,
        **kwargs
    ) -> Tuple[Expression, Expression]:
        """
        Perform crossover between two parent expressions.
        
        Args:
            parent1: First parent expression
            parent2: Second parent expression
            **kwargs: Operator-specific parameters
            
        Returns:
            Tuple of two child expressions
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this crossover operator."""
        pass


class SubtreeCrossover(CrossoverOperator):
    """
    Standard subtree crossover operator.
    
    Exchanges random subtrees between two parent expressions to create
    two new child expressions.
    """
    
    def __init__(self, max_attempts: int = 10):
        """
        Initialize subtree crossover.
        
        Args:
            max_attempts: Maximum attempts to find valid crossover points
        """
        self.max_attempts = max_attempts
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "subtree"
    
    def crossover(
        self, 
        parent1: Expression, 
        parent2: Expression,
        **kwargs
    ) -> Tuple[Expression, Expression]:
        """
        Perform subtree crossover.
        
        Args:
            parent1: First parent expression
            parent2: Second parent expression
            
        Returns:
            Tuple of two child expressions
        """
        try:
            # Create deep copies to avoid modifying originals
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            
            # Try to find valid crossover points
            for attempt in range(self.max_attempts):
                try:
                    # Get all nodes from both trees
                    nodes1 = self._collect_all_nodes(child1)
                    nodes2 = self._collect_all_nodes(child2)
                    
                    if not nodes1 or not nodes2:
                        break
                    
                    # Select random crossover points
                    crossover_node1 = random.choice(nodes1)
                    crossover_node2 = random.choice(nodes2)
                    
                    # Perform the swap
                    self._swap_nodes(crossover_node1, crossover_node2)
                    
                    # Validate results (basic check)
                    if self._validate_expression(child1) and self._validate_expression(child2):
                        return child1, child2
                    
                except Exception as e:
                    self.logger.debug(f"Crossover attempt {attempt + 1} failed: {e}")
                    # Reset children for next attempt
                    child1 = copy.deepcopy(parent1)
                    child2 = copy.deepcopy(parent2)
            
            # If all attempts failed, return copies of parents
            self.logger.debug("All crossover attempts failed, returning parent copies")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        except Exception as e:
            self.logger.error(f"Crossover failed: {e}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    def _collect_all_nodes(self, expression: Expression) -> List[Expression]:
        """Collect all nodes in expression tree."""
        nodes = [expression]
        
        if hasattr(expression, 'operands') and expression.operands:
            for operand in expression.operands:
                if isinstance(operand, Expression):
                    nodes.extend(self._collect_all_nodes(operand))
        
        return nodes
    
    def _swap_nodes(self, node1: Expression, node2: Expression):
        """Swap the content of two nodes."""
        # Swap operator and operands
        temp_operator = node1.operator
        temp_operands = node1.operands
        
        node1.operator = node2.operator
        node1.operands = node2.operands
        
        node2.operator = temp_operator
        node2.operands = temp_operands
    
    def _validate_expression(self, expression: Expression) -> bool:
        """Basic validation of expression structure."""
        try:
            # Try to convert to string (basic validation)
            expression_to_string(expression)
            return True
        except Exception:
            return False


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover operator.
    
    Creates children by randomly choosing each node from either parent
    with equal probability.
    """
    
    def __init__(self, swap_probability: float = 0.5):
        """
        Initialize uniform crossover.
        
        Args:
            swap_probability: Probability of swapping each node
        """
        self.swap_probability = swap_probability
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "uniform"
    
    def crossover(
        self, 
        parent1: Expression, 
        parent2: Expression,
        **kwargs
    ) -> Tuple[Expression, Expression]:
        """
        Perform uniform crossover.
        
        Args:
            parent1: First parent expression
            parent2: Second parent expression
            
        Returns:
            Tuple of two child expressions
        """
        try:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            
            # Apply uniform crossover to all corresponding nodes
            self._uniform_crossover_recursive(child1, child2, parent1, parent2)
            
            return child1, child2
            
        except Exception as e:
            self.logger.error(f"Uniform crossover failed: {e}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    def _uniform_crossover_recursive(
        self, 
        child1: Expression, 
        child2: Expression,
        parent1: Expression, 
        parent2: Expression
    ):
        """Recursively apply uniform crossover."""
        # Decide whether to swap this node
        if random.random() < self.swap_probability:
            # Swap operators and operands at this level
            child1.operator = parent2.operator
            child1.operands = copy.deepcopy(parent2.operands)
            
            child2.operator = parent1.operator
            child2.operands = copy.deepcopy(parent1.operands)
        
        # Recurse into operands if they exist and are expressions
        if (hasattr(child1, 'operands') and hasattr(child2, 'operands') and
            child1.operands and child2.operands):
            
            min_operands = min(len(child1.operands), len(child2.operands))
            
            for i in range(min_operands):
                if (isinstance(child1.operands[i], Expression) and 
                    isinstance(child2.operands[i], Expression) and
                    isinstance(parent1.operands[i], Expression) and
                    isinstance(parent2.operands[i], Expression)):
                    
                    self._uniform_crossover_recursive(
                        child1.operands[i], child2.operands[i],
                        parent1.operands[i], parent2.operands[i]
                    )


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""
    
    @abstractmethod
    def mutate(
        self, 
        individual: Expression, 
        variables: List[Variable],
        **kwargs
    ) -> Expression:
        """
        Mutate an individual expression.
        
        Args:
            individual: Expression to mutate
            variables: Available variables
            **kwargs: Operator-specific parameters
            
        Returns:
            Mutated expression
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this mutation operator."""
        pass


class NodeReplacementMutation(MutationOperator):
    """
    Node replacement mutation operator.
    
    Replaces a random node with a new random node while maintaining
    expression validity.
    """
    
    def __init__(self, generator: ExpressionGenerator):
        """
        Initialize node replacement mutation.
        
        Args:
            generator: Expression generator for creating new nodes
        """
        self.generator = generator
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "node_replacement"
    
    def mutate(
        self, 
        individual: Expression, 
        variables: List[Variable],
        **kwargs
    ) -> Expression:
        """
        Perform node replacement mutation.
        
        Args:
            individual: Expression to mutate
            variables: Available variables
            
        Returns:
            Mutated expression
        """
        try:
            mutated = copy.deepcopy(individual)
            nodes = self._collect_all_nodes(mutated)
            
            if not nodes:
                return mutated
            
            # Select random node to replace
            node = random.choice(nodes)
            
            # Replace with random terminal or simple operator
            if random.random() < 0.7:
                # Replace with terminal
                if random.random() < 0.5:
                    # Variable
                    var = np.random.choice(variables)
                    node.operator = 'var'
                    node.operands = [var]
                else:
                    # Constant
                    constant = np.random.uniform(*self.generator.constant_range)
                    node.operator = 'const'
                    node.operands = [constant]
            else:
                # Replace with simple unary operator
                available_ops = self.generator._get_available_operators()
                unary_ops = [(op, arity) for op, arity in available_ops if arity == 1]
                
                if unary_ops:
                    operator, _ = random.choice(unary_ops)
                    original_operand = copy.deepcopy(node)
                    node.operator = operator
                    node.operands = [original_operand]
            
            return mutated
            
        except Exception as e:
            self.logger.error(f"Node replacement mutation failed: {e}")
            return individual
    
    def _collect_all_nodes(self, expression: Expression) -> List[Expression]:
        """Collect all nodes in expression tree."""
        nodes = [expression]
        
        if hasattr(expression, 'operands') and expression.operands:
            for operand in expression.operands:
                if isinstance(operand, Expression):
                    nodes.extend(self._collect_all_nodes(operand))
        
        return nodes


class SubtreeReplacementMutation(MutationOperator):
    """
    Subtree replacement mutation operator.
    
    Replaces a random subtree with a newly generated random subtree.
    """
    
    def __init__(self, generator: ExpressionGenerator, max_new_depth: int = 3):
        """
        Initialize subtree replacement mutation.
        
        Args:
            generator: Expression generator for creating new subtrees
            max_new_depth: Maximum depth for newly generated subtrees
        """
        self.generator = generator
        self.max_new_depth = max_new_depth
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "subtree_replacement"
    
    def mutate(
        self, 
        individual: Expression, 
        variables: List[Variable],
        max_complexity: Optional[int] = None,
        **kwargs
    ) -> Expression:
        """
        Perform subtree replacement mutation.
        
        Args:
            individual: Expression to mutate
            variables: Available variables
            max_complexity: Maximum allowed complexity
            
        Returns:
            Mutated expression
        """
        try:
            mutated = copy.deepcopy(individual)
            nodes = self._collect_all_nodes(mutated)
            
            if not nodes:
                return mutated
            
            # Select random node to replace
            node = random.choice(nodes)
            
            # Generate new subtree
            old_max_depth = self.generator.max_depth
            self.generator.max_depth = self.max_new_depth
            
            try:
                new_subtree = self.generator.generate_random_expression(
                    variables, max_complexity
                )
                
                if new_subtree:
                    node.operator = new_subtree.operator
                    node.operands = new_subtree.operands
            finally:
                # Restore original max depth
                self.generator.max_depth = old_max_depth
            
            return mutated
            
        except Exception as e:
            self.logger.error(f"Subtree replacement mutation failed: {e}")
            return individual
    
    def _collect_all_nodes(self, expression: Expression) -> List[Expression]:
        """Collect all nodes in expression tree."""
        nodes = [expression]
        
        if hasattr(expression, 'operands') and expression.operands:
            for operand in expression.operands:
                if isinstance(operand, Expression):
                    nodes.extend(self._collect_all_nodes(operand))
        
        return nodes


class ConstantPerturbationMutation(MutationOperator):
    """
    Constant perturbation mutation operator.
    
    Adds Gaussian noise to constant values in the expression.
    """
    
    def __init__(self, perturbation_strength: float = 0.1):
        """
        Initialize constant perturbation mutation.
        
        Args:
            perturbation_strength: Standard deviation for Gaussian noise
        """
        self.perturbation_strength = perturbation_strength
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "constant_perturbation"
    
    def mutate(
        self, 
        individual: Expression, 
        variables: List[Variable],
        **kwargs
    ) -> Expression:
        """
        Perform constant perturbation mutation.
        
        Args:
            individual: Expression to mutate
            variables: Available variables (unused but required by interface)
            
        Returns:
            Mutated expression
        """
        try:
            mutated = copy.deepcopy(individual)
            self._perturb_constants_recursive(mutated)
            return mutated
            
        except Exception as e:
            self.logger.error(f"Constant perturbation mutation failed: {e}")
            return individual
    
    def _perturb_constants_recursive(self, expression: Expression):
        """Recursively perturb constants in expression."""
        if expression.operator == 'const' and expression.operands:
            try:
                current_value = float(expression.operands[0])
                perturbation = np.random.normal(
                    0, abs(current_value) * self.perturbation_strength + 0.01
                )
                expression.operands[0] = current_value + perturbation
            except (ValueError, TypeError):
                # If conversion fails, replace with new random constant
                expression.operands[0] = np.random.normal(0, 1)
        
        # Recurse into operands
        if hasattr(expression, 'operands') and expression.operands:
            for operand in expression.operands:
                if isinstance(operand, Expression):
                    self._perturb_constants_recursive(operand)


class OperatorMutation(MutationOperator):
    """
    Operator mutation operator.
    
    Changes the operator of a random non-terminal node while preserving
    the operand structure when possible.
    """
    
    def __init__(self, generator: ExpressionGenerator):
        """
        Initialize operator mutation.
        
        Args:
            generator: Expression generator for accessing available operators
        """
        self.generator = generator
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "operator_mutation"
    
    def mutate(
        self, 
        individual: Expression, 
        variables: List[Variable],
        **kwargs
    ) -> Expression:
        """
        Perform operator mutation.
        
        Args:
            individual: Expression to mutate
            variables: Available variables (unused but required by interface)
            
        Returns:
            Mutated expression
        """
        try:
            mutated = copy.deepcopy(individual)
            non_terminal_nodes = self._collect_non_terminal_nodes(mutated)
            
            if not non_terminal_nodes:
                return mutated
            
            # Select random non-terminal node
            node = random.choice(non_terminal_nodes)
            current_arity = len(node.operands) if node.operands else 0
            
            # Get operators with same arity
            available_ops = self.generator._get_available_operators()
            compatible_ops = [op for op, arity in available_ops if arity == current_arity]
            
            if compatible_ops:
                # Choose different operator
                current_op = node.operator
                new_operators = [op for op in compatible_ops if op != current_op]
                
                if new_operators:
                    node.operator = random.choice(new_operators)
            
            return mutated
            
        except Exception as e:
            self.logger.error(f"Operator mutation failed: {e}")
            return individual
    
    def _collect_non_terminal_nodes(self, expression: Expression) -> List[Expression]:
        """Collect all non-terminal nodes in expression tree."""
        nodes = []
        
        # Check if this node is non-terminal (has operands that are expressions)
        if (hasattr(expression, 'operands') and expression.operands and
            any(isinstance(op, Expression) for op in expression.operands)):
            nodes.append(expression)
        
        # Recurse into operands
        if hasattr(expression, 'operands') and expression.operands:
            for operand in expression.operands:
                if isinstance(operand, Expression):
                    nodes.extend(self._collect_non_terminal_nodes(operand))
        
        return nodes


# Operator registry for extensibility
_CROSSOVER_REGISTRY: Dict[str, type] = {
    "subtree": SubtreeCrossover,
    "uniform": UniformCrossover
}

_MUTATION_REGISTRY: Dict[str, type] = {
    "node_replacement": NodeReplacementMutation,
    "subtree_replacement": SubtreeReplacementMutation,
    "constant_perturbation": ConstantPerturbationMutation,
    "operator_mutation": OperatorMutation
}


def create_crossover_operator(name: str, **kwargs) -> CrossoverOperator:
    """
    Create a crossover operator by name.
    
    Args:
        name: Name of the crossover operator
        **kwargs: Operator-specific parameters
        
    Returns:
        Configured crossover operator
    """
    if name not in _CROSSOVER_REGISTRY:
        available = list(_CROSSOVER_REGISTRY.keys())
        raise GrammarError(
            f"Unknown crossover operator: {name}",
            f"Available operators: {available}"
        )
    
    return _CROSSOVER_REGISTRY[name](**kwargs)


def create_mutation_operator(
    name: str, 
    generator: ExpressionGenerator, 
    **kwargs
) -> MutationOperator:
    """
    Create a mutation operator by name.
    
    Args:
        name: Name of the mutation operator
        generator: Expression generator for the operator
        **kwargs: Operator-specific parameters
        
    Returns:
        Configured mutation operator
    """
    if name not in _MUTATION_REGISTRY:
        available = list(_MUTATION_REGISTRY.keys())
        raise GrammarError(
            f"Unknown mutation operator: {name}",
            f"Available operators: {available}"
        )
    
    # Add generator to kwargs for operators that need it
    if name in ["node_replacement", "subtree_replacement", "operator_mutation"]:
        kwargs['generator'] = generator
    
    return _MUTATION_REGISTRY[name](**kwargs)


def list_operators() -> Dict[str, List[str]]:
    """List all available operators."""
    return {
        "crossover": list(_CROSSOVER_REGISTRY.keys()),
        "mutation": list(_MUTATION_REGISTRY.keys())
    }


def register_crossover_operator(name: str, operator_class: type):
    """Register a new crossover operator."""
    _CROSSOVER_REGISTRY[name] = operator_class


def register_mutation_operator(name: str, operator_class: type):
    """Register a new mutation operator."""
    _MUTATION_REGISTRY[name] = operator_class
