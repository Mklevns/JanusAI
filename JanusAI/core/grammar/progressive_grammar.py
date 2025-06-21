"""Progressive Grammar core components."""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Set, Any, TypeVar, Generic, Union # Added Generic
from collections import defaultdict
import random
import logging
from scipy.stats import entropy
from sklearn.decomposition import FastICA
import warnings

# Imports that will be needed from the new module structure
from .expression import Expression, Variable
from .denoiser import NoisyObservationProcessor
from .cfg import ContextFreeGrammar, CFGRule # CFGRule is used by ProgressiveGrammar

warnings.filterwarnings('ignore')

# TargetType was defined in base_grammar.py, ensure it's available if needed broadly
# For now, it seems only CFG was using it directly. ProgressiveGrammar uses CFGRule which is generic.
# If ProgressiveGrammar itself needs to be generic over TargetType, it should be defined here or imported.

class ProgressiveGrammar:
    """
    A hierarchical grammar that discovers variables from observations and
    progressively builds mathematical abstractions using information-theoretic
    principles, primarily for symbolic regression and physics discovery.
    """
    COMMUTATIVE_OPS = {'+', '*'}

    def __init__(self,
                 max_variables: int = 20,
                 noise_threshold: float = 0.1,
                 mdl_threshold: float = 10.0,
                 load_defaults: bool = True):

        self.primitives = {
            'constants': {}, 'binary_ops': set(),
            'unary_ops': set(), 'calculus_ops': set()
        }

        if load_defaults:
            self.primitives['constants'] = {'0': 0, '1': 1, 'pi': np.pi, 'e': np.e}
            self.primitives['binary_ops'] = {'+', '-', '*', '/', '**'}
            self.primitives['unary_ops'] = {'neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos'}
            self.primitives['calculus_ops'] = {'diff', 'int'}

        self.variables: Dict[str, Variable] = {}
        self.learned_functions: Dict[str, Expression] = {}
        self.proven_lemmas: Dict[str, Expression] = {}
        self.max_variables = max_variables
        self.noise_threshold = noise_threshold
        self.mdl_threshold = mdl_threshold
        self.denoiser = NoisyObservationProcessor()
        self._expression_cache = {}
        self.cfg_grammar: Optional[ContextFreeGrammar[Union[str, Variable]]] = None # Corrected CFG type hint
        self.cfg_start_symbol: Optional[str] = None


    def add_operators(self, operators: List[str]):
        known_binary = {'+', '-', '*', '/', '**'}
        known_unary = {'neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos'}
        known_calculus = {'diff', 'int'}
        for op in operators:
            if op in known_binary: self.primitives['binary_ops'].add(op)
            elif op in known_unary: self.primitives['unary_ops'].add(op)
            elif op in known_calculus: self.primitives['calculus_ops'].add(op)
            elif op == '**2' or op == '**3': self.primitives['binary_ops'].add('**')
            elif op == '1/': self.primitives['unary_ops'].add('inv')
            else: print(f"Warning: Operator '{op}' has unknown arity and was not added.")

    def discover_variables(self,
                          observations: np.ndarray,
                          time_stamps: Optional[np.ndarray] = None) -> List[Variable]:
        clean_obs = self.denoiser.denoise(observations)
        num_components = min(self.max_variables, clean_obs.shape[1])
        if num_components <= 0: # Handle case with no features or no max_vars
             return []
        ica = FastICA(n_components=num_components, random_state=0) # Added random_state for reproducibility
        try:
            components = ica.fit_transform(clean_obs)
        except ValueError as e:
            # This can happen if all components are flat after denoising, or other ICA issues.
            logging.warning(f"ICA failed during variable discovery: {e}. Returning no variables.")
            return []


        discovered_vars = []
        for i in range(components.shape[1]):
            component = components[:, i]
            properties = self._analyze_component(component, time_stamps)
            if properties['information_content'] > self.noise_threshold:
                var_name = self._generate_variable_name(properties)
                var = Variable(name=var_name, index=i, properties=properties)
                discovered_vars.append(var)
                self.variables[var_name] = var
        return discovered_vars

    def _analyze_component(self,
                          component: np.ndarray,
                          time_stamps: Optional[np.ndarray]) -> Dict[str, float]:
        properties = {}
        hist, _ = np.histogram(component, bins=50)
        hist = hist + 1e-10
        properties['information_content'] = entropy(hist) / np.log(len(hist)) if len(hist) > 1 else 0.0

        if len(component) > 10:
            num_splits = min(10, len(component)//10)
            if num_splits > 0: # Ensure num_splits is at least 1 for array_split
                windows = np.array_split(component, num_splits)
                variances = [np.var(w) for w in windows if len(w) > 0] # Ensure window is not empty
                properties['conservation_score'] = 1.0 / (1.0 + np.var(variances)) if variances else 1.0
            else: # Not enough data for splitting
                 properties['conservation_score'] = 0.0
        else:
            properties['conservation_score'] = 0.0

        if len(component) > 20:
            fft_abs = np.abs(np.fft.fft(component - np.mean(component)))
            power = fft_abs[:len(fft_abs)//2]
            if len(power) > 1: # Need at least one non-DC component
                peak_power = np.max(power[1:])
                avg_power = np.mean(power[1:])
                properties['periodicity_score'] = peak_power / (avg_power + 1e-10)
            else: # Only DC component or less
                properties['periodicity_score'] = 0.0
        else:
            properties['periodicity_score'] = 0.0

        if len(component) > 2:
            properties['smoothness'] = 1.0 / (1.0 + np.std(np.diff(component)))
        else:
            properties['smoothness'] = 0.0
        return properties

    def _generate_variable_name(self, properties: Dict[str, float]) -> str:
        if properties.get('conservation_score', 0) > 0.8: prefix = 'E'
        elif properties.get('periodicity_score', 0) > 5.0: prefix = 'theta'
        elif properties.get('smoothness', 0) > 0.7: prefix = 'x'
        else: prefix = 'q'
        existing = [v for v in self.variables if v.startswith(prefix)]
        return f"{prefix}_{len(existing) + 1}"

    def create_expression(self,
                         operator: str,
                         operands: List[Any],
                         validate: bool = True) -> Optional[Expression]:
        if validate and not self._validate_expression(operator, operands):
            return None
        # Expression class is now imported
        expr = Expression(operator, operands)
        expr_key = self._expression_key(expr) # Uses self._expression_key
        self._expression_cache[expr_key] = expr
        return expr

    def _validate_expression(self, operator: str, operands: List[Any]) -> bool:
        all_ops = (self.primitives['binary_ops'] |
                  self.primitives['unary_ops'] |
                  self.primitives['calculus_ops'] |
                  set(self.learned_functions.keys()) | # Learned functions can be ops
                  {'var', 'const'})
        if operator not in all_ops: return False

        # Arity checks
        expected_arity = -1
        if operator in self.primitives['binary_ops']: expected_arity = 2
        elif operator in self.primitives['unary_ops']: expected_arity = 1
        elif operator in self.primitives['calculus_ops']: expected_arity = 2
        elif operator == 'var': expected_arity = 1
        elif operator == 'const': expected_arity = 1
        elif operator in self.learned_functions: # Assuming learned funcs are unary for now
            expected_arity = 1 # Or store arity with learned functions

        if expected_arity != -1 and len(operands) != expected_arity: return False

        # Type checks
        for op_val in operands:
            if not isinstance(op_val, (Expression, Variable, int, float, str)): # Allow str for var names
                return False
        if operator == 'var' and not isinstance(operands[0], str): return False
        if operator == 'const' and not isinstance(operands[0], (int, float)): return False
        if operator in self.primitives['calculus_ops'] and not isinstance(operands[1], Variable): return False

        return True

    def add_learned_function(self,
                           name: str,
                           expression: Expression,
                           usage_data: List[Expression]) -> bool:
        compression_gain = self._calculate_compression_gain(expression, usage_data)
        if compression_gain > self.mdl_threshold:
            self.learned_functions[name] = expression
            self.primitives['unary_ops'].add(name) # Assuming new functions are unary
            return True
        return False

    def _calculate_compression_gain(self,
                                  candidate: Expression,
                                  corpus: List[Expression]) -> float:
        current_length = sum(expr.complexity for expr in corpus)
        new_length = candidate.complexity
        for expr in corpus:
            occurrences = self._count_subexpression(expr, candidate)
            if occurrences > 0:
                saved = occurrences * (candidate.complexity - 1)
                new_length += expr.complexity - saved
            else:
                new_length += expr.complexity
        return current_length - new_length

    def _count_subexpression(self,
                           expr: Expression,
                           pattern: Expression) -> int:
        if self._expression_key(expr) == self._expression_key(pattern):
            return 1
        count = 0
        if hasattr(expr, 'operands'): # Check if it's an Expression-like object
            for op_node in expr.operands:
                if isinstance(op_node, Expression):
                    count += self._count_subexpression(op_node, pattern)
        return count

    def _expression_key(self, expr: Any) -> str: # Changed type hint for expr
        if isinstance(expr, Variable):
            return f"var:{expr.name}"
        elif isinstance(expr, (int, float)):
            return f"const:{float(expr):.6g}"
        elif isinstance(expr, str): # Should be a variable name string
            return f"var:{expr}"
        elif isinstance(expr, Expression): # Check if it's an Expression object
            operand_keys = []
            for op_node in expr.operands:
                operand_keys.append(self._expression_key(op_node)) # Recursive call
            if expr.operator in self.COMMUTATIVE_OPS:
                operand_keys.sort()
            return f"{expr.operator}({','.join(operand_keys)})"
        else: # Fallback for unknown types
            return str(expr)


    def mine_abstractions(self,
                         hypothesis_library: List[Expression],
                         min_frequency: int = 3) -> Dict[str, Expression]:
        pattern_counts = defaultdict(int)
        pattern_examples = defaultdict(list)
        for hypothesis in hypothesis_library:
            subexprs = self._extract_all_subexpressions(hypothesis)
            for subexpr in subexprs:
                if subexpr.complexity > 2:
                    key = self._expression_key(subexpr)
                    pattern_counts[key] += 1
                    pattern_examples[key].append(subexpr)
        abstractions = {}
        for pattern_key, count in pattern_counts.items():
            if count >= min_frequency:
                example = pattern_examples[pattern_key][0]
                name = f"f_{len(self.learned_functions)}"
                if self.add_learned_function(name, example, hypothesis_library):
                    abstractions[name] = example
        return abstractions

    def set_rules_from_cfg(self, rules: List[CFGRule[Union[str, Variable]]], start_symbol: str = "EXPR"):
        self.cfg_grammar = ContextFreeGrammar[Union[str, Variable]](rules)
        self.cfg_start_symbol = start_symbol
        logging.info(f"CFG rules set in ProgressiveGrammar with start symbol '{start_symbol}'.")

    def _generate_from_symbol_cfg(self, symbol: Union[str, Variable], max_depth: int, current_depth: int) -> Optional[Any]:
        if current_depth > max_depth:
            logging.warning(f"Max recursion depth {max_depth} exceeded for '{symbol}'")
            if isinstance(symbol, Variable): return symbol
            if isinstance(symbol, str) and symbol.isupper() and self.variables: # Heuristic for non-terminal like VAR, CONST
                # Check if it's a known variable name first
                if symbol in self.variables: return self.variables[symbol]
                # Fallback for generic non-terminals if no specific rule matched
                if symbol == 'VAR' and self.variables: return random.choice(list(self.variables.values()))
                if symbol == 'CONST': return random.choice([0, 1, -1, round(random.uniform(-2,2),2)])

            return None # Cannot resolve further

        if isinstance(symbol, Variable): return symbol
        if not isinstance(symbol, str):
            logging.error(f"Unexpected symbol type: {type(symbol)}, value: {symbol}")
            return None

        if self.cfg_grammar and symbol in self.cfg_grammar.rules: # Check cfg_grammar exists
            generated_sequence = self.cfg_grammar.generate_random(symbol)
            if not generated_sequence:
                logging.warning(f"CFG empty sequence for '{symbol}'")
                return None

            op_name_or_terminal = generated_sequence[0]
            # If sequence starts with a known operator
            if isinstance(op_name_or_terminal, str) and self.is_operator_known(op_name_or_terminal):
                op_name = op_name_or_terminal
                arity = self.get_arity(op_name)
                # The rest of the sequence are the operands for this operator
                operand_symbols = generated_sequence[1:]
                if len(operand_symbols) != arity:
                    logging.warning(f"Arity mismatch for '{op_name}'. Exp {arity}, got {len(operand_symbols)} from {symbol} -> {generated_sequence}.")
                    # Attempt to recover if too many symbols, or fail if too few
                    if len(operand_symbols) < arity: return None
                    operand_symbols = operand_symbols[:arity] # Truncate if too many

                operands = []
                for i in range(arity):
                    operand_expr = self._generate_from_symbol_cfg(operand_symbols[i], max_depth, current_depth + 1)
                    if operand_expr is None:
                        logging.warning(f"Failed to gen operand {i} for '{op_name}' from '{operand_symbols[i]}'.")
                        return None
                    operands.append(operand_expr)
                return self.create_expression(op_name, operands, validate=True)
            else: # Sequence does not start with a known operator, treat first element as a terminal/symbol to expand
                 # This covers cases like EXPR -> VAR, or EXPR -> UNARY_OP_EXPR
                 # If generated_sequence has multiple items, but the first isn't an operator,
                 # it implies the CFG rule itself is malformed (e.g., EXPR -> VAR VAR, which isn't standard for expressions)
                 # or it's a sequence of terminals. For expression generation, we expect an operator or a single non-terminal.
                 # For simplicity, we assume a single item to expand or a terminal.
                if len(generated_sequence) > 1:
                     logging.warning(f"CFG rule for '{symbol}' expanded to sequence '{generated_sequence}' not starting with an operator. Expanding first element only.")
                return self._generate_from_symbol_cfg(op_name_or_terminal, max_depth, current_depth+1)


        elif symbol in self.variables: return self.variables[symbol]
        elif symbol == 'CONST': return random.choice([0, 1, -1, round(random.uniform(-2,2),2)])
        elif symbol.startswith("var_") or symbol in [v.name for v in self.variables.values()]: # Generic var from CFG
            if symbol in self.variables: return self.variables[symbol]
            logging.warning(f"CFG unknown var name '{symbol}'. Using random known var.")
            if self.variables: return random.choice(list(self.variables.values()))
            return None # No variables available to choose from
        elif symbol in self.primitives['constants']: # Check if it's a named constant like 'pi'
            return self.primitives['constants'][symbol]
        else: # Unknown terminal from CFG, potentially a number or unhandled symbol
            try: # Attempt to parse as float if it's a string literal number
                return float(symbol)
            except ValueError:
                logging.warning(f"Unknown terminal symbol '{symbol}' from CFG. Cannot resolve.")
                return None


    def generate_random_expression_from_cfg(self, start_symbol: Optional[str] = None, max_depth: int = 10) -> Optional[Expression]:
        if not hasattr(self, 'cfg_grammar') or self.cfg_grammar is None:
            logging.error("CFG not initialized. Call set_rules_from_cfg.")
            return None
        current_start_symbol = start_symbol if start_symbol else self.cfg_start_symbol
        if not current_start_symbol:
            logging.error("No start symbol for CFG generation.")
            return None

        generated_component = self._generate_from_symbol_cfg(current_start_symbol, max_depth, 0)

        if isinstance(generated_component, Expression): return generated_component
        if generated_component is None: return None

        # Wrap non-Expression results (Variable, const) into an Expression
        if isinstance(generated_component, Variable):
            return self.create_expression('var', [generated_component.name], validate=False)
        elif isinstance(generated_component, (int, float)):
            return self.create_expression('const', [generated_component], validate=False)
        else:
            logging.error(f"CFG gen resulted in unexpected type: {type(generated_component)}")
            return None


    def _extract_all_subexpressions(self,
                                   expr: Expression,
                                   collected: Optional[Set[str]] = None) -> List[Expression]:
        if collected is None: collected = set()
        result = []
        key = self._expression_key(expr)
        if key not in collected:
            collected.add(key)
            result.append(expr)
            if hasattr(expr, 'operands'):
                for op_node in expr.operands:
                    if isinstance(op_node, Expression):
                        result.extend(self._extract_all_subexpressions(op_node, collected))
        return result

    def export_grammar_state(self) -> Dict:
        return {
            'variables': { name: { 'index': var.index, 'properties': var.properties }
                           for name, var in self.variables.items() },
            'learned_functions': { name: self._expression_to_dict(expr)
                                  for name, expr in self.learned_functions.items() },
            'proven_lemmas': { name: self._expression_to_dict(expr)
                              for name, expr in self.proven_lemmas.items() }
        }

    def _expression_to_dict(self, expr: Expression) -> Dict:
        return {
            'operator': expr.operator,
            'operands': [ self._expression_to_dict(op) if isinstance(op, Expression)
                         else {'type': 'var', 'name': op.name} if isinstance(op, Variable)
                         else {'type': 'const', 'value': op}
                         for op in expr.operands ],
            'complexity': expr.complexity
        }

    # Methods for arity and operator checking, to be used by Expression or internally
    def get_arity(self, op_name: str) -> int:
        if op_name in self.primitives.get('binary_ops', set()): return 2
        if op_name in self.primitives.get('unary_ops', set()): return 1
        # Assuming calculus ops like diff(expr, var) take 2 args
        if op_name in self.primitives.get('calculus_ops', set()): return 2
        if op_name in self.learned_functions: # Assuming arity 1 for learned functions for now
            # TODO: Store arity with learned functions
            return 1
        if op_name == 'var' or op_name == 'const': return 1 # 'var' takes name, 'const' takes value
        raise ValueError(f"Unknown operator or function: '{op_name}' in ProgressiveGrammar")

    def is_operator_known(self, op_name: str) -> bool:
        try:
            self.get_arity(op_name)
            return True
        except ValueError:
            # Also check constants, as they might be used in CFG rules as terminals that are not operators
            # but need to be recognized by the generation process.
            if op_name in self.primitives.get('constants', {}):
                return True
            return False

# It's good practice to move the __main__ block or ensure it correctly imports
# if this file is run directly. For now, it's removed from the class file.
# The monkey patching of ProgressiveGrammar.get_arity and is_operator_known
# should also be handled carefully. If these methods are central to ProgressiveGrammar,
# they should be part of the class definition itself.

# The original file had:
# ProgressiveGrammar.get_arity = lambda self, op_name: ...
# ProgressiveGrammar.is_operator_known = lambda self, op_name: _is_operator_known_impl(self, op_name)
# This is generally not a good pattern for class methods.
# I've integrated `get_arity` and `is_operator_known` as methods within the class.
# The `_is_operator_known_impl` is effectively what `is_operator_known` now does.
