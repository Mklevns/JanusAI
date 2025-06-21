"""Context-Free Grammar (CFG) components."""

from typing import Dict, List, Tuple, Optional, Set, Any, TypeVar, Generic, Union
from dataclasses import dataclass
from collections import defaultdict
import random
import logging

TargetType = TypeVar('TargetType')

@dataclass
class CFGRule(Generic[TargetType]):
    """Represents a single rule in a context-free grammar."""
    symbol: str
    expression: Union[str, List[Union[str, TargetType]]] # TargetType for direct objects/terminals
    weight: float = 1.0  # For weighted random choice

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("Rule weight must be positive.")

class ContextFreeGrammar(Generic[TargetType]):
    """Represents a context-free grammar with support for weighted random generation."""
    rules: Dict[str, List[CFGRule[TargetType]]]

    def __init__(self, rules: Optional[List[CFGRule[TargetType]]] = None):
        self.rules = defaultdict(list)
        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: CFGRule[TargetType]):
        """Adds a rule to the grammar."""
        self.rules[rule.symbol].append(rule)

    def get_productions(self, symbol: str) -> List[CFGRule[TargetType]]:
        """Returns all production rules for a given symbol."""
        if symbol not in self.rules:
            raise ValueError(f"Symbol '{symbol}' not found in grammar rules.")
        return self.rules[symbol]

    def generate_random(self, start_symbol: str) -> List[Union[str, TargetType]]:
        """
        Generates a random sequence (string or list of items) from the grammar,
        respecting rule weights.
        Returns a list of terminal symbols or direct TargetType objects.
        """
        expansion_stack = [start_symbol]
        result_sequence = []

        max_depth = 100 # Protection against infinite recursion in cyclic grammars without proper terminal paths
        current_depth = 0

        while expansion_stack and current_depth < max_depth:
            current_symbol = expansion_stack.pop(0) # Process first-in (BFS-like for structure)

            if current_symbol not in self.rules:
                # If it's not a non-terminal, it's a terminal symbol or a direct object.
                result_sequence.append(current_symbol)
                continue

            productions = self.get_productions(current_symbol)
            if not productions:
                result_sequence.append(current_symbol)
                continue

            total_weight = sum(rule.weight for rule in productions)
            chosen_weight = random.uniform(0, total_weight)
            cumulative_weight = 0
            chosen_rule = None
            for rule in productions:
                cumulative_weight += rule.weight
                if chosen_weight <= cumulative_weight:
                    chosen_rule = rule
                    break

            if chosen_rule is None: chosen_rule = productions[0]


            if isinstance(chosen_rule.expression, list):
                expansion_stack = list(chosen_rule.expression) + expansion_stack
            else:
                expansion_stack.insert(0, chosen_rule.expression)

            current_depth +=1

        if current_depth >= max_depth:
            logging.warning(f"Max generation depth reached for start symbol '{start_symbol}'. Output may be incomplete.")

        return result_sequence
