# janus/core/search/selection.py
"""
Selection strategies for genetic algorithms.

This module provides various selection strategies for genetic algorithms,
implementing a registry pattern for easy extensibility and configuration.
"""

import numpy as np
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import random
import logging

from janus_ai.core.expressions.expression import Expression
from janus_ai.utils.exceptions import UnsupportedOperationError, OptimizationError


class SelectionStrategy(ABC):
    """
    Abstract base class for selection strategies.
    
    All selection strategies should inherit from this class and implement
    the select method.
    """
    
    @abstractmethod
    def select(
        self, 
        population: List[Expression], 
        fitnesses: List[float], 
        num_parents: int = 2,
        **kwargs
    ) -> List[Expression]:
        """
        Select parents from population based on fitness.
        
        Args:
            population: List of Expression objects
            fitnesses: Fitness values corresponding to population
            num_parents: Number of parents to select
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of selected parent expressions
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this selection strategy."""
        pass
    
    def validate_inputs(
        self, 
        population: List[Expression], 
        fitnesses: List[float]
    ) -> None:
        """Validate inputs for selection."""
        if not population:
            raise OptimizationError("Population is empty")
        
        if len(population) != len(fitnesses):
            raise OptimizationError(
                f"Population size ({len(population)}) doesn't match "
                f"fitness array size ({len(fitnesses)})"
            )
        
        if not any(np.isfinite(f) for f in fitnesses):
            raise OptimizationError("No valid fitness values found")


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection strategy.
    
    Selects parents by running tournaments between randomly chosen individuals.
    The fittest individual in each tournament is selected as a parent.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament
        """
        self.tournament_size = tournament_size
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "tournament"
    
    def select(
        self, 
        population: List[Expression], 
        fitnesses: List[float], 
        num_parents: int = 2,
        tournament_size: Optional[int] = None
    ) -> List[Expression]:
        """
        Perform tournament selection.
        
        Args:
            population: Population to select from
            fitnesses: Fitness values
            num_parents: Number of parents to select
            tournament_size: Size of tournament (overrides default if provided)
        """
        self.validate_inputs(population, fitnesses)
        
        if tournament_size is None:
            tournament_size = self.tournament_size
        
        # Ensure tournament size doesn't exceed population
        tournament_size = min(tournament_size, len(population))
        
        parents = []
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament_indices = np.random.choice(
                len(population), 
                size=tournament_size, 
                replace=False
            )
            
            # Find the fittest individual in tournament
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            
            # Handle case where all tournament fitnesses are invalid
            valid_indices = [
                i for i, fit in zip(tournament_indices, tournament_fitnesses) 
                if np.isfinite(fit)
            ]
            
            if valid_indices:
                valid_fitnesses = [fitnesses[i] for i in valid_indices]
                winner_idx = valid_indices[np.argmax(valid_fitnesses)]
            else:
                # Fallback to random selection if no valid fitnesses
                winner_idx = np.random.choice(tournament_indices)
                self.logger.warning("No valid fitnesses in tournament, selecting randomly")
            
            parents.append(population[winner_idx])
            
        return parents


class RouletteWheelSelection(SelectionStrategy):
    """
    Roulette wheel (fitness proportionate) selection strategy.
    
    Selects parents with probability proportional to their fitness.
    """
    
    def __init__(self, min_fitness_offset: float = 1e-8):
        """
        Initialize roulette wheel selection.
        
        Args:
            min_fitness_offset: Minimum offset to ensure positive probabilities
        """
        self.min_fitness_offset = min_fitness_offset
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "roulette"
    
    def select(
        self, 
        population: List[Expression], 
        fitnesses: List[float], 
        num_parents: int = 2,
        **kwargs
    ) -> List[Expression]:
        """
        Perform roulette wheel selection.
        
        Args:
            population: Population to select from
            fitnesses: Fitness values
            num_parents: Number of parents to select
        """
        self.validate_inputs(population, fitnesses)
        
        # Filter out invalid fitnesses
        valid_indices = [i for i, f in enumerate(fitnesses) if np.isfinite(f)]
        
        if not valid_indices:
            # Fallback to uniform random if no valid fitnesses
            self.logger.warning("No valid fitnesses, falling back to random selection")
            return [random.choice(population) for _ in range(num_parents)]
        
        valid_population = [population[i] for i in valid_indices]
        valid_fitnesses = [fitnesses[i] for i in valid_indices]
        
        # Shift fitnesses to be positive
        min_fitness = min(valid_fitnesses)
        adjusted_fitnesses = [
            f - min_fitness + self.min_fitness_offset 
            for f in valid_fitnesses
        ]
        
        # Calculate selection probabilities
        total_fitness = sum(adjusted_fitnesses)
        
        # Handle case where total fitness is zero or very small
        if total_fitness <= self.min_fitness_offset:
            self.logger.warning("Total fitness too small, falling back to uniform selection")
            probabilities = [1.0 / len(valid_population)] * len(valid_population)
        else:
            probabilities = [f / total_fitness for f in adjusted_fitnesses]
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            try:
                selected_idx = np.random.choice(len(valid_population), p=probabilities)
                parents.append(valid_population[selected_idx])
            except ValueError as e:
                # Fallback to random selection if probability issues
                self.logger.warning(f"Probability error in roulette selection: {e}")
                parents.append(random.choice(valid_population))
        
        return parents


class RankSelection(SelectionStrategy):
    """
    Rank-based selection strategy.
    
    Selects parents based on rank rather than raw fitness values,
    providing more controlled selection pressure.
    """
    
    def __init__(self, selection_pressure: float = 2.0):
        """
        Initialize rank selection.
        
        Args:
            selection_pressure: Controls selection pressure (1.0 = uniform, 2.0 = default)
        """
        if not 1.0 <= selection_pressure <= 2.0:
            raise ValueError("Selection pressure must be between 1.0 and 2.0")
        
        self.selection_pressure = selection_pressure
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "rank"
    
    def select(
        self, 
        population: List[Expression], 
        fitnesses: List[float], 
        num_parents: int = 2,
        selection_pressure: Optional[float] = None
    ) -> List[Expression]:
        """
        Perform rank-based selection.
        
        Args:
            population: Population to select from
            fitnesses: Fitness values
            num_parents: Number of parents to select
            selection_pressure: Selection pressure (overrides default if provided)
        """
        self.validate_inputs(population, fitnesses)
        
        if selection_pressure is None:
            selection_pressure = self.selection_pressure
        
        # Filter out invalid fitnesses and create (index, fitness) pairs
        indexed_fitnesses = [
            (i, f) for i, f in enumerate(fitnesses) if np.isfinite(f)
        ]
        
        if not indexed_fitnesses:
            # Fallback to random selection
            self.logger.warning("No valid fitnesses, falling back to random selection")
            return [random.choice(population) for _ in range(num_parents)]
        
        # Sort by fitness (ascending order for ranking)
        indexed_fitnesses.sort(key=lambda x: x[1])
        
        # Calculate rank-based probabilities
        n = len(indexed_fitnesses)
        ranks = np.arange(1, n + 1)
        
        # Linear ranking formula: P(i) = (2-SP)/n + 2*rank*(SP-1)/(n*(n-1))
        probabilities = (
            (2 - selection_pressure) / n + 
            2 * ranks * (selection_pressure - 1) / (n * (n - 1))
        )
        
        # Normalize probabilities (should already sum to 1, but ensure it)
        probabilities = probabilities / np.sum(probabilities)
        
        # Create mapping from rank index to original population index
        rank_to_pop_idx = [pair[0] for pair in indexed_fitnesses]
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            try:
                selected_rank_idx = np.random.choice(len(probabilities), p=probabilities)
                original_pop_idx = rank_to_pop_idx[selected_rank_idx]
                parents.append(population[original_pop_idx])
            except ValueError as e:
                # Fallback to random selection
                self.logger.warning(f"Probability error in rank selection: {e}")
                original_pop_idx = random.choice(rank_to_pop_idx)
                parents.append(population[original_pop_idx])
        
        return parents


class ElitistSelection(SelectionStrategy):
    """
    Elitist selection strategy.
    
    Always selects the best individuals from the population.
    Useful for ensuring the best solutions survive to the next generation.
    """
    
    @property
    def name(self) -> str:
        return "elitist"
    
    def select(
        self, 
        population: List[Expression], 
        fitnesses: List[float], 
        num_parents: int = 2,
        **kwargs
    ) -> List[Expression]:
        """
        Perform elitist selection.
        
        Args:
            population: Population to select from
            fitnesses: Fitness values
            num_parents: Number of parents to select
        """
        self.validate_inputs(population, fitnesses)
        
        # Filter out invalid fitnesses
        valid_pairs = [
            (pop, fit) for pop, fit in zip(population, fitnesses) 
            if np.isfinite(fit)
        ]
        
        if not valid_pairs:
            # Fallback to random selection
            return [random.choice(population) for _ in range(num_parents)]
        
        # Sort by fitness (descending order)
        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top individuals (with replacement if needed)
        parents = []
        for i in range(num_parents):
            parents.append(valid_pairs[i % len(valid_pairs)][0])
        
        return parents


class StochasticUniversalSampling(SelectionStrategy):
    """
    Stochastic Universal Sampling (SUS) selection strategy.
    
    Provides more uniform sampling than roulette wheel selection
    while maintaining proportional selection pressure.
    """
    
    def __init__(self, min_fitness_offset: float = 1e-8):
        """
        Initialize SUS selection.
        
        Args:
            min_fitness_offset: Minimum offset to ensure positive probabilities
        """
        self.min_fitness_offset = min_fitness_offset
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "sus"
    
    def select(
        self, 
        population: List[Expression], 
        fitnesses: List[float], 
        num_parents: int = 2,
        **kwargs
    ) -> List[Expression]:
        """
        Perform stochastic universal sampling.
        
        Args:
            population: Population to select from
            fitnesses: Fitness values
            num_parents: Number of parents to select
        """
        self.validate_inputs(population, fitnesses)
        
        # Filter out invalid fitnesses
        valid_indices = [i for i, f in enumerate(fitnesses) if np.isfinite(f)]
        
        if not valid_indices:
            return [random.choice(population) for _ in range(num_parents)]
        
        valid_population = [population[i] for i in valid_indices]
        valid_fitnesses = [fitnesses[i] for i in valid_indices]
        
        # Shift fitnesses to be positive
        min_fitness = min(valid_fitnesses)
        adjusted_fitnesses = [
            f - min_fitness + self.min_fitness_offset 
            for f in valid_fitnesses
        ]
        
        total_fitness = sum(adjusted_fitnesses)
        
        if total_fitness <= self.min_fitness_offset:
            return [random.choice(valid_population) for _ in range(num_parents)]
        
        # Calculate cumulative probabilities
        cumulative_probs = []
        cumulative_sum = 0
        for fitness in adjusted_fitnesses:
            cumulative_sum += fitness / total_fitness
            cumulative_probs.append(cumulative_sum)
        
        # Generate evenly spaced pointers
        pointer_distance = 1.0 / num_parents
        start_pointer = np.random.uniform(0, pointer_distance)
        
        parents = []
        for i in range(num_parents):
            pointer = start_pointer + i * pointer_distance
            
            # Find individual corresponding to this pointer
            for j, cum_prob in enumerate(cumulative_probs):
                if pointer <= cum_prob:
                    parents.append(valid_population[j])
                    break
        
        return parents


# Selection strategy registry
_SELECTION_REGISTRY: Dict[str, SelectionStrategy] = {}


def register_selection_strategy(strategy: SelectionStrategy) -> None:
    """
    Register a selection strategy.
    
    Args:
        strategy: SelectionStrategy instance to register
    """
    _SELECTION_REGISTRY[strategy.name] = strategy


def list_selection_strategies() -> List[str]:
    """List all available selection strategy names."""
    return list(_SELECTION_REGISTRY.keys())


def create_selection_strategy(
    name: str, 
    **kwargs
) -> SelectionStrategy:
    """
    Create a selection strategy with parameters using the registry.
    
    Args:
        name: Strategy name
        **kwargs: Strategy-specific parameters
        
    Returns:
        Configured SelectionStrategy instance
    """
    if name not in _SELECTION_REGISTRY:
        available = list(_SELECTION_REGISTRY.keys())
        raise UnsupportedOperationError(
            f"Selection strategy '{name}'",
            context_info="create_selection_strategy",
            alternative=f"Available strategies: {available}"
        )
    
    # Get the strategy instance from registry and configure it
    strategy_instance = _SELECTION_REGISTRY[name]
    
    # For strategies that take constructor parameters, create new instances
    if name == "tournament":
        tournament_size = kwargs.get('tournament_size', 3)
        return TournamentSelection(tournament_size=tournament_size)
    elif name == "roulette":
        min_fitness_offset = kwargs.get('min_fitness_offset', 1e-8)
        return RouletteWheelSelection(min_fitness_offset=min_fitness_offset)
    elif name == "rank":
        selection_pressure = kwargs.get('selection_pressure', 2.0)
        return RankSelection(selection_pressure=selection_pressure)
    elif name == "sus":
        min_fitness_offset = kwargs.get('min_fitness_offset', 1e-8)
        return StochasticUniversalSampling(min_fitness_offset=min_fitness_offset)
    else:
        # For strategies without parameters, return the registry instance
        return strategy_instance


# Initialize default strategies
def _initialize_default_strategies():
    """Initialize and register default selection strategies."""
    strategies = [
        TournamentSelection(),
        RouletteWheelSelection(),
        RankSelection(),
        ElitistSelection(),
        StochasticUniversalSampling()
    ]
    
    for strategy in strategies:
        register_selection_strategy(strategy)


# Initialize default strategies when module is imported
_initialize_default_strategies()


# Utility functions for selection analysis
def analyze_selection_pressure(
    strategy: SelectionStrategy,
    population_size: int = 100,
    num_trials: int = 1000
) -> Dict[str, float]:
    """
    Analyze selection pressure of a strategy.
    
    Args:
        strategy: Selection strategy to analyze
        population_size: Size of test population
        num_trials: Number of selection trials
        
    Returns:
        Dictionary with selection pressure metrics
    """
    from janus_ai.core.expressions.expression import Expression, Variable
    
    # Create dummy population and fitnesses
    dummy_var = Variable("x", 0, {})
    population = [Expression("var", [dummy_var]) for _ in range(population_size)]
    
    # Create fitness gradient (linear from 0 to 1)
    fitnesses = [i / (population_size - 1) for i in range(population_size)]
    
    # Track selection frequency for each individual
    selection_counts = [0] * population_size
    
    for _ in range(num_trials):
        selected = strategy.select(population, fitnesses, num_parents=1)
        selected_idx = population.index(selected[0])
        selection_counts[selected_idx] += 1
    
    # Calculate metrics
    selection_probs = [count / num_trials for count in selection_counts]
    
    # Selection pressure metrics
    best_selection_prob = selection_probs[-1]  # Probability of selecting best
    worst_selection_prob = selection_probs[0]  # Probability of selecting worst
    pressure_ratio = best_selection_prob / worst_selection_prob if worst_selection_prob > 0 else float('inf')
    
    # Diversity metric (entropy)
    entropy = -sum(p * np.log(p + 1e-10) for p in selection_probs if p > 0)
    max_entropy = np.log(population_size)
    normalized_entropy = entropy / max_entropy
    
    return {
        "pressure_ratio": pressure_ratio,
        "best_selection_prob": best_selection_prob,
        "worst_selection_prob": worst_selection_prob,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "selection_variance": np.var(selection_probs)
    }
