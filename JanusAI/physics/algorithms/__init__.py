"""
janus/physics/algorithms/__init__.py

Physics algorithm implementations for the Janus project.
Includes symbolic discovery, reinforcement learning, and model-based approaches.
"""

# Symbolic discovery algorithms
from .genetic import SymbolicRegressor, GeneticAlgorithm
from .ace import ACEAlgorithm  # Placeholder for ACE implementation

# Reinforcement learning algorithms
from .reinforcement import (
    BaseRLAgent, 
    SoftActorCritic, 
    TD3,
    DistributionalSAC,
    SteinSAC
)

# Model-based algorithms
from .model_based import MuZeroAgent, DreamerV2Agent  # Placeholders

# Hybrid algorithms
from .hybrid import CEMRLAgent, EvolutionaryRLHybrid

# Configuration classes
from .config import AlgorithmConfig, GeneticConfig, RLConfig

__all__ = [
    # Symbolic discovery
    "SymbolicRegressor",
    "GeneticAlgorithm", 
    "ACEAlgorithm",
    
    # Reinforcement learning
    "BaseRLAgent",
    "SoftActorCritic",
    "TD3", 
    "DistributionalSAC",
    "SteinSAC",
    
    # Model-based
    "MuZeroAgent",
    "DreamerV2Agent",
    
    # Hybrid
    "CEMRLAgent",
    "EvolutionaryRLHybrid",
    
    # Configuration
    "AlgorithmConfig",
    "GeneticConfig", 
    "RLConfig",
]

# Lazy imports for heavy dependencies
def _import_model_based():
    """Lazy import model-based algorithms to avoid heavy dependencies."""
    try:
        from .model_based import MuZeroAgent, DreamerV2Agent
        return MuZeroAgent, DreamerV2Agent
    except ImportError as e:
        raise ImportError(f"Model-based algorithms require additional dependencies: {e}")

def _import_distributed_rl():
    """Lazy import distributed RL algorithms."""
    try:
        from .distributed import IMPALAAgent, R2D2Agent, D4PGAgent
        return IMPALAAgent, R2D2Agent, D4PGAgent
    except ImportError as e:
        raise ImportError(f"Distributed RL algorithms require additional dependencies: {e}")

# Factory functions
def create_algorithm(algorithm_type: str, config, **kwargs):
    """
    Factory function to create algorithms.
    
    Args:
        algorithm_type: Type of algorithm ('genetic', 'sac', 'td3', etc.)
        config: Algorithm configuration
        **kwargs: Additional arguments
        
    Returns:
        Algorithm instance
    """
    algorithm_map = {
        'genetic': SymbolicRegressor,
        'sac': SoftActorCritic,
        'td3': TD3,
        'dsac': DistributionalSAC,
        's2ac': SteinSAC,
        'cemrl': CEMRLAgent,
    }
    
    if algorithm_type not in algorithm_map:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    return algorithm_map[algorithm_type](config=config, **kwargs)

def list_available_algorithms():
    """List all available algorithms by category."""
    return {
        'symbolic': ['genetic', 'ace'],
        'reinforcement_learning': ['sac', 'td3', 'dsac', 's2ac', 'ppo'],
        'model_based': ['muzero', 'dreamer'],
        'hybrid': ['cemrl', 'evolutionary_rl']
    }


# =============================================================================
# Genetic Algorithm Implementation
# =============================================================================

"""
janus/physics/algorithms/genetic.py

Genetic algorithm implementation for symbolic regression.
"""

import numpy as np
import random
from typing import List, Tuple, Any, Optional, Callable, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

@dataclass
class GeneticConfig:
    """Configuration for genetic algorithms."""
    population_size: int = 100
    max_generations: int = 50
    tournament_size: int = 3
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    max_expression_depth: int = 6
    max_expression_length: int = 20
    parsimony_coefficient: float = 0.001
    fitness_threshold: float = 0.99
    diversity_weight: float = 0.1
    n_jobs: int = 1
    random_seed: Optional[int] = None


class Individual:
    """Represents an individual in the genetic algorithm population."""
    
    def __init__(self, expression: Any, fitness: float = 0.0):
        self.expression = expression
        self.fitness = fitness
        self.age = 0
        self.parents = []
        self.complexity = 0
        
    def __str__(self):
        return f"Individual(expr={self.expression}, fitness={self.fitness:.4f})"
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def copy(self):
        """Create a deep copy of the individual."""
        new_individual = Individual(copy.deepcopy(self.expression), self.fitness)
        new_individual.age = self.age
        new_individual.parents = self.parents.copy()
        new_individual.complexity = self.complexity
        return new_individual


class GeneticOperator(ABC):
    """Abstract base class for genetic operators."""
    
    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """Apply the genetic operator."""
        pass


class Mutation(GeneticOperator):
    """Mutation operator for symbolic expressions."""
    
    def __init__(self, grammar, mutation_rate: float = 0.1):
        self.grammar = grammar
        self.mutation_rate = mutation_rate
    
    def apply(self, individual: Individual) -> Individual:
        """Apply mutation to an individual."""
        if random.random() > self.mutation_rate:
            return individual.copy()
        
        # Create mutated copy
        mutated = individual.copy()
        
        # Apply mutation based on expression type
        if hasattr(mutated.expression, 'mutate'):
            mutated.expression = mutated.expression.mutate(self.grammar)
        else:
            # Fallback: random modification
            mutated.expression = self._random_modify(mutated.expression)
        
        # Reset fitness (will be recalculated)
        mutated.fitness = 0.0
        mutated.age = 0
        mutated.parents = [individual]
        
        return mutated
    
    def _random_modify(self, expression):
        """Random modification fallback."""
        # This would be implemented based on expression representation
        return expression


class Crossover(GeneticOperator):
    """Crossover operator for symbolic expressions."""
    
    def __init__(self, grammar, crossover_rate: float = 0.7):
        self.grammar = grammar
        self.crossover_rate = crossover_rate
    
    def apply(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Apply crossover to two parents."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Create offspring
        child1 = Individual(self._crossover_expressions(parent1.expression, parent2.expression))
        child2 = Individual(self._crossover_expressions(parent2.expression, parent1.expression))
        
        # Set parent information
        child1.parents = [parent1, parent2]
        child2.parents = [parent2, parent1]
        
        return child1, child2
    
    def _crossover_expressions(self, expr1, expr2):
        """Perform crossover between two expressions."""
        if hasattr(expr1, 'crossover'):
            return expr1.crossover(expr2, self.grammar)
        else:
            # Fallback: return one of the expressions
            return expr1 if random.random() < 0.5 else expr2


class Selection(GeneticOperator):
    """Selection operator for genetic algorithms."""
    
    def __init__(self, tournament_size: int = 3, diversity_weight: float = 0.1):
        self.tournament_size = tournament_size
        self.diversity_weight = diversity_weight
    
    def apply(self, population: List[Individual], num_select: int) -> List[Individual]:
        """Select individuals from population."""
        selected = []
        
        for _ in range(num_select):
            if self.diversity_weight > 0:
                individual = self._diversity_tournament(population, selected)
            else:
                individual = self._tournament_selection(population)
            selected.append(individual)
        
        return selected
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Standard tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _diversity_tournament(self, population: List[Individual], 
                            already_selected: List[Individual]) -> Individual:
        """Tournament selection with diversity consideration."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        # Calculate diversity-adjusted fitness
        for individual in tournament:
            diversity_bonus = self._calculate_diversity_bonus(individual, already_selected)
            individual.adjusted_fitness = individual.fitness + self.diversity_weight * diversity_bonus
        
        return max(tournament, key=lambda x: getattr(x, 'adjusted_fitness', x.fitness))
    
    def _calculate_diversity_bonus(self, individual: Individual, 
                                 selected: List[Individual]) -> float:
        """Calculate diversity bonus for an individual."""
        if not selected:
            return 0.0
        
        # Simple diversity measure based on expression similarity
        similarities = []
        for other in selected:
            similarity = self._expression_similarity(individual.expression, other.expression)
            similarities.append(similarity)
        
        # Return negative of average similarity (higher bonus for more diverse)
        return -np.mean(similarities)
    
    def _expression_similarity(self, expr1, expr2) -> float:
        """Calculate similarity between two expressions."""
        # Simple string-based similarity
        str1, str2 = str(expr1), str(expr2)
        if str1 == str2:
            return 1.0
        
        # Jaccard similarity of character sets
        set1, set2 = set(str1), set(str2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class GeneticAlgorithm:
    """
    Core genetic algorithm implementation for symbolic regression.
    """
    
    def __init__(self, 
                 config: GeneticConfig,
                 grammar: Any,
                 fitness_function: Callable[[Any], float],
                 expression_generator: Callable[[], Any]):
        """
        Initialize genetic algorithm.
        
        Args:
            config: Genetic algorithm configuration
            grammar: Grammar for expression generation
            fitness_function: Function to evaluate expression fitness
            expression_generator: Function to generate random expressions
        """
        self.config = config
        self.grammar = grammar
        self.fitness_function = fitness_function
        self.expression_generator = expression_generator
        
        # Initialize operators
        self.mutation = Mutation(grammar, config.mutation_rate)
        self.crossover = Crossover(grammar, config.crossover_rate)
        self.selection = Selection(config.tournament_size, config.diversity_weight)
        
        # Population and statistics
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        
        # Set random seed
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_population(self):
        """Initialize the population with random individuals."""
        self.logger.info(f"Initializing population of size {self.config.population_size}")
        
        self.population = []
        for _ in range(self.config.population_size):
            expression = self.expression_generator()
            individual = Individual(expression)
            self.population.append(individual)
        
        # Evaluate initial population
        self._evaluate_population()
        self._update_statistics()
    
    def evolve(self) -> Tuple[List[Individual], List[float]]:
        """Run one generation of evolution."""
        if not self.population:
            self.initialize_population()
        
        # Selection
        num_elite = int(self.config.elitism_rate * self.config.population_size)
        num_offspring = self.config.population_size - num_elite
        
        # Elite individuals (best performers carry over unchanged)
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:num_elite]
        
        # Generate offspring
        offspring = []
        parents = self.selection.apply(self.population, num_offspring)
        
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = self.crossover.apply(parent1, parent2)
            
            # Apply mutation
            child1 = self.mutation.apply(child1)
            child2 = self.mutation.apply(child2)
            
            offspring.extend([child1, child2])
        
        # Handle odd number of offspring
        if len(offspring) < num_offspring:
            parent = random.choice(parents)
            child = self.mutation.apply(parent)
            offspring.append(child)
        
        # Trim to exact size
        offspring = offspring[:num_offspring]
        
        # Create new population
        self.population = elite + offspring
        
        # Age individuals
        for individual in self.population:
            individual.age += 1
        
        # Evaluate new individuals
        self._evaluate_population()
        self._update_statistics()
        
        self.generation += 1
        
        # Return population and fitness scores
        fitness_scores = [ind.fitness for ind in self.population]
        return self.population.copy(), fitness_scores
    
    def _evaluate_population(self):
        """Evaluate fitness for all individuals in population."""
        if self.config.n_jobs == 1:
            # Sequential evaluation
            for individual in self.population:
                if individual.fitness == 0.0:  # Only evaluate if not already evaluated
                    individual.fitness = self._evaluate_individual(individual)
        else:
            # Parallel evaluation
            unevaluated = [ind for ind in self.population if ind.fitness == 0.0]
            if unevaluated:
                with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                    futures = [executor.submit(self._evaluate_individual, ind) 
                             for ind in unevaluated]
                    
                    for individual, future in zip(unevaluated, futures):
                        try:
                            individual.fitness = future.result()
                        except Exception as e:
                            self.logger.warning(f"Evaluation failed for individual: {e}")
                            individual.fitness = -float('inf')
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """Evaluate a single individual."""
        try:
            # Calculate base fitness
            fitness = self.fitness_function(individual.expression)
            
            # Apply parsimony pressure (penalty for complexity)
            complexity = self._calculate_complexity(individual.expression)
            individual.complexity = complexity
            
            parsimony_penalty = self.config.parsimony_coefficient * complexity
            fitness_with_parsimony = fitness - parsimony_penalty
            
            return max(fitness_with_parsimony, -float('inf'))
            
        except Exception as e:
            self.logger.warning(f"Fitness evaluation failed: {e}")
            return -float('inf')
    
    def _calculate_complexity(self, expression) -> float:
        """Calculate expression complexity."""
        if hasattr(expression, 'complexity'):
            return expression.complexity
        else:
            # Fallback: length-based complexity
            return len(str(expression))
    
    def _update_statistics(self):
        """Update algorithm statistics."""
        if not self.population:
            return
        
        # Find best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
        
        # Record fitness statistics
        fitness_values = [ind.fitness for ind in self.population]
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': max(fitness_values),
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'worst_fitness': min(fitness_values)
        })
    
    def should_terminate(self) -> bool:
        """Check if algorithm should terminate."""
        if self.generation >= self.config.max_generations:
            return True
        
        if self.best_individual and self.best_individual.fitness >= self.config.fitness_threshold:
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_individual.fitness if self.best_individual else 0.0,
            'best_expression': str(self.best_individual.expression) if self.best_individual else None,
            'fitness_history': self.fitness_history
        }


class SymbolicRegressor:
    """
    High-level interface for symbolic regression using genetic algorithms.
    
    This class provides a scikit-learn-like interface for symbolic regression.
    """
    
    def __init__(self, 
                 config: Optional[GeneticConfig] = None,
                 grammar: Optional[Any] = None):
        """Initialize symbolic regressor."""
        self.config = config or GeneticConfig()
        self.grammar = grammar
        self.genetic_algorithm = None
        self.is_fitted = False
        
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SymbolicRegressor':
        """
        Fit symbolic regressor to data.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            
        Returns:
            self: Fitted regressor
        """
        self.logger.info(f"Fitting symbolic regressor to data: X{X.shape}, y{y.shape}")
        
        # Store data
        self.X_train = X
        self.y_train = y
        
        # Create fitness function
        fitness_function = self._create_fitness_function(X, y)
        
        # Create expression generator
        expression_generator = self._create_expression_generator()
        
        # Initialize genetic algorithm
        self.genetic_algorithm = GeneticAlgorithm(
            config=self.config,
            grammar=self.grammar,
            fitness_function=fitness_function,
            expression_generator=expression_generator
        )
        
        # Run evolution
        self.genetic_algorithm.initialize_population()
        
        while not self.genetic_algorithm.should_terminate():
            population, fitness_scores = self.genetic_algorithm.evolve()
            
            # Log progress
            if self.genetic_algorithm.generation % 10 == 0:
                stats = self.genetic_algorithm.get_statistics()
                self.logger.info(
                    f"Generation {stats['generation']}: "
                    f"Best fitness = {stats['best_fitness']:.4f}, "
                    f"Best expr = {stats['best_expression']}"
                )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best discovered expression."""
        if not self.is_fitted:
            raise ValueError("Regressor must be fitted before making predictions")
        
        if self.genetic_algorithm.best_individual is None:
            raise ValueError("No valid expression found during fitting")
        
        # Evaluate best expression on input data
        best_expression = self.genetic_algorithm.best_individual.expression
        
        try:
            if hasattr(best_expression, 'evaluate'):
                predictions = best_expression.evaluate(X)
            else:
                # Fallback evaluation method
                predictions = self._evaluate_expression(best_expression, X)
            
            return np.array(predictions)
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.zeros(X.shape[0])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score on test data."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def get_best_expression(self) -> Any:
        """Get the best discovered expression."""
        if not self.is_fitted or self.genetic_algorithm.best_individual is None:
            return None
        return self.genetic_algorithm.best_individual.expression
    
    def _create_fitness_function(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create fitness function based on training data."""
        def fitness_func(expression):
            try:
                if hasattr(expression, 'evaluate'):
                    predictions = expression.evaluate(X)
                else:
                    predictions = self._evaluate_expression(expression, X)
                
                # Calculate R² score
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                return max(r2, 0.0)  # Ensure non-negative fitness
                
            except Exception:
                return 0.0  # Invalid expressions get zero fitness
        
        return fitness_func
    
    def _create_expression_generator(self) -> Callable:
        """Create function to generate random expressions."""
        def generator():
            if self.grammar and hasattr(self.grammar, 'generate_random_expression'):
                return self.grammar.generate_random_expression(
                    max_depth=self.config.max_expression_depth
                )
            else:
                # Fallback: create simple random expression
                return f"x0 + {np.random.randn():.3f}"
        
        return generator
    
    def _evaluate_expression(self, expression, X: np.ndarray) -> np.ndarray:
        """Fallback expression evaluation."""
        # This would need to be implemented based on expression representation
        return np.random.randn(X.shape[0])  # Placeholder


# Configuration classes
@dataclass 
class AlgorithmConfig:
    """Base configuration for all algorithms."""
    algorithm_type: str
    random_seed: Optional[int] = None
    verbose: bool = True


@dataclass
class RLConfig(AlgorithmConfig):
    """Configuration for RL algorithms."""
    algorithm_type: str = 'sac'
    learning_rate: float = 3e-4
    buffer_size: int = 10000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1


# Example usage
if __name__ == "__main__":
    # Test genetic algorithm
    config = GeneticConfig(
        population_size=20,
        max_generations=10,
        mutation_rate=0.2
    )
    
    print(f"Created genetic config: {config}")
    print(f"Available algorithms: {list_available_algorithms()}")