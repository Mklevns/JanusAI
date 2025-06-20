# janus/core/search/__init__.py
"""
Janus Core Search Module

This package provides modular components for genetic algorithm-based search
and optimization, specifically designed for symbolic regression tasks.

The module has been refactored from a monolithic implementation to provide:
- Clean separation of concerns
- High performance with caching and parallelization
- Extensible architecture with registry patterns
- Comprehensive configuration and statistics tracking
- Robust error handling and logging

Main Components:
- SymbolicRegressor: Main genetic algorithm implementation
- GAConfig: Configuration management
- SelectionStrategy: Various selection strategies
- GeneticOperators: Crossover and mutation operators
- StatsTracker: Comprehensive statistics and monitoring
- FitnessCache: Performance optimization through caching
"""

# Import main classes and functions for easy access
from janus.core.search.config import (
    GAConfig,
    PerformanceConfig,  # Deprecated but kept for compatibility
    ExpressionConfig,
    create_default_config,
    create_fast_config,
    create_thorough_config,
    create_production_config,
    create_default_expression_config,
    create_conservative_expression_config,
    create_exploratory_expression_config
)

from janus.core.search.stats import (
    StatsTracker,
    SearchStats,
    GenerationStats,
    create_stats_tracker,
    analyze_search_run
)

from janus.core.search.selection import (
    SelectionStrategy,
    TournamentSelection,
    RouletteWheelSelection,
    RankSelection,
    ElitistSelection,
    StochasticUniversalSampling,
    create_selection_strategy,
    list_selection_strategies,
    analyze_selection_pressure
)

from janus.core.search.operators import (
    ExpressionGenerator,
    CrossoverOperator,
    MutationOperator,
    SubtreeCrossover,
    UniformCrossover,
    NodeReplacementMutation,
    SubtreeReplacementMutation,
    ConstantPerturbationMutation,
    OperatorMutation,
    create_crossover_operator,
    create_mutation_operator,
    list_operators
)

# Import main regressor from the parent module to maintain clean imports
# This allows: from janus.core.search import SymbolicRegressor
from janus.physics.algorithms.genetic import (
    SymbolicRegressor,
    FitnessCache,
    create_regressor_from_config,
    run_symbolic_regression,
    create_default_fitness_function
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Janus Development Team"
__description__ = "Modular genetic algorithm components for symbolic regression"

# Define public API
__all__ = [
    # Main classes
    "SymbolicRegressor",
    "FitnessCache",
    
    # Configuration
    "GAConfig",
    "PerformanceConfig",  # Deprecated
    "ExpressionConfig",
    "create_default_config",
    "create_fast_config",
    "create_thorough_config",
    "create_production_config",
    "create_default_expression_config",
    "create_conservative_expression_config", 
    "create_exploratory_expression_config",
    
    # Statistics
    "StatsTracker",
    "SearchStats",
    "GenerationStats",
    "create_stats_tracker",
    "analyze_search_run",
    
    # Selection strategies
    "SelectionStrategy",
    "TournamentSelection",
    "RouletteWheelSelection", 
    "RankSelection",
    "ElitistSelection",
    "StochasticUniversalSampling",
    "create_selection_strategy",
    "list_selection_strategies",
    "analyze_selection_pressure",
    
    # Genetic operators
    "ExpressionGenerator",
    "CrossoverOperator",
    "MutationOperator",
    "SubtreeCrossover",
    "UniformCrossover",
    "NodeReplacementMutation",
    "SubtreeReplacementMutation",
    "ConstantPerturbationMutation", 
    "OperatorMutation",
    "create_crossover_operator",
    "create_mutation_operator",
    "list_operators",
    
    # Utility functions
    "create_regressor_from_config",
    "run_symbolic_regression",
    "create_default_fitness_function"
]


def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "author": __author__, 
        "description": __description__,
        "components": {
            "config": "Configuration management with validation",
            "stats": "Comprehensive statistics tracking and analysis",
            "selection": "Multiple selection strategies with registry pattern",
            "operators": "Genetic operators for crossover and mutation",
            "regressor": "Main genetic algorithm implementation with optimizations"
        },
        "improvements": [
            "Fixed critical early stopping bug",
            "Added lambdify caching for 10-100x speedup",
            "Implemented parallel fitness evaluation",
            "Modular architecture with separation of concerns",
            "Registry pattern for extensibility",
            "Comprehensive error handling",
            "Extensive configuration options",
            "Performance monitoring and statistics"
        ]
    }


def create_example_regressor(grammar, mode="balanced"):
    """
    Create an example regressor with common configurations.
    
    Args:
        grammar: BaseGrammar instance
        mode: "fast", "balanced", or "thorough"
        
    Returns:
        Configured SymbolicRegressor instance
    """
    if mode == "fast":
        config = create_fast_config()
    elif mode == "thorough":
        config = create_thorough_config()
    else:  # balanced
        config = create_default_config()
        config.population_size = 100
        config.generations = 50
        config.enable_caching = True
        config.enable_parallel = True
    
    return SymbolicRegressor(grammar=grammar, config=config)


def benchmark_selection_strategies(population_size=100, num_trials=1000):
    """
    Benchmark different selection strategies.
    
    Args:
        population_size: Size of test population
        num_trials: Number of selection trials
        
    Returns:
        Dictionary with benchmark results for each strategy
    """
    strategies = [
        ("tournament", TournamentSelection()),
        ("roulette", RouletteWheelSelection()),
        ("rank", RankSelection()),
        ("elitist", ElitistSelection()),
        ("sus", StochasticUniversalSampling())
    ]
    
    results = {}
    
    for name, strategy in strategies:
        try:
            analysis = analyze_selection_pressure(
                strategy, population_size, num_trials
            )
            results[name] = analysis
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return results


# Module-level configuration for common use cases
DEFAULT_CONFIG = create_default_config()
FAST_CONFIG = create_fast_config()
THOROUGH_CONFIG = create_thorough_config()
PRODUCTION_CONFIG = create_production_config()


# Convenience function for quick testing
def quick_test(X, y, grammar, variable_names=None, verbose=True):
    """
    Quick test function for immediate symbolic regression.
    
    Args:
        X: Input data
        y: Target data  
        grammar: Grammar to use
        variable_names: Variable names
        verbose: Whether to show progress
        
    Returns:
        (best_expression, fitness, duration)
    """
    import time
    
    config = create_fast_config()
    config.verbose = verbose
    
    regressor = SymbolicRegressor(grammar=grammar, config=config)
    
    start_time = time.time()
    best_expr = regressor.fit(X, y, variable_names)
    duration = time.time() - start_time
    
    return best_expr, regressor.best_fitness, duration
