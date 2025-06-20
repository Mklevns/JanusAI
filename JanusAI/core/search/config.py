"""
Configuration classes for genetic algorithm search components.

This module provides clean configuration objects for genetic algorithm
parameters, replacing scattered initialization parameters.
"""

from dataclasses import dataclass
from typing import Optional, Callable, List, Any, Tuple
import numpy as np


@dataclass
class GAConfig:
    """
    Configuration for genetic algorithm parameters.
    
    This dataclass bundles all GA hyperparameters into a single, 
    easy-to-manage configuration object.
    """
    
    # Population parameters
    population_size: int = 100
    generations: int = 50
    
    # Genetic operator rates
    mutation_rate: float = 0.1
    crossover_rate: float = 0.9
    
    # Selection parameters
    elitism_size: int = 2
    tournament_size: int = 3
    selection_strategy: str = "tournament"  # "tournament", "roulette", "rank"
    
    # Expression generation parameters
    max_depth: int = 6
    max_complexity: Optional[int] = None
    
    # Early stopping parameters
    early_stopping_threshold: float = 1e-10
    early_stopping_generations: int = 10
    
    # Performance parameters - UNIFIED
    enable_caching: bool = True
    enable_parallel: bool = True
    parallel_backend: str = "threading"  # "threading", "multiprocessing", "joblib"
    n_jobs: int = -1  # -1 for all available cores
    batch_size: Optional[int] = None
    
    # Logging and output
    verbose: bool = False
    random_state: Optional[int] = None
    
    # Fitness function configuration
    complexity_weight: float = 0.01
    parsimony_pressure: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        
        if self.generations <= 0:
            raise ValueError("generations must be positive")
            
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
            
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be between 0 and 1")
            
        if self.elitism_size < 0:
            raise ValueError("elitism_size must be non-negative")
            
        if self.elitism_size >= self.population_size:
            raise ValueError("elitism_size must be less than population_size")
            
        if self.tournament_size <= 0:
            raise ValueError("tournament_size must be positive")
            
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
            
        if self.early_stopping_threshold < 0:
            raise ValueError("early_stopping_threshold must be non-negative")
            
        if self.early_stopping_generations <= 0:
            raise ValueError("early_stopping_generations must be positive")
            
        valid_strategies = ["tournament", "roulette", "rank"]
        if self.selection_strategy not in valid_strategies:
            raise ValueError(f"selection_strategy must be one of {valid_strategies}")
        
        # Validate parallel backend
        valid_backends = ["threading", "multiprocessing", "joblib"]
        if self.parallel_backend not in valid_backends:
            raise ValueError(f"parallel_backend must be one of {valid_backends}")


@dataclass
class PerformanceConfig:
    """
    DEPRECATED: Performance configuration - use GAConfig instead.
    
    This class is maintained for backward compatibility but most settings
    have been moved to GAConfig.
    """
    
    # Memory management
    cache_size_limit: int = 10000
    enable_garbage_collection: bool = True
    gc_frequency: int = 10  # Run GC every N generations
    
    # Numerical stability
    inf_penalty: float = -1e6
    nan_penalty: float = -1e6
    overflow_threshold: float = 1e10
    
    def __post_init__(self):
        """Show deprecation warning."""
        import warnings
        warnings.warn(
            "PerformanceConfig is deprecated. Use GAConfig for performance settings.",
            DeprecationWarning,
            stacklevel=2
        )


@dataclass
class ExpressionConfig:
    """Configuration for expression generation and evaluation."""
    
    # Generation parameters
    terminal_probability: float = 0.3
    constant_range: Tuple[float, float] = (-5.0, 5.0)
    constant_precision: int = 3
    
    # Variable vs constant preference
    variable_probability: float = 0.7  # When generating terminals
    
    # Complexity control
    enable_complexity_limit: bool = True
    complexity_penalty_type: str = "linear"  # "linear", "exponential", "logarithmic"
    
    # Evaluation parameters
    numerical_tolerance: float = 1e-12
    max_evaluation_time: float = 1.0  # seconds
    
    # Simplification
    enable_simplification: bool = True
    simplification_level: str = "basic"  # "none", "basic", "aggressive"
    
    # Grammar-driven generation
    use_grammar_probabilities: bool = True
    fallback_operators: List[str] = None  # Default operators if grammar fails
    
    def __post_init__(self):
        """Validate and set defaults."""
        if not 0 <= self.terminal_probability <= 1:
            raise ValueError("terminal_probability must be between 0 and 1")
        
        if not 0 <= self.variable_probability <= 1:
            raise ValueError("variable_probability must be between 0 and 1")
        
        if self.constant_range[0] >= self.constant_range[1]:
            raise ValueError("constant_range must be (min, max) with min < max")
        
        if self.constant_precision < 0:
            raise ValueError("constant_precision must be non-negative")
        
        if self.fallback_operators is None:
            self.fallback_operators = ['+', '-', '*', '/', 'sin', 'cos']
        
        valid_complexity_types = ["linear", "exponential", "logarithmic"]
        if self.complexity_penalty_type not in valid_complexity_types:
            raise ValueError(f"complexity_penalty_type must be one of {valid_complexity_types}")
        
        valid_simplification = ["none", "basic", "aggressive"]
        if self.simplification_level not in valid_simplification:
            raise ValueError(f"simplification_level must be one of {valid_simplification}")


def create_default_expression_config() -> ExpressionConfig:
    """Create default expression configuration."""
    return ExpressionConfig()


def create_conservative_expression_config() -> ExpressionConfig:
    """Create conservative expression configuration (simpler expressions)."""
    return ExpressionConfig(
        terminal_probability=0.5,
        constant_range=(-2.0, 2.0),
        enable_complexity_limit=True,
        simplification_level="basic"
    )


def create_exploratory_expression_config() -> ExpressionConfig:
    """Create exploratory expression configuration (more complex expressions)."""
    return ExpressionConfig(
        terminal_probability=0.2,
        constant_range=(-10.0, 10.0),
        enable_complexity_limit=False,
        simplification_level="none"
    )


def create_default_config() -> GAConfig:
    """Create a default GA configuration suitable for most tasks."""
    return GAConfig()


def create_fast_config() -> GAConfig:
    """Create a configuration optimized for speed over accuracy."""
    return GAConfig(
        population_size=50,
        generations=25,
        max_depth=4,
        enable_caching=True,
        enable_parallel=True,
        early_stopping_generations=5
    )


def create_thorough_config() -> GAConfig:
    """Create a configuration optimized for solution quality."""
    return GAConfig(
        population_size=200,
        generations=100,
        max_depth=8,
        mutation_rate=0.15,
        crossover_rate=0.85,
        elitism_size=5,
        early_stopping_generations=20,
        complexity_weight=0.005
    )


def create_production_config() -> GAConfig:
    """Create a configuration suitable for production use."""
    return GAConfig(
        population_size=100,
        generations=50,
        max_depth=6,
        enable_caching=True,
        enable_parallel=True,
        verbose=False,
        early_stopping_threshold=1e-8,
        early_stopping_generations=15
    )
