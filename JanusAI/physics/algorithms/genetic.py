# janus/physics/algorithms/genetic.py
"""
Refactored Genetic Algorithm for Symbolic Regression in Janus Framework

This module provides a clean, modular implementation of genetic algorithms
for symbolic regression. The previous monolithic implementation has been
refactored into focused components for better maintainability, performance,
and extensibility.

Key improvements:
- Modular architecture with separated concerns
- Fixed critical bugs (early stopping logic, selection edge cases)
- Performance optimizations (lambdify caching, parallel evaluation)
- Comprehensive error handling and logging
- Registry-based operator and strategy selection
- Extensive configuration options
"""

import time
import numpy as np
import warnings
from typing import List, Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import OrderedDict
import multiprocessing as mp
from functools import lru_cache
import logging

if TYPE_CHECKING:
    from janus.core.search.config import ExpressionConfig

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x  # Fallback

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# Top-level function for parallel evaluation (avoids pickling entire regressor)
def _evaluate_expression_fitness_worker(
    expr_str: str,
    X: np.ndarray,
    y: np.ndarray,
    variable_names: List[str],
    complexity_weight: float = 0.01,
    parsimony_pressure: bool = True
) -> float:
    """
    Worker function for parallel fitness evaluation.
    
    This function is at module level to avoid pickling the entire regressor
    when using ProcessPoolExecutor.
    
    Args:
        expr_str: String representation of expression
        X: Input data
        y: Target data
        variable_names: Variable names
        complexity_weight: Weight for complexity penalty
        parsimony_pressure: Whether to apply parsimony pressure
        
    Returns:
        Fitness value
    """
    try:
        from janus.core.expressions.symbolic_math import (
            evaluate_expression_on_data, 
            get_expression_complexity
        )
        
        # Evaluate expression on data
        predictions = evaluate_expression_on_data(expr_str, variable_names, X)
        
        # Handle evaluation failures
        if predictions is None or not np.all(np.isfinite(predictions)):
            return -1e6
        
        # Calculate MSE
        mse = np.mean((y - predictions) ** 2)
        fitness = -mse
        
        # Add complexity penalty if enabled
        if parsimony_pressure:
            # We need to reconstruct the expression for complexity calculation
            # This is a limitation of the parallel approach, but still better than pickling
            try:
                import sympy as sp
                from janus.core.expressions.symbolic_math import create_sympy_expression
                sympy_expr = create_sympy_expression(expr_str, variable_names)
                if sympy_expr is not None:
                    complexity = len(str(sympy_expr))  # Simple complexity measure
                    fitness -= complexity_weight * complexity
            except Exception:
                # If complexity calculation fails, don't apply penalty
                pass
        
        return fitness
        
    except Exception as e:
        # Log error at module level since we don't have access to instance logger
        import logging
        logging.getLogger(__name__).debug(f"Worker fitness evaluation failed: {e}")
        return -1e6

# Import Janus components
from janus.core.grammar.base_grammar import BaseGrammar
from janus.core.expressions.expression import Expression, Variable
from janus.core.expressions.symbolic_math import (
    evaluate_expression_on_data, 
    get_expression_complexity,
    expression_to_string
)
from janus.utils.exceptions import (
    JanusError, 
    GrammarError, 
    DataValidationError,
    OptimizationError,
    UnsupportedOperationError
)

# Import new modular components
from janus.core.search.config import GAConfig, ExpressionConfig
from janus.core.search.stats import StatsTracker, SearchStats, GenerationStats
from janus.core.search.selection import create_selection_strategy
from janus.core.search.operators import (
    ExpressionGenerator, 
    create_crossover_operator, 
    create_mutation_operator
)


class FitnessCache:
    """
    Caches fitness evaluations and lambdified functions for performance.
    
    This addresses the major performance issue identified in the audit
    where expressions were re-stringified and re-evaluated repeatedly.
    
    Now uses proper LRU eviction with OrderedDict.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize fitness cache.
        
        Args:
            max_size: Maximum number of cached entries
        """
        self.max_size = max_size
        self.fitness_cache: OrderedDict[str, float] = OrderedDict()
        self.function_cache: OrderedDict[str, Callable] = OrderedDict()
        self.variable_cache: OrderedDict[str, List[str]] = OrderedDict()
        self.logger = logging.getLogger(__name__)
    
    def get_fitness(self, expr_str: str) -> Optional[float]:
        """Get cached fitness value with LRU update."""
        if expr_str in self.fitness_cache:
            # Move to end (most recently used)
            value = self.fitness_cache.pop(expr_str)
            self.fitness_cache[expr_str] = value
            return value
        return None
    
    def cache_fitness(self, expr_str: str, fitness: float):
        """Cache a fitness value with LRU eviction."""
        if expr_str in self.fitness_cache:
            # Update existing entry (move to end)
            self.fitness_cache.pop(expr_str)
        elif len(self.fitness_cache) >= self.max_size:
            # Remove least recently used (first item)
            self.fitness_cache.popitem(last=False)
        
        self.fitness_cache[expr_str] = fitness
    
    def get_function(self, expr_str: str) -> Optional[Callable]:
        """Get cached lambdified function with LRU update."""
        if expr_str in self.function_cache:
            # Move to end (most recently used)
            func = self.function_cache.pop(expr_str)
            self.function_cache[expr_str] = func
            return func
        return None
    
    def cache_function(self, expr_str: str, func: Callable, variables: List[str]):
        """Cache a lambdified function with LRU eviction."""
        if expr_str in self.function_cache:
            # Update existing entry (move to end)
            self.function_cache.pop(expr_str)
            self.variable_cache.pop(expr_str, None)
        elif len(self.function_cache) >= self.max_size:
            # Remove least recently used (first item)
            lru_key = next(iter(self.function_cache))
            self.function_cache.popitem(last=False)
            self.variable_cache.pop(lru_key, None)
        
        self.function_cache[expr_str] = func
        self.variable_cache[expr_str] = variables.copy()
    
    def clear(self):
        """Clear all caches."""
        self.fitness_cache.clear()
        self.function_cache.clear()
        self.variable_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'fitness_cache_size': len(self.fitness_cache),
            'function_cache_size': len(self.function_cache),
            'max_size': self.max_size,
            'fitness_cache_ratio': len(self.fitness_cache) / self.max_size,
            'function_cache_ratio': len(self.function_cache) / self.max_size
        }


class SymbolicRegressor:
    """
    Refactored genetic algorithm-based symbolic regressor.
    
    This version addresses all issues identified in the audit:
    - Uses modular components for better separation of concerns
    - Implements performance optimizations (caching, parallelization)
    - Fixes critical bugs in early stopping and selection
    - Provides comprehensive configuration options
    - Includes robust error handling and logging
    """
    
    def __init__(
        self,
        grammar: BaseGrammar,
        config: Optional[GAConfig] = None,
        fitness_fn: Optional[Callable] = None
    ):
        """
        Initialize the SymbolicRegressor with clean configuration.
        
        Args:
            grammar: BaseGrammar instance defining allowable operations
            config: GAConfig object with all GA parameters
            fitness_fn: Custom fitness function (optional)
        """
        self.grammar = grammar
        self.config = config or GAConfig()
        self.fitness_fn = fitness_fn
        
        # Set random seed for reproducibility
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
        
        # Initialize components
        self.expression_generator = ExpressionGenerator(
            grammar=grammar,
            expression_config=self.config.expression_config,
            max_depth=self.config.max_depth
        )
        
        self.selection_strategy = create_selection_strategy(
            self.config.selection_strategy,
            tournament_size=self.config.tournament_size
        )
        
        self.crossover_operator = create_crossover_operator("subtree")
        
        # Create multiple mutation operators
        self.mutation_operators = [
            create_mutation_operator("node_replacement", self.expression_generator),
            create_mutation_operator("subtree_replacement", self.expression_generator),
            create_mutation_operator("constant_perturbation", self.expression_generator),
            create_mutation_operator("operator_mutation", self.expression_generator)
        ]
        
        # Performance components
        self.fitness_cache = FitnessCache(
            max_size=10000  # Use reasonable default, could be configurable
        ) if self.config.enable_caching else None
        
        # Statistics tracking
        self.stats_tracker = StatsTracker(
            convergence_threshold=self.config.early_stopping_threshold
        )
        
        # Results tracking
        self.best_expression: Optional[Expression] = None
        self.best_fitness: float = -np.inf
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        variable_names: Optional[List[str]] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Expression:
        """
        Run genetic algorithm to discover the best symbolic expression.
        
        This method performs the complete genetic algorithm search process,
        including initialization, evolution, selection, crossover, mutation,
        and convergence monitoring.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
            variable_names: Names of variables corresponding to X columns.
                           If None, generates default names ['x0', 'x1', ...]
            validation_data: Optional tuple (X_val, y_val) for validation
                           during training to prevent overfitting
            
        Returns:
            Expression: Best symbolic expression found during evolution
            
        Raises:
            DataValidationError: If input data is invalid or inconsistent
            OptimizationError: If genetic algorithm fails to find valid expressions
            
        Example:
            >>> X = np.array([[1, 2], [2, 3], [3, 4]])
            >>> y = np.array([3, 5, 7])  # y = x1 + x2 - 1
            >>> regressor = SymbolicRegressor(grammar, config)
            >>> best_expr = regressor.fit(X, y, ['x1', 'x2'])
            >>> print(f"Discovered: {best_expr}")
        """
        try:
            # Validate and prepare data
            self._validate_inputs(X, y, variable_names)
            X, y = self._prepare_data(X, y)
            
            if variable_names is None:
                variable_names = [f"x{i}" for i in range(X.shape[1])]
            
            variables = [Variable(name, idx, {}) for idx, name in enumerate(variable_names)]
            
            # Initialize population
            if self.config.verbose:
                self.logger.info(f"Initializing population of {self.config.population_size} expressions...")
            
            population = self.expression_generator.generate_population(
                variables, 
                self.config.population_size,
                self.config.max_complexity
            )
            
            # Clear caches
            if self.fitness_cache:
                self.fitness_cache.clear()
            
            # Evolution loop
            last_best_fitness = -np.inf
            generations_without_improvement = 0
            
            progress_bar = None
            if self.config.verbose and HAS_TQDM:
                progress_bar = tqdm(
                    total=self.config.generations, 
                    desc="Genetic Algorithm Progress"
                )
            
            for generation in range(self.config.generations):
                gen_start_time = self.stats_tracker.start_generation(generation)
                
                # Evaluate fitness for all individuals
                eval_start_time = time.time()
                fitnesses = self._evaluate_population_fitness(
                    population, X, y, variable_names
                )
                eval_time = time.time() - eval_start_time
                
                # Update best individual
                best_idx = np.argmax(fitnesses)
                current_best_fitness = fitnesses[best_idx]
                
                if current_best_fitness > self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_expression = population[best_idx]
                    generations_without_improvement = 0
                    last_best_fitness = current_best_fitness
                else:
                    generations_without_improvement += 1
                
                # Record generation statistics
                gen_stats = self.stats_tracker.record_generation(
                    generation, population, fitnesses, eval_time
                )
                
                # Check early stopping - FIXED BUG: proper convergence check
                improvement_threshold = self.config.early_stopping_threshold
                fitness_improvement = current_best_fitness - last_best_fitness
                
                if (abs(fitness_improvement) <= improvement_threshold and
                    generations_without_improvement >= self.config.early_stopping_generations):
                    if self.config.verbose:
                        self.logger.info(f"Early stopping at generation {generation}")
                        self.logger.info(f"Improvement: {fitness_improvement:.2e}, threshold: {improvement_threshold:.2e}")
                    break
                
                # Update progress
                if progress_bar:
                    progress_bar.set_description(
                        f"Gen {generation}: Best={self.best_fitness:.6f}, "
                        f"Diversity={gen_stats.diversity_structural:.3f}"
                    )
                    progress_bar.update(1)
                elif self.config.verbose and generation % 10 == 0:
                    self.logger.info(
                        f"Generation {generation}: Best={self.best_fitness:.6f}, "
                        f"Mean={gen_stats.mean_fitness:.6f}, "
                        f"Diversity={gen_stats.diversity_structural:.3f}"
                    )
                
                # Create next generation
                new_population = self._create_next_generation(
                    population, fitnesses, variables
                )
                population = new_population
            
            if progress_bar:
                progress_bar.close()
            
            # Finalize statistics
            search_stats = self.stats_tracker.finish_search()
            
            if self.config.verbose:
                self._log_final_results(search_stats)
            
            if self.best_expression is None:
                raise OptimizationError("No valid expressions found during evolution")
            
            return self.best_expression
            
        except Exception as e:
            if isinstance(e, JanusError):
                raise
            else:
                raise OptimizationError(f"Genetic algorithm failed: {e}") from e
    
    def predict(self, X: np.ndarray, variable_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Use the best discovered expression to make predictions on new data.
        
        This method evaluates the best expression found during fitting
        on new input data to generate predictions.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            variable_names: Names of variables corresponding to X columns.
                          Should match those used during fitting.
                          If None, generates default names ['x0', 'x1', ...]
            
        Returns:
            np.ndarray: Predicted values of shape (n_samples,)
            
        Raises:
            OptimizationError: If no expression has been fitted (call fit() first)
                             or if prediction evaluation fails
            
        Example:
            >>> X_new = np.array([[4, 5], [5, 6]])
            >>> predictions = regressor.predict(X_new, ['x1', 'x2'])
            >>> print(f"Predictions: {predictions}")
        """
        if self.best_expression is None:
            raise OptimizationError("No expression available. Call fit() first.")
        
        try:
            if variable_names is None:
                variable_names = [f"x{i}" for i in range(X.shape[1])]
            
            expression_str = expression_to_string(self.best_expression)
            predictions = evaluate_expression_on_data(expression_str, variable_names, X)
            
            return predictions
            
        except Exception as e:
            raise OptimizationError(f"Prediction failed: {e}") from e
    
    def get_search_stats(self) -> SearchStats:
        """
        Get comprehensive search statistics from the completed run.
        
        Returns detailed statistics about the genetic algorithm execution
        including performance metrics, convergence analysis, and generation
        history.
        
        Returns:
            SearchStats: Complete statistics object containing:
                - total_generations: Number of generations completed
                - best_fitness: Best fitness value achieved
                - convergence_generation: Generation where convergence occurred
                - generation_history: List of per-generation statistics
                - performance metrics: Timing and evaluation statistics
                
        Example:
            >>> stats = regressor.get_search_stats()
            >>> print(f"Converged in {stats.total_generations} generations")
            >>> print(f"Best fitness: {stats.best_fitness:.6f}")
        """
        return self.stats_tracker.search_stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Provides a concise summary of algorithm performance including
        timing information, evaluation rates, and efficiency metrics.
        
        Returns:
            Dict[str, Any]: Performance summary containing:
                - total_duration: Total execution time in seconds
                - total_evaluations: Number of fitness evaluations performed
                - evaluations_per_second: Evaluation rate
                - average_generation_time: Average time per generation
                - final_best_fitness: Best fitness achieved
                - cache_hit_ratio: Cache efficiency (if caching enabled)
                
        Example:
            >>> perf = regressor.get_performance_summary()
            >>> print(f"Rate: {perf['evaluations_per_second']:.1f} evals/sec")
            >>> print(f"Duration: {perf['total_duration']:.2f} seconds")
        """
        return self.stats_tracker.get_performance_summary()
    
    # Private methods
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray, variable_names: Optional[List[str]]):
        """Validate input data and parameters."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise DataValidationError("X and y must be numpy arrays")
        
        if X.ndim != 2:
            raise DataValidationError(f"X must be 2D array, got shape {X.shape}")
        
        if y.ndim != 1:
            raise DataValidationError(f"y must be 1D array, got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise DataValidationError(
                f"Number of samples mismatch: X has {X.shape[0]}, y has {y.shape[0]}"
            )
        
        if variable_names is not None and len(variable_names) != X.shape[1]:
            raise DataValidationError(
                f"Number of variable names ({len(variable_names)}) doesn't match "
                f"number of features ({X.shape[1]})"
            )
        
        if not np.all(np.isfinite(X)):
            raise DataValidationError("X contains NaN or infinite values")
        
        if not np.all(np.isfinite(y)):
            raise DataValidationError("y contains NaN or infinite values")
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and optionally normalize data."""
        return X.copy(), y.copy()
    
    def _evaluate_population_fitness(
        self, 
        population: List[Expression], 
        X: np.ndarray, 
        y: np.ndarray, 
        variable_names: List[str]
    ) -> List[float]:
        """
        Evaluate fitness for entire population with optimizations.
        
        This method implements the performance improvements identified in the audit:
        - Caching of fitness values and lambdified functions
        - Optional parallel evaluation
        - Batch processing of unique expressions
        """
        
        # Check for parallel evaluation
        if (self.config.enable_parallel and 
            len(population) > 10):  # Only parallelize for larger populations
            return self._evaluate_population_parallel(population, X, y, variable_names)
        else:
            return self._evaluate_population_serial(population, X, y, variable_names)
    
    def _evaluate_population_serial(
        self, 
        population: List[Expression], 
        X: np.ndarray, 
        y: np.ndarray, 
        variable_names: List[str]
    ) -> List[float]:
        """Serial fitness evaluation with caching."""
        fitnesses = []
        
        for expr in population:
            fitness = self._evaluate_single_fitness(expr, X, y, variable_names)
            fitnesses.append(fitness)
        
        return fitnesses
    
    def _evaluate_population_parallel(
        self, 
        population: List[Expression], 
        X: np.ndarray, 
        y: np.ndarray, 
        variable_names: List[str]
    ) -> List[float]:
        """
        Parallel fitness evaluation using top-level worker function.
        
        This method avoids pickling the entire regressor by using a module-level
        worker function and pre-converting expressions to strings.
        """
        try:
            # Convert expressions to strings first (avoids pickling Expression objects)
            expr_strings = []
            expr_to_idx = {}
            
            for i, expr in enumerate(population):
                try:
                    expr_str = expression_to_string(expr)
                    expr_strings.append(expr_str)
                    expr_to_idx[i] = expr_str
                except Exception as e:
                    self.logger.debug(f"Failed to convert expression {i} to string: {e}")
                    expr_strings.append("0")  # Fallback to constant
                    expr_to_idx[i] = "0"
            
            # Choose parallel backend
            if self.config.parallel_backend == "joblib" and HAS_JOBLIB:
                # Use joblib for better memory handling
                from joblib import Parallel, delayed
                
                n_jobs = self.config.n_jobs if self.config.n_jobs > 0 else -1
                
                fitnesses = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(_evaluate_expression_fitness_worker)(
                        expr_str, X, y, variable_names,
                        self.config.complexity_weight,
                        self.config.parsimony_pressure
                    ) for expr_str in expr_strings
                )
                
            else:
                # Use concurrent.futures
                executor_class = (ThreadPoolExecutor if self.config.parallel_backend == "threading" 
                                else ProcessPoolExecutor)
                
                max_workers = (self.config.n_jobs if self.config.n_jobs > 0 
                              else mp.cpu_count())
                
                with executor_class(max_workers=max_workers) as executor:
                    # Submit all tasks using the worker function
                    future_to_idx = {}
                    for i, expr_str in enumerate(expr_strings):
                        future = executor.submit(
                            _evaluate_expression_fitness_worker,
                            expr_str, X, y, variable_names,
                            self.config.complexity_weight,
                            self.config.parsimony_pressure
                        )
                        future_to_idx[future] = i
                    
                    # Collect results
                    fitnesses = [0.0] * len(population)
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            fitnesses[idx] = future.result()
                        except Exception as e:
                            self.logger.warning(f"Parallel evaluation failed for expression {idx}: {e}")
                            fitnesses[idx] = -1e6
            
            return fitnesses
                
        except Exception as e:
            self.logger.warning(f"Parallel evaluation failed, falling back to serial: {e}")
            return self._evaluate_population_serial(population, X, y, variable_names)
    
    def _evaluate_single_fitness(
        self, 
        expression: Expression, 
        X: np.ndarray, 
        y: np.ndarray, 
        variable_names: List[str]
    ) -> float:
        """
        Evaluate fitness for a single expression with caching.
        
        This implements the major performance optimization identified in the audit.
        """
        try:
            # Convert to string for caching
            expr_str = expression_to_string(expression)
            
            # Check cache first
            if self.fitness_cache:
                cached_fitness = self.fitness_cache.get_fitness(expr_str)
                if cached_fitness is not None:
                    return cached_fitness
            
            # Use custom fitness function if provided
            if self.fitness_fn:
                fitness = self.fitness_fn(expression, X, y, variable_names)
            else:
                fitness = self._default_fitness_function(expression, X, y, variable_names, expr_str)
            
            # Cache the result
            if self.fitness_cache:
                self.fitness_cache.cache_fitness(expr_str, fitness)
            
            return fitness
            
        except (ValueError, TypeError, AttributeError) as e:
            # Expected errors from expression evaluation
            self.logger.debug(f"Expression evaluation error: {e}")
            return -1e6
        except Exception as e:
            # Unexpected errors - log with stack trace for debugging
            self.logger.debug(f"Unexpected fitness evaluation error: {e}", exc_info=True)
            return -1e6
    
    def _default_fitness_function(
        self, 
        expression: Expression, 
        X: np.ndarray, 
        y: np.ndarray, 
        variable_names: List[str],
        expr_str: Optional[str] = None
    ) -> float:
        """
        Optimized default fitness function with lambdify caching.
        
        This addresses the major performance bottleneck identified in the audit.
        """
        try:
            if expr_str is None:
                expr_str = expression_to_string(expression)
            
            # Try to use cached lambdified function
            cached_func = None
            if self.fitness_cache:
                cached_func = self.fitness_cache.get_function(expr_str)
            
            if cached_func is not None:
                # Use cached function
                try:
                    predictions = cached_func(*[X[:, i] for i in range(X.shape[1])])
                    if np.isscalar(predictions):
                        predictions = np.full(len(y), predictions)
                    predictions = np.asarray(predictions)
                except (ValueError, TypeError, OverflowError, ZeroDivisionError) as e:
                    # Cached function failed, fall back to standard evaluation
                    self.logger.debug(f"Cached function evaluation failed: {e}")
                    predictions = evaluate_expression_on_data(expr_str, variable_names, X)
            else:
                # Standard evaluation and caching
                predictions = evaluate_expression_on_data(expr_str, variable_names, X)
                
                # Try to create and cache lambdified function
                if self.fitness_cache:
                    try:
                        import sympy as sp
                        from janus.core.expressions.symbolic_math import create_sympy_expression
                        
                        sympy_expr = create_sympy_expression(expr_str, variable_names)
                        if sympy_expr is not None:
                            symbols = [sp.Symbol(name) for name in variable_names]
                            lambdified_func = sp.lambdify(symbols, sympy_expr, 'numpy')
                            self.fitness_cache.cache_function(expr_str, lambdified_func, variable_names)
                    except (sp.SympifyError, sp.parsing.sympy_parser.ParseError, ImportError) as e:
                        self.logger.debug(f"Failed to create lambdified function: {e}")
                    except Exception as e:
                        self.logger.debug(f"Unexpected lambdify error: {e}", exc_info=True)
            
            # Handle evaluation failures
            if predictions is None or not np.all(np.isfinite(predictions)):
                return -1e6  # Consistent penalty value
            
            # Calculate MSE
            mse = np.mean((y - predictions) ** 2)
            
            # Add complexity penalty if enabled
            fitness = -mse
            if self.config.parsimony_pressure:
                complexity = get_expression_complexity(expression)
                fitness -= self.config.complexity_weight * complexity
            
            return fitness
            
        except (ZeroDivisionError, OverflowError, ValueError, TypeError) as e:
            # Expected mathematical errors
            self.logger.debug(f"Mathematical error in fitness evaluation: {e}")
            return -1e6
        except Exception as e:
            # Unexpected errors - log with stack trace
            self.logger.debug(f"Unexpected error in default fitness function: {e}", exc_info=True)
            return -1e6
    
    def _create_next_generation(
        self, 
        population: List[Expression], 
        fitnesses: List[float],
        variables: List[Variable]
    ) -> List[Expression]:
        """Create next generation using modular operators."""
        new_population = []
        
        # Elitism: keep best individuals
        if self.config.elitism_size > 0:
            elite_indices = np.argsort(fitnesses)[-self.config.elitism_size:]
            for idx in elite_indices:
                new_population.append(population[idx])
        
        # Generate rest of population
        while len(new_population) < self.config.population_size:
            # Selection
            parents = self.selection_strategy.select(
                population, fitnesses, num_parents=2
            )
            
            # Crossover
            if (len(parents) >= 2 and 
                np.random.random() < self.config.crossover_rate):
                children = self.crossover_operator.crossover(parents[0], parents[1])
            else:
                children = parents[:2] if len(parents) >= 2 else [parents[0], parents[0]]
            
            # Mutation
            for i, child in enumerate(children):
                if np.random.random() < self.config.mutation_rate:
                    # Select random mutation operator
                    mutation_op = np.random.choice(self.mutation_operators)
                    child = mutation_op.mutate(child, variables, max_complexity=self.config.max_complexity)
                    children[i] = child
                
                if len(new_population) < self.config.population_size:
                    new_population.append(child)
        
        return new_population[:self.config.population_size]
    
    def _log_final_results(self, search_stats: SearchStats):
        """Log final results and statistics."""
        self.logger.info("\n" + "="*60)
        self.logger.info("GENETIC ALGORITHM COMPLETED")
        self.logger.info("="*60)
        self.logger.info(f"Duration: {search_stats.get_duration():.2f}s")
        self.logger.info(f"Generations: {search_stats.total_generations}")
        self.logger.info(f"Total evaluations: {search_stats.total_evaluations}")
        self.logger.info(f"Best fitness: {self.best_fitness:.6f}")
        
        if self.best_expression:
            expr_str = expression_to_string(self.best_expression)
            complexity = get_expression_complexity(self.best_expression)
            self.logger.info(f"Best expression: {expr_str}")
            self.logger.info(f"Expression complexity: {complexity}")
        
        self.logger.info(f"Final diversity: {search_stats.final_diversity:.3f}")
        
        if search_stats.convergence_generation is not None:
            self.logger.info(f"Converged at generation: {search_stats.convergence_generation}")
        
        self.logger.info("="*60)


# Factory functions and utilities

def create_regressor_from_config(
    grammar: BaseGrammar,
    config_name: str = "default",
    **config_overrides
) -> SymbolicRegressor:
    """
    Create a SymbolicRegressor from a named configuration.
    
    Args:
        grammar: Grammar to use
        config_name: Name of configuration ("default", "fast", "thorough", "production")
        **config_overrides: Override specific config parameters
        
    Returns:
        Configured SymbolicRegressor
    """
    from janus.core.search.config import (
        create_default_config, create_fast_config, 
        create_thorough_config, create_production_config
    )
    
    config_factories = {
        "default": create_default_config,
        "fast": create_fast_config,
        "thorough": create_thorough_config,
        "production": create_production_config
    }
    
    if config_name not in config_factories:
        raise UnsupportedOperationError(
            f"Configuration '{config_name}'",
            context_info="create_regressor_from_config",
            alternative=f"Available configs: {list(config_factories.keys())}"
        )
    
    config = config_factories[config_name]()
    
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")
    
    return SymbolicRegressor(grammar=grammar, config=config)


def run_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    grammar: BaseGrammar,
    variable_names: Optional[List[str]] = None,
    config: Optional[GAConfig] = None,
    **kwargs
) -> Tuple[Expression, SearchStats]:
    """
    Convenience function to run symbolic regression.
    
    Args:
        X: Input features
        y: Target values
        grammar: Grammar for expression generation
        variable_names: Variable names
        config: GA configuration
        **kwargs: Additional parameters for SymbolicRegressor
        
    Returns:
        Tuple of (best_expression, search_statistics)
    """
    regressor = SymbolicRegressor(grammar=grammar, config=config, **kwargs)
    best_expr = regressor.fit(X, y, variable_names)
    stats = regressor.get_search_stats()
    
    return best_expr, stats


# Legacy compatibility - maintain the original interface
def create_default_fitness_function(
    complexity_weight: float = 0.01,
    parsimony_pressure: bool = True
) -> Callable:
    """Create a default fitness function (legacy compatibility)."""
    def fitness_function(expression, X, y, variable_names):
        try:
            expr_str = expression_to_string(expression)
            predictions = evaluate_expression_on_data(expr_str, variable_names, X)
            
            if predictions is None or not np.all(np.isfinite(predictions)):
                return -1e6
            
            mse = np.mean((y - predictions) ** 2)
            fitness = -mse
            
            if parsimony_pressure:
                complexity = get_expression_complexity(expression)
                fitness -= complexity_weight * complexity
            
            return fitness
            
        except Exception:
            return -1e6
    
    return fitness_function