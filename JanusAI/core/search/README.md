# Janus Core Search Module

A high-performance, modular genetic algorithm implementation for symbolic regression in the Janus framework.

## Overview

The `janus.core.search` module provides a complete genetic algorithm implementation specifically designed for symbolic regression tasks. It has been refactored from a monolithic implementation into focused, extensible components that deliver both high performance and maintainability.

### Key Features

- 🚀 **High Performance**: Lambdify caching provides 10-100x speedup over naive evaluation
- ⚡ **Parallel Processing**: Support for threading, multiprocessing, and joblib backends
- 🧩 **Modular Architecture**: Clean separation of concerns with registry-based extensibility
- 📊 **Rich Analytics**: Comprehensive statistics and diversity metrics
- 🎛️ **Flexible Configuration**: Extensive configuration options with validation
- 🔧 **Grammar-Driven**: Operators automatically match grammar capabilities
- 🧪 **Well-Tested**: Comprehensive unit test coverage

## Quick Start

```python
from janus.core.search import SymbolicRegressor, create_fast_config
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])  # y = 2*x

# Configure and run symbolic regression
config = create_fast_config()
regressor = SymbolicRegressor(grammar=my_grammar, config=config)
best_expression = regressor.fit(X, y, variable_names=['x'])

print(f"Discovered expression: {best_expression}")
```

## Architecture

### Core Components

1. **SymbolicRegressor**: Main genetic algorithm implementation
2. **GAConfig**: Unified configuration management
3. **SelectionStrategy**: Multiple selection strategies with registry pattern
4. **GeneticOperators**: Crossover and mutation operators
5. **StatsTracker**: Comprehensive statistics and monitoring
6. **FitnessCache**: Performance optimization through caching

### Module Structure

```
janus/core/search/
├── __init__.py          # Public API
├── genetic.py           # Main SymbolicRegressor class
├── config.py            # Configuration classes
├── selection.py         # Selection strategies
├── operators.py         # Genetic operators
├── stats.py             # Statistics tracking
└── README.md           # This file
```

## Configuration

### Basic Configuration

```python
from janus.core.search import GAConfig, ExpressionConfig

# Create basic configuration
config = GAConfig(
    population_size=100,
    generations=50,
    mutation_rate=0.1,
    crossover_rate=0.9,
    enable_caching=True,
    enable_parallel=True
)
```

### Expression Configuration

```python
# Detailed expression generation control
expression_config = ExpressionConfig(
    terminal_probability=0.3,
    constant_range=(-5.0, 5.0),
    variable_probability=0.7,
    enable_simplification=True
)

config = GAConfig(
    population_size=50,
    expression_config=expression_config
)
```

### Pre-configured Options

```python
from janus.core.search import (
    create_fast_config,      # Speed-optimized
    create_thorough_config,  # Quality-optimized
    create_production_config # Balanced for production
)

# Use pre-configured setups
fast_config = create_fast_config()
thorough_config = create_thorough_config()
```

## Selection Strategies

The module supports multiple selection strategies through a registry pattern:

```python
from janus.core.search import create_selection_strategy

# Tournament selection (default)
tournament = create_selection_strategy("tournament", tournament_size=3)

# Roulette wheel selection
roulette = create_selection_strategy("roulette")

# Rank-based selection
rank = create_selection_strategy("rank", selection_pressure=1.5)

# List available strategies
from janus.core.search import list_selection_strategies
print(list_selection_strategies())  # ['tournament', 'roulette', 'rank', 'elitist', 'sus']
```

## Genetic Operators

### Crossover Operators

```python
from janus.core.search import create_crossover_operator

# Subtree crossover (default)
subtree_crossover = create_crossover_operator("subtree")

# Uniform crossover
uniform_crossover = create_crossover_operator("uniform", swap_probability=0.5)
```

### Mutation Operators

```python
from janus.core.search import create_mutation_operator

# Multiple mutation types available
node_mutation = create_mutation_operator("node_replacement", generator)
subtree_mutation = create_mutation_operator("subtree_replacement", generator)
constant_mutation = create_mutation_operator("constant_perturbation", generator)
operator_mutation = create_mutation_operator("operator_mutation", generator)
```

## Performance Optimization

### Caching

The module includes intelligent caching for both fitness values and lambdified functions:

```python
# Caching is enabled by default
config = GAConfig(enable_caching=True)

# Access cache statistics
regressor = SymbolicRegressor(grammar, config)
cache_stats = regressor.fitness_cache.get_cache_stats()
print(f"Cache hit ratio: {cache_stats['fitness_cache_ratio']:.2%}")
```

### Parallel Processing

```python
# Configure parallel processing
config = GAConfig(
    enable_parallel=True,
    parallel_backend="threading",  # or "multiprocessing", "joblib"
    n_jobs=-1  # Use all available cores
)
```

## Statistics and Monitoring

### Comprehensive Statistics

```python
# Access detailed statistics after fitting
regressor = SymbolicRegressor(grammar, config)
best_expr = regressor.fit(X, y)

# Get complete search statistics
stats = regressor.get_search_stats()
print(f"Total generations: {stats.total_generations}")
print(f"Best fitness: {stats.best_fitness}")
print(f"Convergence generation: {stats.convergence_generation}")

# Get performance summary
performance = regressor.get_performance_summary()
print(f"Evaluations per second: {performance['eval_rate']:.1f}")
```

### Generation-by-Generation Analysis

```python
# Analyze each generation
for gen_stats in stats.generation_history:
    print(f"Gen {gen_stats.generation}: "
          f"Best={gen_stats.best_fitness:.4f}, "
          f"Diversity={gen_stats.diversity_structural:.3f}")
```

### Enhanced Diversity Metrics

The module includes multiple diversity measures:

- **Structural Diversity**: Unique expression strings
- **Fingerprint Diversity**: Structural fingerprints ignoring constants  
- **Tree Distance**: Average pairwise tree edit distance
- **Fitness Diversity**: Coefficient of variation in fitness

```python
# Access diversity metrics
gen_stats = stats.generation_history[-1]  # Last generation
print(f"Structural diversity: {gen_stats.diversity_structural:.3f}")
print(f"Fingerprint diversity: {gen_stats.diversity_fingerprint:.3f}")
print(f"Average tree distance: {gen_stats.diversity_tree_distance:.1f}")
```

## Advanced Usage

### Custom Fitness Functions

```python
def custom_fitness(expression, X, y, variable_names):
    """Custom fitness function with domain-specific constraints."""
    from janus.core.expressions.symbolic_math import evaluate_expression_on_data
    
    # Standard evaluation
    predictions = evaluate_expression_on_data(
        str(expression.symbolic), variable_names, X
    )
    
    if predictions is None:
        return -1e6
    
    # Custom loss function
    mse = np.mean((y - predictions) ** 2)
    
    # Add custom constraints
    if np.any(predictions < 0):  # Penalize negative predictions
        return -1e6
    
    return -mse

# Use custom fitness function
regressor = SymbolicRegressor(grammar, config, fitness_fn=custom_fitness)
```

### Custom Selection Strategies

```python
from janus.core.search.selection import SelectionStrategy, register_selection_strategy

class CustomSelection(SelectionStrategy):
    @property
    def name(self):
        return "custom"
    
    def select(self, population, fitnesses, num_parents=2, **kwargs):
        # Custom selection logic
        indices = np.random.choice(len(population), size=num_parents)
        return [population[i] for i in indices]

# Register and use custom strategy
register_selection_strategy(CustomSelection())
config = GAConfig(selection_strategy="custom")
```

### Integration with Grammar

The module automatically adapts to grammar capabilities:

```python
class MyGrammar(BaseGrammar):
    def get_operators(self):
        return ['+', '-', '*', '/', 'sin', 'cos', 'exp']
    
    def get_arity(self, operator):
        if operator in ['+', '-', '*', '/']:
            return 2
        else:
            return 1

# Grammar capabilities automatically detected
grammar = MyGrammar()
regressor = SymbolicRegressor(grammar, config)
```

## Best Practices

### Configuration Guidelines

1. **Start Small**: Use `create_fast_config()` for initial experiments
2. **Scale Up**: Use `create_thorough_config()` for production runs
3. **Enable Caching**: Always use `enable_caching=True` for repeated evaluations
4. **Parallel Processing**: Enable for populations > 50 and sufficient cores

### Performance Tips

1. **Cache Configuration**: Larger cache sizes for repeated expressions
2. **Population Size**: 50-200 typically optimal, depends on problem complexity
3. **Early Stopping**: Set appropriate thresholds to avoid overrunning
4. **Complexity Limits**: Use `max_complexity` to prevent expression bloat

### Debugging

```python
# Enable verbose logging
config = GAConfig(verbose=True)

# Access detailed logs
import logging
logging.getLogger('janus.core.search').setLevel(logging.DEBUG)

# Monitor cache performance
cache_stats = regressor.fitness_cache.get_cache_stats()
print(f"Cache efficiency: {cache_stats}")
```

## Examples

### Basic Symbolic Regression

```python
import numpy as np
from janus.core.search import SymbolicRegressor, create_default_config

# Generate synthetic data: y = x^2 + 2*x + 1
X = np.linspace(-2, 2, 50).reshape(-1, 1)
y = X.flatten()**2 + 2*X.flatten() + 1

# Configure and run
config = create_default_config()
config.verbose = True

regressor = SymbolicRegressor(grammar=my_grammar, config=config)
best_expr = regressor.fit(X, y, variable_names=['x'])

print(f"Discovered: {best_expr}")
```

### Multi-Variable Regression

```python
# Multi-variable data: z = x*y + sin(x)
X = np.random.uniform(-2, 2, (100, 2))
y = X[:, 0] * X[:, 1] + np.sin(X[:, 0])

regressor = SymbolicRegressor(grammar=my_grammar)
best_expr = regressor.fit(X, y, variable_names=['x', 'y'])
```

### Performance Benchmarking

```python
from janus.core.search import benchmark_selection_strategies
import time

# Benchmark different configurations
configs = {
    'fast': create_fast_config(),
    'thorough': create_thorough_config()
}

results = {}
for name, config in configs.items():
    start_time = time.time()
    regressor = SymbolicRegressor(grammar, config)
    best_expr = regressor.fit(X, y)
    duration = time.time() - start_time
    
    results[name] = {
        'duration': duration,
        'fitness': regressor.best_fitness,
        'generations': len(regressor.get_search_stats().generation_history)
    }

print("Benchmark Results:")
for name, result in results.items():
    print(f"{name}: {result['duration']:.1f}s, fitness: {result['fitness']:.4f}")
```

## API Reference

### Main Classes

- **SymbolicRegressor**: Main genetic algorithm implementation
- **GAConfig**: Configuration for genetic algorithm parameters
- **ExpressionConfig**: Configuration for expression generation
- **StatsTracker**: Statistics collection and analysis
- **FitnessCache**: Performance optimization through caching

### Factory Functions

- **create_fast_config()**: Speed-optimized configuration
- **create_thorough_config()**: Quality-optimized configuration  
- **create_production_config()**: Balanced production configuration
- **create_selection_strategy()**: Create selection strategies
- **create_crossover_operator()**: Create crossover operators
- **create_mutation_operator()**: Create mutation operators

### Utility Functions

- **run_symbolic_regression()**: Convenience function for quick runs
- **benchmark_selection_strategies()**: Performance analysis
- **list_selection_strategies()**: Available selection methods
- **list_operators()**: Available genetic operators

## Migration Guide

### From Legacy Implementation

If upgrading from the previous monolithic implementation:

```python
# Old way
regressor = SymbolicRegressor(
    grammar=grammar,
    population_size=100,
    generations=50,
    mutation_rate=0.1,
    # ... many parameters
)

# New way
config = GAConfig(
    population_size=100,
    generations=50,
    mutation_rate=0.1
)
regressor = SymbolicRegressor(grammar=grammar, config=config)
```

### Performance Configuration

Replace performance-related parameters:

```python
# Old PerformanceConfig (deprecated)
from janus.core.search.config import PerformanceConfig  # Shows warning

# New unified config
config = GAConfig(
    enable_caching=True,
    enable_parallel=True,
    parallel_backend="threading"
)
```

## Troubleshooting

### Common Issues

1. **Poor Performance**: Enable caching and check parallel processing settings
2. **Memory Issues**: Reduce cache size or population size
3. **No Convergence**: Increase generations or adjust early stopping parameters
4. **Grammar Errors**: Ensure grammar provides required operator interfaces

### Error Messages

- **"Selection strategy not found"**: Use `list_selection_strategies()` to see available options
- **"Operator not available"**: Check grammar provides required operators
- **"Configuration validation failed"**: Review parameter ranges and constraints

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use minimal config for debugging
debug_config = GAConfig(
    population_size=10,
    generations=5,
    verbose=True,
    enable_parallel=False  # Easier debugging
)
```

## Contributing

To extend the module:

1. **New Selection Strategies**: Inherit from `SelectionStrategy` and register
2. **New Operators**: Inherit from appropriate operator base class
3. **Custom Metrics**: Extend `GenerationStats` or `StatsTracker`
4. **Performance Improvements**: Focus on fitness evaluation and caching

## Performance Characteristics

### Complexity Analysis

- **Time Complexity**: O(G × P × E) where G=generations, P=population size, E=evaluation cost
- **Space Complexity**: O(P × C) where C=cache size
- **Caching Impact**: 10-100x speedup for repeated evaluations
- **Parallel Scaling**: Near-linear speedup up to core count

### Benchmark Results

Typical performance on modern hardware:
- **Small problems** (pop=50, gen=25): 1-10 seconds
- **Medium problems** (pop=100, gen=50): 10-60 seconds  
- **Large problems** (pop=200, gen=100): 1-10 minutes

Results vary significantly based on:
- Grammar complexity
- Expression evaluation cost
- Data size
- Hardware specifications

---

For more examples and detailed API documentation, see the test files in `tests/unit/test_genetic.py`.