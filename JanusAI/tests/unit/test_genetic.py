"""
Unit tests for genetic algorithm components.

These tests cover the core functionality of the modular genetic algorithm
implementation including selection strategies, genetic operators, caching,
and statistics tracking.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Import components to test
from core.search.config import (
    GAConfig, ExpressionConfig, 
    create_default_config, create_fast_config
)
from core.search.selection import (
    TournamentSelection, RouletteWheelSelection, RankSelection,
    create_selection_strategy
)
from core.search.operators import (
    ExpressionGenerator, SubtreeCrossover, NodeReplacementMutation,
    create_crossover_operator, create_mutation_operator
)
from core.search.stats import (
    StatsTracker, calculate_structural_fingerprint, 
    calculate_tree_edit_distance, calculate_average_pairwise_distance
)
from physics.algorithms.genetic import (
    SymbolicRegressor, FitnessCache, _evaluate_expression_fitness_worker
)
from core.expressions.expression import Expression, Variable
from core.grammar.progressive_grammar import ProgressiveGrammar as BaseGrammar # Updated import


class MockGrammar(BaseGrammar): # Now correctly inherits from aliased ProgressiveGrammar
    """Mock grammar for testing."""
    
    def __init__(self):
        super().__init__()
        self.operators = ['+', '-', '*', '/', 'sin', 'cos']
    
    def is_operator_known(self, operator: str) -> bool:
        return operator in self.operators
    
    def get_arity(self, operator: str) -> int:
        if operator in ['+', '-', '*', '/', '**']:
            return 2
        elif operator in ['sin', 'cos', 'exp', 'log', 'sqrt', 'neg']:
            return 1
        else:
            return 2  # Default


def create_test_population() -> list:
    """Create a test population of expressions."""
    var_x = Variable("x", 0, {})
    var_y = Variable("y", 1, {})
    
    expressions = [
        Expression("var", [var_x]),  # x
        Expression("var", [var_y]),  # y
        Expression("const", [1.0]),  # 1.0
        Expression("+", [Expression("var", [var_x]), Expression("const", [2.0])]),  # x + 2
        Expression("*", [Expression("var", [var_x]), Expression("var", [var_y])]),  # x * y
    ]
    
    return expressions


class TestGAConfig:
    """Test GA configuration validation and creation."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = GAConfig()
        assert config.population_size == 100
        assert config.generations == 50
        assert config.enable_caching is True
        assert config.expression_config is not None
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should pass
        config = GAConfig(population_size=50, generations=25)
        assert config.population_size == 50
        
        # Invalid population size should raise error
        with pytest.raises(ValueError, match="population_size must be positive"):
            GAConfig(population_size=0)
        
        # Invalid selection strategy should raise error
        with pytest.raises(ValueError, match="selection_strategy must be one of"):
            GAConfig(selection_strategy="invalid")
    
    def test_config_factories(self):
        """Test configuration factory functions."""
        fast_config = create_fast_config()
        assert fast_config.population_size == 50
        assert fast_config.generations == 25
        
        default_config = create_default_config()
        assert default_config.population_size == 100


class TestSelectionStrategies:
    """Test selection strategy implementations."""
    
    @pytest.fixture
    def test_population(self):
        """Fixture providing test population."""
        return create_test_population()
    
    @pytest.fixture
    def test_fitnesses(self):
        """Fixture providing test fitness values."""
        return [0.1, 0.5, 0.9, 0.3, 0.7]
    
    def test_tournament_selection(self, test_population, test_fitnesses):
        """Test tournament selection strategy."""
        strategy = TournamentSelection(tournament_size=3)
        
        # Test normal selection
        parents = strategy.select(test_population, test_fitnesses, num_parents=2)
        assert len(parents) == 2
        assert all(isinstance(p, Expression) for p in parents)
    
    def test_tournament_selection_edge_cases(self, test_population):
        """Test tournament selection edge cases."""
        strategy = TournamentSelection(tournament_size=3)
        
        # All fitnesses invalid
        invalid_fitnesses = [float('inf')] * len(test_population)
        parents = strategy.select(test_population, invalid_fitnesses, num_parents=2)
        assert len(parents) == 2
        
        # All fitnesses equal
        equal_fitnesses = [0.5] * len(test_population)
        parents = strategy.select(test_population, equal_fitnesses, num_parents=2)
        assert len(parents) == 2
    
    def test_roulette_wheel_selection(self, test_population, test_fitnesses):
        """Test roulette wheel selection."""
        strategy = RouletteWheelSelection()
        
        parents = strategy.select(test_population, test_fitnesses, num_parents=2)
        assert len(parents) == 2
        assert all(isinstance(p, Expression) for p in parents)
    
    def test_roulette_wheel_edge_cases(self, test_population):
        """Test roulette wheel selection edge cases."""
        strategy = RouletteWheelSelection()
        
        # All negative fitnesses
        negative_fitnesses = [-1.0, -0.5, -2.0, -0.1, -1.5]
        parents = strategy.select(test_population, negative_fitnesses, num_parents=2)
        assert len(parents) == 2
        
        # All zero fitnesses
        zero_fitnesses = [0.0] * len(test_population)
        parents = strategy.select(test_population, zero_fitnesses, num_parents=2)
        assert len(parents) == 2
    
    def test_rank_selection(self, test_population, test_fitnesses):
        """Test rank-based selection."""
        strategy = RankSelection()
        
        parents = strategy.select(test_population, test_fitnesses, num_parents=2)
        assert len(parents) == 2
        assert all(isinstance(p, Expression) for p in parents)
    
    def test_selection_strategy_factory(self):
        """Test selection strategy factory function."""
        strategy = create_selection_strategy("tournament", tournament_size=5)
        assert isinstance(strategy, TournamentSelection)
        assert strategy.tournament_size == 5
        
        with pytest.raises(Exception):  # Should raise UnsupportedOperationError
            create_selection_strategy("invalid_strategy")


class TestGeneticOperators:
    """Test genetic operators (crossover and mutation)."""
    
    @pytest.fixture
    def mock_grammar(self):
        """Fixture providing mock grammar."""
        return MockGrammar()
    
    @pytest.fixture
    def expression_generator(self, mock_grammar):
        """Fixture providing expression generator."""
        return ExpressionGenerator(mock_grammar)
    
    @pytest.fixture
    def test_variables(self):
        """Fixture providing test variables."""
        return [Variable("x", 0, {}), Variable("y", 1, {})]
    
    def test_expression_generation(self, expression_generator, test_variables):
        """Test random expression generation."""
        expr = expression_generator.generate_random_expression(test_variables)
        assert expr is not None
        assert isinstance(expr, Expression)
    
    def test_population_generation(self, expression_generator, test_variables):
        """Test population generation."""
        population = expression_generator.generate_population(
            test_variables, population_size=10
        )
        assert len(population) == 10
        assert all(isinstance(expr, Expression) for expr in population)
    
    def test_subtree_crossover(self):
        """Test subtree crossover operation."""
        crossover = SubtreeCrossover()
        
        var_x = Variable("x", 0, {})
        parent1 = Expression("+", [Expression("var", [var_x]), Expression("const", [1.0])])
        parent2 = Expression("*", [Expression("var", [var_x]), Expression("const", [2.0])])
        
        child1, child2 = crossover.crossover(parent1, parent2)
        
        # Children should be valid expressions
        assert isinstance(child1, Expression)
        assert isinstance(child2, Expression)
        
        # Parents should be unchanged (deep copy)
        assert parent1.operator == "+"
        assert parent2.operator == "*"
    
    def test_node_replacement_mutation(self, expression_generator, test_variables):
        """Test node replacement mutation."""
        mutation = NodeReplacementMutation(expression_generator)
        
        var_x = Variable("x", 0, {})
        original = Expression("+", [Expression("var", [var_x]), Expression("const", [1.0])])
        
        mutated = mutation.mutate(original, test_variables)
        
        # Should return a valid expression
        assert isinstance(mutated, Expression)
        
        # Original should be unchanged
        assert original.operator == "+"
    
    def test_operator_creation_factories(self, expression_generator):
        """Test operator creation factories."""
        crossover = create_crossover_operator("subtree")
        assert isinstance(crossover, SubtreeCrossover)
        
        mutation = create_mutation_operator("node_replacement", expression_generator)
        assert isinstance(mutation, NodeReplacementMutation)


class TestFitnessCache:
    """Test fitness caching functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = FitnessCache(max_size=3)
        
        # Test caching and retrieval
        cache.cache_fitness("expr1", 0.5)
        assert cache.get_fitness("expr1") == 0.5
        assert cache.get_fitness("nonexistent") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = FitnessCache(max_size=2)
        
        # Fill cache
        cache.cache_fitness("expr1", 0.1)
        cache.cache_fitness("expr2", 0.2)
        
        # Access expr1 (makes it most recent)
        _ = cache.get_fitness("expr1")
        
        # Add third item (should evict expr2)
        cache.cache_fitness("expr3", 0.3)
        
        # expr1 should still be there, expr2 should be evicted
        assert cache.get_fitness("expr1") == 0.1
        assert cache.get_fitness("expr2") is None
        assert cache.get_fitness("expr3") == 0.3
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = FitnessCache(max_size=10)
        cache.cache_fitness("expr1", 0.1)
        cache.cache_fitness("expr2", 0.2)
        
        stats = cache.get_cache_stats()
        assert stats['fitness_cache_size'] == 2
        assert stats['max_size'] == 10
        assert 0 <= stats['fitness_cache_ratio'] <= 1


class TestStatsTracking:
    """Test statistics tracking functionality."""
    
    @pytest.fixture
    def stats_tracker(self):
        """Fixture providing stats tracker."""
        return StatsTracker()
    
    @pytest.fixture
    def test_population(self):
        """Fixture providing test population."""
        return create_test_population()
    
    def test_generation_recording(self, stats_tracker, test_population):
        """Test generation statistics recording."""
        fitnesses = [0.1, 0.5, 0.9, 0.3, 0.7]
        
        stats_tracker.start_generation(0)
        gen_stats = stats_tracker.record_generation(
            0, test_population, fitnesses, evaluation_time=1.0
        )
        
        assert gen_stats.generation == 0
        assert gen_stats.best_fitness == 0.9
        assert gen_stats.worst_fitness == 0.1
        assert gen_stats.mean_fitness == 0.5
        assert gen_stats.evaluations_count == 5
    
    def test_diversity_metrics(self, test_population):
        """Test diversity metric calculations."""
        # Test structural fingerprinting
        expr = test_population[0]
        fingerprint = calculate_structural_fingerprint(expr)
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0
        
        # Test tree edit distance
        distance = calculate_tree_edit_distance(test_population[0], test_population[1])
        assert isinstance(distance, int)
        assert distance >= 0
        
        # Test average pairwise distance
        avg_distance = calculate_average_pairwise_distance(test_population)
        assert isinstance(avg_distance, float)
        assert avg_distance >= 0


class TestSymbolicRegressor:
    """Test the main SymbolicRegressor class."""
    
    @pytest.fixture
    def mock_grammar(self):
        """Fixture providing mock grammar."""
        return MockGrammar()
    
    @pytest.fixture
    def simple_data(self):
        """Fixture providing simple test data."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([3, 5, 7, 9])  # y = x + 2
        variable_names = ["x1", "x2"]
        return X, y, variable_names
    
    def test_regressor_initialization(self, mock_grammar):
        """Test regressor initialization."""
        config = GAConfig(population_size=10, generations=5)
        regressor = SymbolicRegressor(grammar=mock_grammar, config=config)
        
        assert regressor.grammar == mock_grammar
        assert regressor.config.population_size == 10
        assert regressor.best_expression is None
        assert regressor.fitness_cache is not None  # Should be enabled by default
    
    def test_input_validation(self, mock_grammar, simple_data):
        """Test input validation."""
        X, y, variable_names = simple_data
        regressor = SymbolicRegressor(grammar=mock_grammar)
        
        # Valid inputs should pass
        regressor._validate_inputs(X, y, variable_names)
        
        # Invalid inputs should raise errors
        with pytest.raises(Exception):  # DataValidationError
            regressor._validate_inputs(X[:-1], y, variable_names)  # Mismatched sizes
    
    def test_worker_function(self):
        """Test parallel worker function."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        variable_names = ["x"]
        
        # Test simple expression
        fitness = _evaluate_expression_fitness_worker("2*x", X, y, variable_names)
        assert isinstance(fitness, float)
        assert fitness > -1e6  # Should not be penalty value for this simple case
        
        # Test invalid expression
        fitness = _evaluate_expression_fitness_worker("invalid_expr", X, y, variable_names)
        assert fitness == -1e6  # Should be penalty value
    
    @patch('janus.physics.algorithms.genetic.HAS_TQDM', False)
    def test_fit_basic(self, mock_grammar, simple_data):
        """Test basic fitting functionality."""
        X, y, variable_names = simple_data
        
        # Use minimal config for fast test
        config = GAConfig(
            population_size=5, 
            generations=2, 
            verbose=False,
            enable_parallel=False  # Disable for simpler testing
        )
        
        regressor = SymbolicRegressor(grammar=mock_grammar, config=config)
        
        # Should complete without errors
        best_expr = regressor.fit(X, y, variable_names)
        
        assert best_expr is not None
        assert isinstance(best_expr, Expression)
        assert regressor.best_fitness > -np.inf
    
    def test_prediction(self, mock_grammar, simple_data):
        """Test prediction functionality."""
        X, y, variable_names = simple_data
        
        # Create a simple regressor and fit
        config = GAConfig(population_size=5, generations=2, verbose=False)
        regressor = SymbolicRegressor(grammar=mock_grammar, config=config)
        
        # Manually set a simple best expression for testing
        var_x = Variable("x1", 0, {})
        regressor.best_expression = Expression("var", [var_x])
        
        # Test prediction
        predictions = regressor.predict(X, variable_names)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test data
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])  # y = 2*x
        variable_names = ["x"]
        
        # Create grammar and config
        grammar = MockGrammar()
        config = GAConfig(
            population_size=10,
            generations=3,
            verbose=False,
            enable_parallel=False
        )
        
        # Run symbolic regression
        regressor = SymbolicRegressor(grammar=grammar, config=config)
        best_expr = regressor.fit(X, y, variable_names)
        
        # Check results
        assert best_expr is not None
        
        # Test prediction
        predictions = regressor.predict(X, variable_names)
        assert len(predictions) == len(y)
        
        # Test statistics
        stats = regressor.get_search_stats()
        assert stats.total_generations > 0
        assert len(stats.generation_history) > 0


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])