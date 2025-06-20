"""
Comprehensive tests for BaseGrammar and ProgressiveGrammar functionality.
Tests cover variable discovery, expression creation, MDL-based compression,
and grammar evolution.
"""
import pytest
import numpy as np
import sympy as sp
from unittest.mock import MagicMock, patch, PropertyMock
from sklearn.decomposition import FastICA

from JanusAI.core.grammar.base_grammar import BaseGrammar, ProgressiveGrammar, NoisyObservationProcessor
from JanusAI.core.expressions.expression import Expression, Variable


class TestBaseGrammar:
    """Test the BaseGrammar abstract class functionality."""
    
    @pytest.fixture
    def mock_grammar(self):
        """Create a mock concrete implementation of BaseGrammar."""
        class MockGrammar(BaseGrammar):
            def discover_variables(self, observations, time_stamps=None):
                return []
            
            def _analyze_component(self, component, time_stamps=None):
                return {}
            
            def _generate_variable_name(self, properties):
                return "mock_var"
        
        return MockGrammar()
    
    def test_expression_key_commutative(self, mock_grammar):
        """Test that expression keys handle commutativity correctly."""
        x = Variable("x", 0, {})
        y = Variable("y", 1, {})
        
        # Create x + y and y + x
        expr1 = mock_grammar.create_expression('+', [x, y])
        expr2 = mock_grammar.create_expression('+', [y, x])
        
        # Keys should be the same for commutative operators
        key1 = mock_grammar._expression_key(expr1)
        key2 = mock_grammar._expression_key(expr2)
        assert key1 == key2
        
        # Test with multiplication (also commutative)
        expr3 = mock_grammar.create_expression('*', [x, y])
        expr4 = mock_grammar.create_expression('*', [y, x])
        key3 = mock_grammar._expression_key(expr3)
        key4 = mock_grammar._expression_key(expr4)
        assert key3 == key4
        
        # Test with non-commutative operator (subtraction)
        expr5 = mock_grammar.create_expression('-', [x, y])
        expr6 = mock_grammar.create_expression('-', [y, x])
        key5 = mock_grammar._expression_key(expr5)
        key6 = mock_grammar._expression_key(expr6)
        assert key5 != key6
    
    def test_create_expression_validation(self, mock_grammar):
        """Test expression creation with arity validation."""
        x = Variable("x", 0, {})
        y = Variable("y", 1, {})
        
        # Valid arities
        assert mock_grammar.create_expression('+', [x, y]) is not None
        assert mock_grammar.create_expression('sin', [x]) is not None
        assert mock_grammar.create_expression('const', [3.14]) is not None
        
        # Invalid arities
        assert mock_grammar.create_expression('+', [x]) is None  # + needs 2 operands
        assert mock_grammar.create_expression('sin', [x, y]) is None  # sin needs 1 operand
        assert mock_grammar.create_expression('unknown_op', [x]) is None  # Unknown operator


class TestProgressiveGrammar:
    """Test the ProgressiveGrammar implementation."""
    
    @pytest.fixture
    def grammar(self):
        """Create a ProgressiveGrammar instance."""
        return ProgressiveGrammar()
    
    @pytest.fixture
    def sample_variables(self):
        """Create sample variables for testing."""
        return {
            'x': Variable("x", 0, {"type": "position"}),
            'y': Variable("y", 1, {"type": "velocity"}),
            'z': Variable("z", 2, {"type": "energy"})
        }
    
    @patch('JanusAI.core.grammar.base_grammar.FastICA')
    @patch.object(NoisyObservationProcessor, 'denoise')
    def test_discover_variables_basic(self, mock_denoise, mock_fastica, grammar):
        """Test basic variable discovery from observations."""
        # Setup mocks
        mock_denoise.side_effect = lambda obs, epochs=50: obs  # Pass through
        
        mock_ica = mock_fastica.return_value
        # Create mock components with different characteristics
        n_samples = 100
        t = np.linspace(0, 10, n_samples)
        components = np.array([
            np.sin(2 * np.pi * 0.5 * t),  # Periodic
            np.ones(n_samples) * 2.5 + 0.01 * np.random.randn(n_samples),  # Conserved
            np.random.randn(n_samples) * 0.1  # Noise
        ]).T
        mock_ica.fit_transform.return_value = components
        
        # Create observations
        observations = np.random.randn(n_samples, 3)
        time_stamps = t
        
        # Discover variables
        variables = grammar.discover_variables(observations, time_stamps)
        
        # Verify calls
        mock_denoise.assert_called_once()
        mock_fastica.assert_called_once_with(n_components=3)
        
        # Should discover at least 2 variables (noise filtered out)
        assert len(variables) >= 2
        assert all(isinstance(var, Variable) for var in variables)
        
        # Check that variables are stored in grammar
        for var in variables:
            assert var.name in grammar.variables
            assert grammar.variables[var.name] == var
    
    def test_analyze_component_properties(self, grammar):
        """Test component analysis for variable properties."""
        # Create synthetic components
        n_samples = 1000
        t = np.linspace(0, 100, n_samples)
        
        # Periodic component
        periodic_component = np.sin(2 * np.pi * 0.1 * t)
        props_periodic = grammar._analyze_component(periodic_component, t)
        assert props_periodic['periodicity_score'] > 5.0  # Should be highly periodic
        assert props_periodic['information_content'] > 0.5
        
        # Conserved component
        conserved_component = np.ones(n_samples) * 5.0 + 0.001 * np.random.randn(n_samples)
        props_conserved = grammar._analyze_component(conserved_component, t)
        assert props_conserved['conservation_score'] > 0.8  # Should be highly conserved
        assert props_conserved['smoothness'] > 0.8
        
        # Noisy component
        noisy_component = np.random.randn(n_samples) * 0.01
        props_noisy = grammar._analyze_component(noisy_component, t)
        assert props_noisy['information_content'] < 0.3  # Low information content
    
    def test_generate_variable_name(self, grammar):
        """Test variable name generation based on properties."""
        # Energy-like variable (high conservation)
        props_energy = {
            'conservation_score': 0.95,
            'periodicity_score': 0.1,
            'smoothness': 0.9,
            'information_content': 0.8
        }
        assert grammar._generate_variable_name(props_energy) == "E_1"
        
        # Periodic variable (high periodicity)
        props_periodic = {
            'conservation_score': 0.2,
            'periodicity_score': 8.5,
            'smoothness': 0.5,
            'information_content': 0.7
        }
        assert grammar._generate_variable_name(props_periodic) == "theta_1"
        
        # Position-like variable (smooth, not conserved)
        props_position = {
            'conservation_score': 0.3,
            'periodicity_score': 0.2,
            'smoothness': 0.85,
            'information_content': 0.8
        }
        assert grammar._generate_variable_name(props_position) == "x_1"
        
        # Generic variable
        props_generic = {
            'conservation_score': 0.4,
            'periodicity_score': 2.0,
            'smoothness': 0.4,
            'information_content': 0.5
        }
        assert grammar._generate_variable_name(props_generic) == "q_1"
    
    def test_count_subexpression(self, grammar, sample_variables):
        """Test counting subexpression occurrences."""
        x, y, z = sample_variables['x'], sample_variables['y'], sample_variables['z']
        
        # Create pattern: x + y
        pattern = grammar.create_expression('+', [x, y])
        
        # Expression containing pattern once
        expr1 = grammar.create_expression('*', [pattern, z])
        assert grammar._count_subexpression(expr1, pattern) == 1
        
        # Expression containing pattern twice
        expr2 = grammar.create_expression('*', [pattern, pattern])
        assert grammar._count_subexpression(expr2, pattern) == 2
        
        # Expression not containing pattern
        expr3 = grammar.create_expression('*', [x, y])
        assert grammar._count_subexpression(expr3, pattern) == 0
        
        # Nested occurrence
        nested = grammar.create_expression('+', [
            grammar.create_expression('*', [pattern, z]),
            pattern
        ])
        assert grammar._count_subexpression(nested, pattern) == 2
    
    def test_calculate_compression_gain(self, grammar, sample_variables):
        """Test MDL compression gain calculation."""
        x, y, z = sample_variables['x'], sample_variables['y'], sample_variables['z']
        
        # Create candidate pattern: x + y
        candidate = grammar.create_expression('+', [x, y])
        assert candidate.complexity == 3
        
        # Create corpus with multiple uses of pattern
        expr1 = grammar.create_expression('*', [candidate, z])  # Uses pattern once
        expr2 = grammar.create_expression('+', [candidate, grammar.create_expression('const', [1.0])])  # Uses pattern once
        expr3 = grammar.create_expression('*', [x, y])  # Doesn't use pattern
        
        corpus = [expr1, expr2, expr3]
        
        # Calculate expected gain
        # Current length: sum of all complexities
        current_length = sum(e.complexity for e in corpus)
        
        # New length: cost of defining candidate + reduced corpus
        # expr1: 5 - 1*(3-1) = 3
        # expr2: 6 - 1*(3-1) = 4
        # expr3: 3 (unchanged)
        # Total: 3 (candidate) + 3 + 4 + 3 = 13
        
        gain = grammar._calculate_compression_gain(candidate, corpus)
        assert gain > 0  # Should show positive compression
    
    def test_find_common_patterns(self, grammar, sample_variables):
        """Test finding common patterns in expression corpus."""
        x, y, z = sample_variables['x'], sample_variables['y'], sample_variables['z']
        
        # Create expressions with common subexpressions
        xy_sum = grammar.create_expression('+', [x, y])
        
        corpus = [
            grammar.create_expression('*', [xy_sum, z]),
            grammar.create_expression('+', [xy_sum, grammar.create_expression('const', [1.0])]),
            grammar.create_expression('sin', [xy_sum]),
            grammar.create_expression('*', [x, y])  # Different pattern
        ]
        
        # Find patterns
        patterns = grammar._find_common_patterns(corpus, min_occurrences=2)
        
        # Should find x+y pattern
        pattern_keys = [grammar._expression_key(p) for p in patterns]
        xy_key = grammar._expression_key(xy_sum)
        assert xy_key in pattern_keys
    
    def test_create_abstraction(self, grammar, sample_variables):
        """Test creating abstractions from patterns."""
        x, y = sample_variables['x'], sample_variables['y']
        
        # Create a pattern
        pattern = grammar.create_expression('+', [x, y])
        
        # Create abstraction
        abstraction = grammar._create_abstraction(pattern, "sum_xy")
        
        assert isinstance(abstraction, Variable)
        assert abstraction.name == "sum_xy"
        assert abstraction.properties['derived_from'] == pattern
        assert abstraction.complexity == 1  # Abstractions have complexity 1
    
    def test_apply_abstractions(self, grammar, sample_variables):
        """Test applying abstractions to expressions."""
        x, y, z = sample_variables['x'], sample_variables['y'], sample_variables['z']
        
        # Create pattern and abstraction
        pattern = grammar.create_expression('+', [x, y])
        abstraction = grammar._create_abstraction(pattern, "A_1")
        abstractions = {grammar._expression_key(pattern): abstraction}
        
        # Create expression using pattern
        original = grammar.create_expression('*', [pattern, z])
        
        # Apply abstractions
        new_expr = grammar._apply_abstractions(original, abstractions)
        
        # Should replace pattern with abstraction
        assert new_expr.operator == '*'
        assert new_expr.operands[0] == abstraction
        assert new_expr.operands[1] == z
    
    def test_compress_corpus(self, grammar, sample_variables):
        """Test full corpus compression workflow."""
        x, y, z = sample_variables['x'], sample_variables['y'], sample_variables['z']
        
        # Create corpus with repeated patterns
        xy_sum = grammar.create_expression('+', [x, y])
        
        corpus = [
            grammar.create_expression('*', [xy_sum, z]),
            grammar.create_expression('sin', [xy_sum]),
            grammar.create_expression('+', [xy_sum, grammar.create_expression('const', [2.0])]),
            grammar.create_expression('/', [z, xy_sum])
        ]
        
        # Compress corpus
        compressed, abstractions = grammar.compress_corpus(
            corpus,
            min_occurrences=2,
            min_compression_gain=0.1
        )
        
        # Should create abstraction for x+y
        assert len(abstractions) > 0
        
        # Compressed expressions should use abstractions
        assert len(compressed) == len(corpus)
        
        # Total complexity should be reduced
        original_complexity = sum(e.complexity for e in corpus)
        compressed_complexity = sum(e.complexity for e in compressed)
        abstraction_complexity = sum(3 for _ in abstractions.values())  # Pattern complexity
        assert compressed_complexity + abstraction_complexity < original_complexity
    
    def test_evolve_grammar(self, grammar, sample_variables):
        """Test grammar evolution with new expressions."""
        x, y = sample_variables['x'], sample_variables['y']
        
        # Create initial expressions
        expressions = [
            grammar.create_expression('+', [x, y]),
            grammar.create_expression('*', [x, y]),
            grammar.create_expression('sin', [x])
        ]
        
        # Evolve grammar
        grammar.evolve(expressions)
        
        # Check that operators are tracked
        assert '+' in grammar.discovered_operators
        assert '*' in grammar.discovered_operators
        assert 'sin' in grammar.discovered_operators
    
    def test_integration_full_workflow(self, grammar):
        """Test complete workflow from data to compressed expressions."""
        # Generate synthetic data
        n_samples = 100
        t = np.linspace(0, 10, n_samples)
        
        # Pendulum-like data
        theta = np.sin(2 * np.pi * 0.5 * t)
        omega = 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
        energy = 0.5 * omega**2 + 9.8 * (1 - np.cos(theta))
        
        observations = np.column_stack([theta, omega, energy])
        
        with patch.object(NoisyObservationProcessor, 'denoise', side_effect=lambda obs, epochs=50: obs):
            with patch('janus.core.grammar.base_grammar.FastICA') as mock_fastica:
                mock_ica = mock_fastica.return_value
                mock_ica.fit_transform.return_value = observations
                
                # Discover variables
                variables = grammar.discover_variables(observations, t)
                assert len(variables) >= 2
                
                # Create expressions using discovered variables
                if len(variables) >= 2:
                    v1, v2 = variables[0], variables[1]
                    
                    expressions = [
                        grammar.create_expression('+', [v1, v2]),
                        grammar.create_expression('*', [v1, v1]),
                        grammar.create_expression('sin', [v1])
                    ]
                    
                    # Compress if patterns found
                    compressed, abstractions = grammar.compress_corpus(
                        expressions,
                        min_occurrences=1
                    )
                    
                    assert len(compressed) == len(expressions)