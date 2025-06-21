"""
Tests for core/grammar/base_grammar.py: CFGRule, ContextFreeGrammar, NoisyObservationProcessor, ProgressiveGrammar, AIGrammar.
"""
import pytest
import random
from collections import defaultdict
from core.grammar.base_grammar import CFGRule, ContextFreeGrammar
from JanusAI.core.expressions.expression import Variable # For TargetType in CFG

# Tests for CFGRule
class TestCFGRule:
    def test_cfgrule_creation_basic(self):
        """Test basic CFGRule creation."""
        rule = CFGRule(symbol="S", expression=["A", "B"])
        assert rule.symbol == "S"
        assert rule.expression == ["A", "B"]
        assert rule.weight == 1.0

    def test_cfgrule_creation_with_weight(self):
        """Test CFGRule creation with a specific weight."""
        rule = CFGRule(symbol="A", expression="a", weight=2.5)
        assert rule.symbol == "A"
        assert rule.expression == "a"
        assert rule.weight == 2.5

    def test_cfgrule_creation_with_target_type(self):
        """Test CFGRule with TargetType instances in expression."""
        var_x = Variable(name="x", index=0)
        rule = CFGRule[Variable](symbol="VAR", expression=[var_x])
        assert rule.symbol == "VAR"
        assert rule.expression == [var_x]
        assert isinstance(rule.expression[0], Variable)

    def test_cfgrule_post_init_weight_validation(self):
        """Test that __post_init__ validates the weight."""
        with pytest.raises(ValueError, match="Rule weight must be positive."):
            CFGRule(symbol="B", expression="b", weight=0)
        with pytest.raises(ValueError, match="Rule weight must be positive."):
            CFGRule(symbol="C", expression="c", weight=-1.0)

    def test_cfgrule_repr(self): # Assuming dataclass generates a sensible repr
        rule = CFGRule(symbol="S", expression=["A"], weight=0.5)
        assert repr(rule) == "CFGRule(symbol='S', expression=['A'], weight=0.5)"


# Tests for ContextFreeGrammar
class TestContextFreeGrammar:
    def test_cfg_creation_empty(self):
        """Test ContextFreeGrammar creation with no initial rules."""
        cfg = ContextFreeGrammar()
        assert cfg.rules == defaultdict(list)

    def test_cfg_creation_with_rules(self):
        """Test ContextFreeGrammar creation with initial rules."""
        rule1 = CFGRule("S", ["A", "B"])
        rule2 = CFGRule("A", "a")
        cfg = ContextFreeGrammar(rules=[rule1, rule2])
        assert "S" in cfg.rules
        assert "A" in cfg.rules
        assert cfg.rules["S"] == [rule1]
        assert cfg.rules["A"] == [rule2]

    def test_cfg_add_rule(self):
        """Test adding rules using add_rule method."""
        cfg = ContextFreeGrammar()
        rule_s = CFGRule("S", ["A"])
        rule_a = CFGRule("A", "a")

        cfg.add_rule(rule_s)
        assert cfg.rules["S"] == [rule_s]

        cfg.add_rule(rule_a)
        assert cfg.rules["A"] == [rule_a]

        # Add another rule for the same symbol
        rule_s2 = CFGRule("S", ["B"])
        cfg.add_rule(rule_s2)
        assert cfg.rules["S"] == [rule_s, rule_s2]

    def test_cfg_get_productions(self):
        """Test get_productions method."""
        rule_s1 = CFGRule("S", ["A"])
        rule_s2 = CFGRule("S", ["B"])
        cfg = ContextFreeGrammar(rules=[rule_s1, rule_s2])

        productions_s = cfg.get_productions("S")
        assert len(productions_s) == 2
        assert rule_s1 in productions_s
        assert rule_s2 in productions_s

    def test_cfg_get_productions_unknown_symbol(self):
        """Test get_productions for a symbol not in the grammar."""
        cfg = ContextFreeGrammar()
        with pytest.raises(ValueError, match="Symbol 'X' not found in grammar rules."):
            cfg.get_productions("X")

    def test_cfg_generate_random_simple(self):
        """Test generate_random for a simple grammar with one derivation."""
        # S -> A
        # A -> "a"
        rules = [CFGRule("S", ["A"]), CFGRule("A", ["a_terminal"])] # Ensure terminal is distinguishable
        cfg = ContextFreeGrammar(rules)
        # Mock random.uniform to always pick the first rule if there's a choice
        random.seed(0) # For reproducibility if multiple calls are made

        generated = cfg.generate_random("S")
        assert generated == ["a_terminal"]

    def test_cfg_generate_random_choice_and_weights(self, monkeypatch):
        """Test generate_random with choices and weights."""
        # S -> "a" (weight 1.0)
        # S -> "b" (weight 9.0)
        rule_sa = CFGRule("S", ["a_term"], weight=1.0)
        rule_sb = CFGRule("S", ["b_term"], weight=9.0)
        cfg = ContextFreeGrammar(rules=[rule_sa, rule_sb])

        # Mock random.uniform to control rule selection
        # Total weight is 10.0.
        # To select rule_sa (weight 1.0), random.uniform must return value in (0, 1.0]
        # To select rule_sb (weight 9.0), random.uniform must return value in (1.0, 10.0]

        # Test selection of "a_term"
        monkeypatch.setattr(random, 'uniform', lambda a, b: 0.5) # Should pick rule_sa
        generated_a = cfg.generate_random("S")
        assert generated_a == ["a_term"]

        # Test selection of "b_term"
        monkeypatch.setattr(random, 'uniform', lambda a, b: 5.0) # Should pick rule_sb
        generated_b = cfg.generate_random("S")
        assert generated_b == ["b_term"]

    def test_cfg_generate_random_multi_step_derivation(self):
        """Test generate_random with a multi-step derivation."""
        # EXPR -> OP EXPR EXPR
        # EXPR -> VAR
        # OP   -> "+"
        # OP   -> "*"
        # VAR  -> "x"
        var_x_obj = Variable("x",0)
        rules = [
            CFGRule("EXPR", ["OP", "EXPR", "EXPR"], weight=0.5),
            CFGRule("EXPR", ["VAR"], weight=0.5),
            CFGRule("OP", ["+"], weight=0.5),
            CFGRule("OP", ["*"], weight=0.5),
            CFGRule[Variable]("VAR", [var_x_obj], weight=1.0) # Using Variable object
        ]
        cfg = ContextFreeGrammar[Variable](rules) # Specify TargetType
        random.seed(42) # For some predictability

        # This can generate things like: ["+", "x", "x"], ["*", "x", ["+", "x", "x"]], etc.
        # We check that output is a list of strings or Variable objects.
        # Max depth is 100, so for simple non-recursive grammars it's fine.
        # For this recursive grammar, it should terminate due to VAR path.
        for _ in range(10): # Try a few generations
            generated = cfg.generate_random("EXPR")
            assert isinstance(generated, list)
            for item in generated:
                assert isinstance(item, (str, Variable))
            # A simple check: if it produced an OP, it should have other elements too
            if "+" in generated or "*" in generated:
                assert len(generated) > 1
            # If only VAR was chosen at top level:
            if generated == [var_x_obj]:
                assert True # This is a valid output

    def test_cfg_generate_random_max_depth(self, caplog):
        """Test generate_random hitting max_depth for a cyclic grammar."""
        # S -> S (cyclic, no terminal path from S directly)
        # S -> A (alternative path to ensure it can terminate if this is chosen)
        # A -> "a"
        rules = [
            CFGRule("S", ["S"], weight=1.0), # Cyclic rule
            CFGRule("S", ["A"], weight=0.01), # Very low weight escape
            CFGRule("A", ["a_term"])
        ]
        cfg = ContextFreeGrammar(rules)

        # Force selection of the cyclic rule by controlling random.uniform
        # Total weight for S is 1.01. To pick S->S, random must be in (0, 1.0]
        original_uniform = random.uniform
        def mock_uniform_cyclic(a,b):
            # If choosing for S (total weight 1.01), pick the S->S rule.
            # If choosing for A (total weight 1.0 for A->"a"), pick that rule.
            if b == 1.01: return 0.5 # Selects S->S
            return original_uniform(a,b)

        random.uniform = mock_uniform_cyclic

        # The CFG's internal max_depth is 100
        generated = cfg.generate_random("S")

        random.uniform = original_uniform # Restore

        assert "Max generation depth reached" in caplog.text
        # The result might be an incomplete list of non-terminals, e.g. ['S'] if max depth is hit early
        # Or it could be ['S', 'S', ... , 'S'] up to some limit based on BFS-like processing
        # The current implementation processes one symbol at a time from expansion_stack (BFS-like)
        # and adds its expansion to the front.
        # So, if S -> S, stack: [S] -> [S]. This will repeat.
        # The check is primarily for the warning.
        # The exact output depends on how max_depth truncates the process.
        # If it stops expanding, 'S' might be left in the result.
        # If it stops adding to stack, result might be empty if 'S' was popped but not replaced.
        # Current code: if stack is non-empty & depth < max_depth.
        # current_symbol = expansion_stack.pop(0). If this S is popped at max_depth,
        # and not processed, result_sequence might be empty or contain prior terminals.
        # Let's trace:
        # stack: [S], result: [] depth:0
        # pop S. chosen_rule S->S. stack: [S], result: [], depth: 1
        # ...
        # pop S. chosen_rule S->S. stack: [S], result: [], depth: 99
        # pop S. chosen_rule S->S. stack: [S], result: [], depth: 100. Loop terminates.
        # Result is []. This seems plausible.
        assert generated == [], "Expected empty list if max depth hit with only non-terminals"


    def test_cfg_generate_random_non_existent_start_symbol(self):
        """Test generate_random with a start symbol that has no rules."""
        cfg = ContextFreeGrammar()
        # If symbol 'X' has no rules, it's treated as a terminal.
        generated = cfg.generate_random("X")
        assert generated == ["X"]

    def test_cfg_generate_random_rule_with_empty_expansion(self):
        """Test generate_random where a rule expands to an empty list (epsilon production)."""
        # S -> [] (epsilon)
        rules = [CFGRule("S", [])]
        cfg = ContextFreeGrammar(rules)
        generated = cfg.generate_random("S")
        assert generated == []

    def test_cfg_generate_random_with_direct_target_type_terminal(self):
        """Test generate_random when a rule directly yields a TargetType object."""
        var_x = Variable(name="x", index=0)
        var_y = Variable(name="y", index=1)
        rules = [
            CFGRule[Variable]("VAR", [var_x], weight=1.0),
            CFGRule[Variable]("VAR", [var_y], weight=1.0)
        ]
        cfg = ContextFreeGrammar[Variable](rules)

        results = set()
        for _ in range(20): # Generate multiple times to see both outcomes
            generated = cfg.generate_random("VAR")
            assert len(generated) == 1
            assert isinstance(generated[0], Variable)
            results.add(generated[0].name)

        assert "x" in results
        assert "y" in results

# Tests for NoisyObservationProcessor
# Mocking torch and related components for NoisyObservationProcessor tests
# to avoid actual model training which is slow and resource-intensive.
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Mock DenoisingAutoencoder if torch is not available or for focused testing
if TORCH_AVAILABLE:
    class MockDenoisingAutoencoder(nn.Module):
        def __init__(self, input_dim_ae, latent_dim_ae):
            super().__init__()
            self.encoder = nn.Linear(input_dim_ae, latent_dim_ae)
            self.decoder = nn.Linear(latent_dim_ae, input_dim_ae)
            self.has_been_trained = False # Mock attribute

        def forward(self, x, noise_level=0.1):
            # Simulate behavior: return something of the correct shape
            # For testing purposes, we might just return the input or a modified version
            if self.has_been_trained: # Simulate effect of training
                 return x, self.encoder(x) # Return "denoised" x and some latent
            return x + torch.randn_like(x) * 0.01, self.encoder(x) # Simulate noisy reconstruction before "training"
else: # Minimal mock if torch is not available at all
    class MockDenoisingAutoencoder:
        def __init__(self, input_dim_ae, latent_dim_ae):
            self.input_dim = input_dim_ae
            self.latent_dim = latent_dim_ae
            self.has_been_trained = False
            # Mock a sub-structure that the main code might check
            self.encoder = [type('MockLinear', (), {'in_features': input_dim_ae})()]

        def parameters(self): return [] # Mock
        def train(self): pass # Mock
        def eval(self): pass # Mock
        def __call__(self, x, noise_level=0.1): # Mock forward pass
            if self.has_been_trained:
                return x, x[:, :self.latent_dim] # Mock latent space as first few cols
            return x * 1.01, x[:, :self.latent_dim] # Slightly alter x to show "not denoised"


from JanusAI.core.grammar.base_grammar import NoisyObservationProcessor
import numpy as np

@pytest.fixture
def mock_torch_optimizer(monkeypatch):
    if TORCH_AVAILABLE:
        class MockOptimizer:
            def __init__(self, params, lr): pass
            def zero_grad(self): pass
            def step(self): pass
        monkeypatch.setattr(torch.optim, 'Adam', MockOptimizer)
        monkeypatch.setattr(torch.nn, 'MSELoss', lambda: lambda x, y: torch.tensor(0.0)) # Mock loss function


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not available, skipping NoisyObservationProcessor DAE tests.")
class TestNoisyObservationProcessor:
    @pytest.fixture
    def processor(self):
        return NoisyObservationProcessor(latent_dim=8)

    def test_nop_initialization(self, processor):
        """Test NoisyObservationProcessor initialization."""
        assert processor.latent_dim == 8
        assert processor.model is None

    def test_nop_build_autoencoder(self, processor, monkeypatch):
        """Test the build_autoencoder method."""
        input_dim = 10

        # Monkeypatch the DenoisingAutoencoder class within the scope of build_autoencoder
        # This is a bit tricky because it's defined inside the method.
        # A common way is to patch it where it's imported or referenced from.
        # Since it's defined *locally*, we'd have to patch 'nn.Module' or similar,
        # or refactor DenoisingAutoencoder out.
        # For this test, we'll rely on the fact that processor.model will be an instance
        # of the actual DenoisingAutoencoder if torch is available.

        # If we want to use MockDenoisingAutoencoder consistently:
        monkeypatch.setattr('JanusAI.core.grammar.base_grammar.DenoisingAutoencoder', MockDenoisingAutoencoder)

        model = processor.build_autoencoder(input_dim)
        assert model is not None
        assert isinstance(model, MockDenoisingAutoencoder) # Check if our mock was used
        # Or, if checking actual model:
        # assert isinstance(model.encoder, torch.nn.Sequential)
        # assert model.encoder[-1].out_features == processor.latent_dim
        # assert model.decoder[-1].out_features == input_dim
        assert model.encoder.in_features == input_dim if hasattr(model.encoder, 'in_features') else True # Actual model
        assert model.decoder.out_features == input_dim if hasattr(model.decoder, 'out_features') else True # Actual model


    def test_nop_simple_denoise(self, processor):
        """Test _simple_denoise for small datasets."""
        # Test data: 10 samples, 2 features
        obs = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4],
                        [1.5, 2.5], [1.6, 2.6], [1.7, 2.7], [1.8, 2.8], [1.9, 2.9]])

        # Window will be min(5, 10 // 10) = min(5,1) = 1. If window < 2, returns original.
        denoised_obs = processor._simple_denoise(obs.copy())
        assert np.array_equal(denoised_obs, obs)

        # Test data: 20 samples, 1 feature. Window = min(5, 20//10) = 2
        obs_long = np.arange(20).reshape(-1,1).astype(float)
        denoised_long = processor._simple_denoise(obs_long.copy())
        assert denoised_long.shape == obs_long.shape
        # With window=2, convolve [0,1,2,3] with [0.5,0.5] mode 'same'
        # Result for first element: (0*0.5 (implicit zero padding) + 0*0.5 + 1*0.5) -> depends on padding
        # SciPy's convolve with 'same' effectively pads.
        # np.convolve([0,1,2,3], [0.5,0.5], mode='same') -> [0. , 0.5, 1.5, 2.5] (approx due to float)
        # Let's check a middle value. obs_long[2,0]=2.0. Expected: (1.0+2.0)/2 = 1.5
        assert denoised_long[2,0] == pytest.approx(1.5)
        assert denoised_long[0,0] != obs_long[0,0] # Should be changed due to smoothing

    def test_nop_denoise_small_dataset_uses_simple(self, processor, monkeypatch):
        """Test that denoise uses _simple_denoise for small datasets."""
        obs = np.random.rand(10, 3) # Small dataset (10 samples)

        # Mock _simple_denoise to check if it's called
        mock_simple_called = False
        original_simple_denoise = processor._simple_denoise
        def mock_simple(*args, **kwargs):
            nonlocal mock_simple_called
            mock_simple_called = True
            return original_simple_denoise(*args, **kwargs)

        monkeypatch.setattr(processor, '_simple_denoise', mock_simple)

        processor.denoise(obs)
        assert mock_simple_called

    def test_nop_denoise_large_dataset_mocked_ae(self, processor, monkeypatch, mock_torch_optimizer):
        """Test denoise with a larger dataset, using a mocked autoencoder."""
        obs = np.random.rand(150, 5) # Larger dataset

        # Patch the DenoisingAutoencoder class used by build_autoencoder
        monkeypatch.setattr('JanusAI.core.grammar.base_grammar.DenoisingAutoencoder', MockDenoisingAutoencoder)

        # Ensure build_autoencoder is called and uses the mock
        processor.build_autoencoder(obs.shape[1])
        assert isinstance(processor.model, MockDenoisingAutoencoder)

        # Mock the training loop part (optimizer.step, loss.backward)
        # by setting the mock model's "has_been_trained" flag after some "epochs"

        # We need to ensure that the mocked model's parameters() is called by the optimizer,
        # and that forward is called.

        # The actual training loop in denoise:
        # optimizer.zero_grad()
        # reconstructed, _ = self.model(data) # model.__call__ or model.forward
        # loss = criterion(reconstructed, data)
        # loss.backward()
        # optimizer.step()

        # For this test, we mainly care that the flow goes through the DAE path
        # and returns something of the correct shape.

        # Let's make the mock model's forward pass set the flag
        def mock_forward_set_trained(self_model, x, noise_level=0.1):
            self_model.has_been_trained = True # Simulate training happened
            # Return dummy data of correct shape. x is scaled torch tensor.
            return x.clone(), x.clone()[:, :self_model.latent_dim]

        monkeypatch.setattr(MockDenoisingAutoencoder, 'forward', mock_forward_set_trained, raising=False)

        denoised_obs = processor.denoise(obs, epochs=1) # Run with minimal epochs

        assert processor.model.has_been_trained
        assert denoised_obs.shape == obs.shape
        # Given the mock_forward returns x.clone(), inverse_transform should give back ~original obs
        # (some minor diff due to scaling and float precision)
        assert np.allclose(denoised_obs, obs, atol=1e-6)


    def test_nop_denoise_model_rebuild_on_dim_change(self, processor, monkeypatch, mock_torch_optimizer):
        """Test that the DAE model is rebuilt if input dimensions change."""
        monkeypatch.setattr('JanusAI.core.grammar.base_grammar.DenoisingAutoencoder', MockDenoisingAutoencoder)

        obs1 = np.random.rand(150, 5)
        processor.denoise(obs1, epochs=1)
        model1_id = id(processor.model)
        assert processor.model.encoder[0].in_features == 5


        obs2 = np.random.rand(150, 8) # Different number of features
         # Reset mock model's trained state for clarity if needed, though denoise should rebuild
        processor.model.has_been_trained = False
        processor.denoise(obs2, epochs=1)
        model2_id = id(processor.model)

        assert model1_id != model2_id, "Model should have been rebuilt for different input_dim."
        assert processor.model.encoder[0].in_features == 8


from JanusAI.core.grammar.base_grammar import ProgressiveGrammar, Expression # Expression is needed
from sklearn.decomposition import FastICA # For mocking
import logging # For checking log messages from CFG generation

# Mock FastICA for ProgressiveGrammar tests
class MockFastICA:
    def __init__(self, n_components, random_state):
        self.n_components_ = n_components
    def fit_transform(self, X):
        # Return mock components, ensuring shape is (n_samples, n_components)
        # Make it simple: just return the first n_components columns of X, or zeros if not enough.
        n_samples, n_features = X.shape
        if self.n_components_ <= n_features:
            return X[:, :self.n_components_]
        else: # Not enough features in X to provide self.n_components_
              # This case should ideally be handled by n_components logic in discover_variables
              # which does min(max_variables, clean_obs.shape[1])
              # So, self.n_components_ should not exceed n_features passed to fit_transform.
              # However, to be safe, pad with zeros if it happens.
            components = np.zeros((n_samples, self.n_components_))
            components[:, :n_features] = X
            return components
    def fit(self,X): # if separate fit and transform are used
        return self


@pytest.fixture
def progressive_grammar_instance(monkeypatch):
    """Fixture for a ProgressiveGrammar instance with mocked dependencies."""
    # Mock NoisyObservationProcessor's denoise method
    def mock_denoise(self_nop, observations, epochs=50): # Add self_nop to mock method
        return observations # Pass through data for predictable testing
    monkeypatch.setattr(NoisyObservationProcessor, 'denoise', mock_denoise)

    # Mock sklearn.decomposition.FastICA
    monkeypatch.setattr('JanusAI.core.grammar.base_grammar.FastICA', MockFastICA)

    grammar = ProgressiveGrammar(noise_threshold=0.01, mdl_threshold=5.0) # Use low thresholds for easier testing
    return grammar

class TestProgressiveGrammar:
    def test_pg_initialization(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        assert pg.max_variables == 20
        assert pg.noise_threshold == 0.01
        assert pg.mdl_threshold == 5.0
        assert 'constants' in pg.primitives
        assert '+' in pg.primitives['binary_ops']
        assert 'sin' in pg.primitives['unary_ops']
        assert isinstance(pg.denoiser, NoisyObservationProcessor)

    def test_pg_add_operators(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        pg.add_operators(['custom_unary', 'custom_binary', '**2', '1/']) # Test with some known patterns
        # Note: 'custom_unary' and 'custom_binary' will be ignored as they are not in known lists
        assert '**' in pg.primitives['binary_ops'] # from '**2'
        assert 'inv' in pg.primitives['unary_ops'] # from '1/'
        # Check that unknown ops are not added (current behavior prints warning)
        assert 'custom_unary' not in pg.primitives['unary_ops']
        assert 'custom_binary' not in pg.primitives['binary_ops']


    def test_pg_discover_variables_simple(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        # Observations: 100 samples, 3 features.
        # MockFastICA will return these as components if n_components=3.
        # We need to ensure _analyze_component returns high enough info content.
        obs = np.random.rand(100, 3)
        obs[:,0] = np.sin(np.linspace(0,10,100)) # Periodic
        obs[:,1] = np.linspace(0,1,100) # Smooth
        obs[:,2] = 1.0 # Constant-like (low info, but might pass if noise_threshold is very low)

        # Mock _analyze_component to return predictable properties
        def mock_analyze(component_data, time_stamps):
            props = {'information_content': 0.5, 'conservation_score': 0.1,
                     'periodicity_score': 0.1, 'smoothness': 0.1}
            if np.allclose(component_data, obs[:,0]): # Periodic
                props['periodicity_score'] = 10.0
            elif np.allclose(component_data, obs[:,1]): # Smooth
                props['smoothness'] = 0.9
            elif np.allclose(component_data, obs[:,2]): # Constant-like
                props['information_content'] = 0.05 # Below default 0.1, but test instance has 0.01
                props['conservation_score'] = 0.95
            return props

        pg._analyze_component = mock_analyze # Monkeypatch directly on instance for this test

        discovered = pg.discover_variables(obs)

        assert len(discovered) == 3 # All 3 should pass noise_threshold=0.01
        var_names = [v.name for v in discovered]
        assert "theta_1" in var_names # From periodic
        assert "x_1" in var_names     # From smooth
        assert "E_1" in var_names     # From constant-like with high conservation
        assert pg.variables["theta_1"] == discovered[var_names.index("theta_1")]


    def test_pg_discover_variables_no_components_from_ica(self, progressive_grammar_instance, monkeypatch):
        pg = progressive_grammar_instance
        obs = np.random.rand(100,1) # Only one feature
        pg.max_variables = 0 # Force num_components to be 0

        discovered = pg.discover_variables(obs)
        assert len(discovered) == 0

        # Test ICA failure
        pg.max_variables = 2 # Reset
        def mock_fit_transform_fail(self_ica, X):
            raise ValueError("ICA failed")
        monkeypatch.setattr(MockFastICA, 'fit_transform', mock_fit_transform_fail)

        discovered_fail = pg.discover_variables(obs)
        assert len(discovered_fail) == 0


    def test_pg_create_expression_valid_and_invalid(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        var_x = Variable("x",0)
        pg.variables["x"] = var_x # Add to grammar's known variables

        # Valid
        expr_plus = pg.create_expression('+', [var_x, Expression('const', [1.0])])
        assert expr_plus is not None
        assert str(expr_plus.symbolic) == "x + 1.0"

        # Invalid (arity)
        expr_bad_plus = pg.create_expression('+', [var_x])
        assert expr_bad_plus is None

        # Invalid (unknown operator for validation purposes - _validate_expression checks pg.primitives)
        expr_unknown_op = pg.create_expression('unknown_op', [var_x])
        assert expr_unknown_op is None

    def test_pg_expression_key_commutative(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        var_x = Variable("x",0)
        var_y = Variable("y",1)
        expr_x_plus_y = Expression('+', [var_x, var_y])
        expr_y_plus_x = Expression('+', [var_y, var_x])

        # '+' is in COMMUTATIVE_OPS by default in ProgressiveGrammar
        key1 = pg._expression_key(expr_x_plus_y)
        key2 = pg._expression_key(expr_y_plus_x)
        assert key1 == key2

        expr_x_minus_y = Expression('-', [var_x, var_y])
        expr_y_minus_x = Expression('-', [var_y, var_x])
        key3 = pg._expression_key(expr_x_minus_y)
        key4 = pg._expression_key(expr_y_minus_x)
        assert key3 != key4 # '-' is not commutative

    def test_pg_add_learned_function(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        var_x = Variable("x",0)
        pg.variables["x"] = var_x
        # f(x) = x + 1
        candidate_expr = Expression('+', [var_x, Expression('const', [1.0])])
        # Corpus where f(x) appears
        corpus = [
            Expression('*', [candidate_expr.clone(), Expression('const',[2.0])]), # (x+1)*2
            candidate_expr.clone() # x+1
        ]

        # Mock compression gain
        pg._calculate_compression_gain = lambda cand, corp: pg.mdl_threshold + 1.0 # Force gain > threshold

        added = pg.add_learned_function("f1", candidate_expr, corpus)
        assert added is True
        assert "f1" in pg.learned_functions
        assert pg.learned_functions["f1"] == candidate_expr
        assert "f1" in pg.primitives['unary_ops'] # Assumes new learned functions are unary

        pg._calculate_compression_gain = lambda cand, corp: pg.mdl_threshold - 1.0 # Force gain < threshold
        added_fail = pg.add_learned_function("f2", candidate_expr, corpus)
        assert added_fail is False
        assert "f2" not in pg.learned_functions

    def test_pg_calculate_compression_gain(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        var_x = Variable("x",0); pg.variables["x"]=var_x
        # Candidate: x+1 (Complexity: 1(+) + C(x) + C(1) = 1+2+2=5)
        # C(x) = 1(var op) + 1(x var) = 2
        # C(1) = 1(const op) + 1(1 const) = 2
        candidate = Expression('+', [Expression('var',[var_x]), Expression('const',[1.0])])
        assert candidate.complexity == 5

        # Corpus:
        # expr1 = (x+1)*2 (Complexity: 1(*) + C(x+1) + C(2) = 1+5+2=8)
        # expr2 = sin(x+1) (Complexity: 1(sin) + C(x+1) = 1+5=6)
        # expr3 = y (Complexity: C(y)=2, assume y is another var)
        var_y = Variable("y",1); pg.variables["y"]=var_y
        expr1 = Expression('*', [candidate.clone(), Expression('const', [2.0])])
        expr2 = Expression('sin', [candidate.clone()])
        expr3 = Expression('var', [var_y])
        corpus = [expr1, expr2, expr3]
        current_length = expr1.complexity + expr2.complexity + expr3.complexity # 8 + 6 + 2 = 16

        # Expected new length:
        # Cost of defining candidate "f_new = x+1" is C(candidate) = 5.
        # Corpus rewritten:
        # expr1_rw = f_new * 2 (C = 1(*) + C(f_new_ref) + C(2) = 1+1+2 = 4). Saved: C(candidate)-1 = 5-1=4. Original 8.
        # expr2_rw = sin(f_new) (C = 1(sin) + C(f_new_ref) = 1+1 = 2). Saved: C(candidate)-1 = 5-1=4. Original 6.
        # expr3 not changed. C = 2.
        # Total new_length = C(candidate_def) + C(expr1_rw) + C(expr2_rw) + C(expr3_rw)
        #                  = 5 + (8 - (5-1)) + (6 - (5-1)) + 2
        #                  = 5 + (8-4) + (6-4) + 2 = 5 + 4 + 2 + 2 = 13
        # Gain = current_length - new_length = 16 - 13 = 3

        gain = pg._calculate_compression_gain(candidate, corpus)
        assert gain == 3.0

    def test_pg_mine_abstractions(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        var_x = Variable("x",0); pg.variables["x"]=var_x
        var_y = Variable("y",1); pg.variables["y"]=var_y

        # Sub-expression pattern: (x+1)
        sub_pattern = Expression('+', [Expression('var',[var_x]), Expression('const',[1.0])]) # C=5

        # Hypothesis library
        # (x+1)*y
        # sin(x+1)
        # (x+1) + (x+1)  -- pattern appears twice here if _count_subexpression is exact match
        # (y+1) -- different pattern

        h1 = Expression('*', [sub_pattern.clone(), Expression('var',[var_y])])
        h2 = Expression('sin', [sub_pattern.clone()])
        h3 = Expression('+', [sub_pattern.clone(), sub_pattern.clone()])
        h4 = Expression('+', [Expression('var',[var_y]), Expression('const',[1.0])])
        hypothesis_library = [h1, h2, h3, h4]

        pg.mdl_threshold = 0.01 # Ensure even small gains lead to abstraction

        abstractions = pg.mine_abstractions(hypothesis_library, min_frequency=3)

        assert len(abstractions) == 1
        func_name = list(abstractions.keys())[0]
        assert func_name.startswith("f_")
        # Check if the abstracted expression is (x+1)
        # Need to compare Expression objects. The .symbolic might be easier.
        assert str(abstractions[func_name].symbolic) == str(sub_pattern.symbolic)
        assert func_name in pg.learned_functions

    def test_pg_cfg_generation(self, progressive_grammar_instance, caplog):
        pg = progressive_grammar_instance
        var_x = Variable("x",0); pg.variables["x"] = var_x
        var_y = Variable("y",1); pg.variables["y"] = var_y

        rules = [
            CFGRule("EXPR", ["OP", "EXPR", "EXPR"], weight=0.3),
            CFGRule("EXPR", ["VAR"], weight=0.6),
            CFGRule("EXPR", ["CONST"], weight=0.1),
            CFGRule("OP",   ["+"], weight=0.5),
            CFGRule("OP",   ["-"], weight=0.5),
            CFGRule("VAR",  [var_x]), # Direct Variable object
            CFGRule("VAR",  ["y"]),   # Variable name string
            CFGRule("CONST",["CONST"]) # Special symbol for ProgressiveGrammar
        ]
        pg.set_rules_from_cfg(rules, start_symbol="EXPR")
        assert pg.cfg_grammar is not None
        assert pg.cfg_start_symbol == "EXPR"

        # Make generation somewhat predictable for testing
        random.seed(10)
        caplog.set_level(logging.INFO)

        expr = pg.generate_random_expression_from_cfg(max_depth=5)

        assert expr is not None
        assert isinstance(expr, Expression)
        # Based on seed 10 and the grammar, a possible output could be x, y, a const, or simple op like x+y
        # Example: if VAR -> "y" is chosen: Expression(operator='var', operands=['y'])
        # Example: if CONST is chosen: Expression(operator='const', operands=[<random_num>])
        # This is hard to assert exact structure without deep diving into random sequence.
        # Check if it's a valid expression structure.
        assert expr.operator in ['var', 'const', '+', '-']
        if expr.operator == 'var':
            assert isinstance(expr.operands[0], str) # name of variable
            assert expr.operands[0] in ["x", "y"]
        elif expr.operator == 'const':
            assert isinstance(expr.operands[0], (int, float))
        else: # binary op
            assert len(expr.operands) == 2
            assert all(isinstance(op, Expression) for op in expr.operands)

        # Test generation failure (e.g., if max_depth is too small for any terminal production)
        caplog.clear()
        expr_fail = pg.generate_random_expression_from_cfg(max_depth=0) # max_depth 0 from start
        # This should hit max depth warning in _generate_from_symbol_cfg
        # and likely return None as no expression can be formed.
        assert expr_fail is None
        assert "Max recursion depth 0 exceeded" in caplog.text # or similar warning

    def test_pg_get_arity_is_operator_known(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        # Default primitives
        assert pg.get_arity('+') == 2
        assert pg.get_arity('sin') == 1
        assert pg.get_arity('diff') == 2
        assert pg.is_operator_known('+') is True

        with pytest.raises(ValueError):
            pg.get_arity('unknown_op')
        assert pg.is_operator_known('unknown_op') is False

        # Add a learned function (assumed unary)
        var_x = Variable("x",0); pg.variables["x"]=var_x
        learned_f_expr = Expression('+', [Expression('var',[var_x]), Expression('const',[1.0])])
        pg.learned_functions['my_func'] = learned_f_expr
        pg.primitives['unary_ops'].add('my_func') # Manually add to primitives for arity lookup by current _validate_expression

        # This test reveals that get_arity/is_operator_known might not automatically know about learned_functions
        # unless they are also added to primitive sets. The current _validate_expression DOES check learned_functions.
        # The current get_arity is a lambda that only checks primitive sets.
        # This suggests a potential inconsistency or area for refinement in ProgressiveGrammar.
        # For now, test based on current implementation. If 'my_func' is added to unary_ops, it should be found.
        assert pg.get_arity('my_func') == 1
        assert pg.is_operator_known('my_func') is True


    def test_pg_export_grammar_state(self, progressive_grammar_instance):
        pg = progressive_grammar_instance
        var_x = Variable("x",0, properties={"unit":"m"})
        pg.variables["x"] = var_x

        expr_f1 = Expression('sin', [Expression('var', [var_x])])
        pg.learned_functions["f1"] = expr_f1

        state = pg.export_grammar_state()

        assert "variables" in state
        assert "x" in state["variables"]
        assert state["variables"]["x"]["index"] == 0
        assert state["variables"]["x"]["properties"] == {"unit":"m"}

        assert "learned_functions" in state
        assert "f1" in state["learned_functions"]
        f1_dict = state["learned_functions"]["f1"]
        assert f1_dict["operator"] == "sin"
        assert f1_dict["operands"][0]["operator"] == "var"
        assert f1_dict["operands"][0]["operands"][0]["name"] == "x" # Check var name in dict
        assert f1_dict["complexity"] == expr_f1.complexity

        assert "proven_lemmas" in state # Should be empty by default
        assert not state["proven_lemmas"]


from JanusAI.core.grammar.base_grammar import AIGrammar # AIGrammar specific import
import sympy as sp # For checking symbolic forms from AIGrammar

@pytest.fixture
def ai_grammar_instance():
    """Fixture for an AIGrammar instance."""
    return AIGrammar()

class TestAIGrammar:
    def test_aig_initialization(self, ai_grammar_instance):
        ag = ai_grammar_instance
        assert not ag.primitives['constants'] # No default constants from ProgressiveGrammar
        assert not ag.primitives['binary_ops'] - {'residual'} # Only 'residual' might be there initially
        assert not ag.primitives['calculus_ops'] # No default calculus ops

        # Check for AI specific primitives
        assert 'activation_types' in ag.primitives['custom_sets']
        assert 'relu' in ag.primitives['custom_sets']['activation_types']
        assert 'attention' in ag.primitives['custom_callable_ops']
        assert 'relu' in ag.primitives['unary_ops'] # Added during AIGrammar init
        assert 'residual' in ag.primitives['binary_ops']

    def test_aig_add_primitive_set_and_primitive(self, ai_grammar_instance):
        ag = ai_grammar_instance
        ag.add_primitive_set("my_layers", ["dense", "conv2d"])
        assert "my_layers" in ag.primitives["custom_sets"]
        assert "dense" in ag.primitives["custom_sets"]["my_layers"]

        def my_custom_op(x): return x
        ag.add_primitive("my_op", my_custom_op, category="test_custom_ops")
        assert "test_custom_ops" in ag.primitives
        assert ag.primitives["test_custom_ops"]["my_op"] == my_custom_op

        ag.add_primitive("my_values", [1,2,3], category="named_lists") # Test named_lists category
        assert "my_values" in ag.primitives["named_lists"]

        ag.add_primitive("my_single_value", 10, category="custom_values")
        assert "my_single_value" in ag.primitives["custom_values"]


    def test_aig_attention_op_symbolic(self, ai_grammar_instance):
        ag = ai_grammar_instance
        q_sym = sp.Symbol("Q")
        k_sym = sp.Symbol("K")
        v_sym = sp.Symbol("V")

        # Test symbolic mode of _attention_op
        symbolic_attn = ag._attention_op(q_sym, k_sym, v_sym)
        assert str(symbolic_attn) == "Attention(Q, K, V)"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not available for AIGrammar numeric ops.")
    def test_aig_attention_op_numeric_mocked(self, ai_grammar_instance, monkeypatch):
        ag = ai_grammar_instance
        # Mock torch.tensor, torch.matmul, torch.softmax, np.sqrt for numeric mode
        # This is to avoid actual tensor computations but check the flow
        mock_output_tensor = torch.rand(2,3) # Example shape

        monkeypatch.setattr(torch, 'tensor', lambda x, dtype: x) # Pass through numpy array
        monkeypatch.setattr(torch, 'matmul', lambda a,b : mock_output_tensor if b.ndim == 2 else mock_output_tensor) # Simplified mock
        monkeypatch.setattr(torch, 'softmax', lambda x, dim: x) # Pass through
        monkeypatch.setattr(np, 'sqrt', lambda x: 1.0) # Mock sqrt

        q_np = np.random.rand(2,4)
        k_np = np.random.rand(2,4)
        v_np = np.random.rand(2,3) # Values determine output's last dim if using matmul(weights, v)

        numeric_attn_result = ag._attention_op(q_np, k_np, v_np)
        assert isinstance(numeric_attn_result, np.ndarray)
        # Based on mock_matmul returning mock_output_tensor (shape 2,3)
        assert numeric_attn_result.shape == (2,3)


    def test_aig_embedding_op_symbolic(self, ai_grammar_instance):
        ag = ai_grammar_instance
        idx_sym = sp.Symbol("Indices")
        emb_matrix_sym = sp.Symbol("EmbeddingMatrix")
        symbolic_emb = ag._embedding_op(idx_sym, emb_matrix_sym)
        assert str(symbolic_emb) == "Embedding(Indices, EmbeddingMatrix)"

    def test_aig_to_sympy_policy_ai_ops(self, ai_grammar_instance):
        ag = ai_grammar_instance
        var_q = Variable("Q",0); ag.variables["Q"] = var_q
        var_k = Variable("K",1); ag.variables["K"] = var_k
        var_v = Variable("V",2); ag.variables["V"] = var_v

        # Create Expression objects for AI ops
        expr_q = Expression('var', [var_q])
        expr_k = Expression('var', [var_k])
        expr_v = Expression('var', [var_v])

        # Attention(Q, K, V)
        expr_attn = ag.create_expression('attention', [expr_q, expr_k, expr_v])
        assert expr_attn is not None, "AIGrammar failed to create 'attention' expression"
        # The Expression's _to_sympy will call AIGrammar's _to_sympy policy
        # For unknown ops in Expression, it defaults to sp.Function(op.capitalize())
        # AIGrammar's _to_sympy policy should override this for its known AI ops.
        assert str(expr_attn.symbolic) == "Attention(Q, K, V)"

        # Relu(Q)
        expr_relu = ag.create_expression('relu', [expr_q])
        assert expr_relu is not None
        assert str(expr_relu.symbolic) == "Relu(Q)" # AIGrammar policy for other AI ops

    def test_aig_validate_expression_ai_ops(self, ai_grammar_instance):
        ag = ai_grammar_instance
        var_x = Variable("x",0); ag.variables["x"] = var_x
        expr_x = Expression('var', [var_x])

        # Valid 'relu'
        assert ag._validate_expression('relu', [expr_x]) is True
        # Invalid 'relu' (wrong arity)
        assert ag._validate_expression('relu', [expr_x, expr_x]) is False

        # Valid 'attention'
        var_k = Variable("k",1); ag.variables["k"] = var_k
        var_v = Variable("v",2); ag.variables["v"] = var_v
        expr_k = Expression('var', [var_k])
        expr_v = Expression('var', [var_v])
        assert ag._validate_expression('attention', [expr_x, expr_k, expr_v]) is True
        # Invalid 'attention' (wrong arity)
        assert ag._validate_expression('attention', [expr_x, expr_k]) is False

        # Test an op not known to ProgressiveGrammar but known to AIGrammar (like 'attention')
        # create_expression uses self._validate_expression
        created_expr = ag.create_expression('attention', [expr_x, expr_k, expr_v])
        assert created_expr is not None

        # Test an op completely unknown
        assert ag._validate_expression('completely_unknown_op', [expr_x]) is False
        created_expr_unknown = ag.create_expression('completely_unknown_op', [expr_x])
        assert created_expr_unknown is None


    def test_aig_get_arity_ai_ops(self, ai_grammar_instance):
        ag = ai_grammar_instance
        assert ag.get_arity('relu') == 1
        assert ag.get_arity('attention') == 3
        assert ag.get_arity('residual') == 2

        # Test fallback to parent for standard ops (AIGrammar doesn't load them by default)
        # So, to test fairly, add a standard op to AIGrammar's primitives
        ag.primitives['binary_ops'].add('+')
        assert ag.get_arity('+') == 2

        with pytest.raises(ValueError):
            ag.get_arity('non_existent_op')

    def test_aig_interaction_with_expression_to_sympy(self, ai_grammar_instance):
        """
        Test that Expression._to_sympy correctly uses AIGrammar's policy
        if the grammar instance is an AIGrammar.
        This is implicitly tested by test_aig_to_sympy_policy_ai_ops,
        as Expression's .symbolic property calls _to_sympy.
        If an Expression is created using an AIGrammar instance, its internal _to_sympy
        should correctly defer to the AIGrammar's _to_sympy method for AI-specific ops.
        The current Expression class does not take grammar as an argument to __init__.
        It seems Expression._to_sympy is standalone and does not know about grammar policies.
        This needs clarification on how Expression would use grammar's _to_sympy policy.

        Revisiting `Expression._to_sympy()`: It does not currently have a hook to call a
        grammar policy. It handles a fixed set of operators and then has a fallback for
        unknown ones.

        Revisiting `AIGrammar._to_sympy()`: This seems to be intended as a policy that
        *would be called by Expression* if Expression was grammar-aware for symbolization.
        The line `temp_expr_for_std_op_symb = Expression(operator, sympy_operands)` in
        AIGrammar._to_sympy implies that standard ops are symbolized by creating a temporary
        Expression. This is okay if AIGrammar._to_sympy is the main entry point for symbolization.

        If ProgressiveGrammar.create_expression returns an Expression object, and that
        Expression object's __post_init__ calls its own _to_sympy, then AIGrammar's
        _to_sympy policy is NOT used by default unless Expression is modified or
        ProgressiveGrammar.create_expression has special logic for AIGrammar.

        Let's assume the design means that `Expression`'s `_to_sympy` is the final say for
        its own structure, and `AIGrammar` might use `Expression` objects as part of its
        representation, but `AIGrammar._to_sympy` is more of a utility or a policy for
        a future, more integrated system.

        Given current code:
        `Expression(operator='relu', operands=[...])` will use `Expression._to_sympy`.
        If 'relu' is not in Expression's known list, it becomes `sp.Function('Relu')`.
        `AIGrammar` adds 'relu' to its `primitives['unary_ops']`.
        `AIGrammar._validate_expression` would allow it.
        `AIGrammar.create_expression` would thus create `Expression('relu', ...)`
        The symbolic form would be `Relu(...)` due to `Expression._to_sympy`'s fallback.
        This matches the assertion in `test_aig_to_sympy_policy_ai_ops`.

        This test confirms that the symbolic representation of AI operators, when an Expression
        is created via AIGrammar, relies on Expression's fallback for unknown operators,
        which AIGrammar's _to_sympy policy happens to align with (Function(OpName.capitalize())).
        """
        ag = ai_grammar_instance
        var_x = Variable("x",0); ag.variables["x"] = var_x
        expr_x_node = Expression('var', [var_x])

        # Create an expression for an AI op like 'softmax' using AIGrammar
        # AIGrammar adds 'softmax' to its 'unary_ops'
        expr_softmax = ag.create_expression('softmax', [expr_x_node])
        assert expr_softmax is not None
        # Expression._to_sympy will handle 'softmax' as an unknown op.
        # It will generate sp.Function('Softmax')(Symbol('x'))
        assert str(expr_softmax.symbolic) == "Softmax(x)"

        # Create an expression for a standard op that AIGrammar doesn't load by default
        # Add '+' to AIGrammar for this test
        ag.primitives['binary_ops'].add('+')
        expr_plus = ag.create_expression('+', [expr_x_node, Expression('const', [1.0])])
        assert expr_plus is not None
        # Expression._to_sympy knows '+'
        assert str(expr_plus.symbolic) == "x + 1.0"
