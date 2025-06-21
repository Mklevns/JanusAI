"""
Tests for ai_interpretability/evaluation/fidelity.py
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from unittest.mock import MagicMock, patch

from janus_ai.ai_interpretability.evaluation.fidelity import FidelityCalculator

# --- Helper Mocks ---
@pytest.fixture
def mock_ai_model():
    model = MagicMock(spec=nn.Module) # Mocking a generic nn.Module
    model.eval = MagicMock() # Mock the eval method

    # Mocking the output of the model for attention extraction
    # (batch, heads, seq, seq)
    mock_attention_output = torch.rand(1, 2, 4, 4) # Batch=1, Heads=2, SeqLen=4
    # Mocking model outputs for 'output' behavior
    mock_logits_output = torch.rand(1, 4, 10) # Batch=1, SeqLen=4, NumClasses=10


    def model_forward_side_effect(*args, **kwargs):
        if kwargs.get('output_attentions'):
            outputs = MagicMock()
            outputs.attentions = [mock_attention_output] # List of tensors, one per layer
            return outputs
        else: # For target_behavior == 'output'
            outputs = MagicMock()
            outputs.logits = mock_logits_output
            return outputs


    model.forward = MagicMock(side_effect=model_forward_side_effect)
    # Make the mock model callable to simulate `ai_model(input_ids=...)`
    model.__call__ = model.forward
    return model

@pytest.fixture
def mock_expression():
    expr = MagicMock()
    expr.symbolic = sp.Symbol('x') + sp.Symbol('y')
    return expr

@pytest.fixture
def mock_variables():
    var_x = MagicMock()
    var_x.name = 'x'
    var_y = MagicMock()
    var_y.name = 'y'
    return [var_x, var_y]

@pytest.fixture
def sample_test_data():
    return {
        'input_ids': np.random.randint(0, 100, (1, 4)), # Batch=1, SeqLen=4
        'inputs': np.random.randint(0, 100, (1, 4)), # Alternative key for input_ids
        'attention_mask': np.ones((1, 4)),
        'target_layer': 0,
        'target_head': 0,
        'sequence_length': 4,
        'x': np.array([1, 2, 3, 4]),
        'y': np.array([5, 6, 7, 8]),
        'token_types': np.random.randint(0, 2, (4,))
    }

@pytest.fixture
def fidelity_calculator_instance():
    return FidelityCalculator()

# --- Tests for FidelityCalculator ---

class TestFidelityCalculator:

    def test_init(self):
        calc = FidelityCalculator(correlation_threshold=0.05, max_eval_points=5000, numerical_tolerance=1e-7)
        assert calc.correlation_threshold == 0.05
        assert calc.max_eval_points == 5000
        assert calc.numerical_tolerance == 1e-7

    # --- Tests for _extract_ground_truth ---
    def test_extract_ground_truth_attention_specific_head(self, fidelity_calculator_instance, mock_ai_model, sample_test_data):
        sample_test_data['target_head'] = 0 # Specific head
        ground_truth = fidelity_calculator_instance._extract_ground_truth(mock_ai_model, sample_test_data, 'attention')
        # Expected shape: (batch * seq_len * seq_len) = 1 * 4 * 4 = 16 (flattened)
        # mock_attention_output is (1,2,4,4), specific head is (1,4,4)
        assert ground_truth.shape == (1 * 4 * 4,)
        mock_ai_model.eval.assert_called_once()
        mock_ai_model.forward.assert_called_once()


    def test_extract_ground_truth_attention_average_heads(self, fidelity_calculator_instance, mock_ai_model, sample_test_data):
        sample_test_data['target_head'] = None # Average over heads
        # Mock ai_model's attention output to have multiple heads
        mock_attention_output = torch.rand(1, 2, 4, 4) # Batch=1, Heads=2, SeqLen=4

        # Adjusting side effect for this specific test case if necessary, or ensure default mock is fine
        def model_forward_side_effect(*args, **kwargs):
            outputs = MagicMock()
            outputs.attentions = [mock_attention_output]
            return outputs
        mock_ai_model.forward.side_effect = model_forward_side_effect
        mock_ai_model.eval.reset_mock() # Reset from previous calls if any
        mock_ai_model.forward.reset_mock()


        ground_truth = fidelity_calculator_instance._extract_ground_truth(mock_ai_model, sample_test_data, 'attention')
        assert ground_truth.shape == (1 * 4 * 4,) # Shape remains the same after averaging and flattening
        mock_ai_model.eval.assert_called_once()
        mock_ai_model.forward.assert_called_once()


    @patch('JanusAI.ai_interpretability.evaluation.fidelity.FidelityCalculator._extract_output_logits')
    def test_extract_ground_truth_output(self, mock_extract_logits, fidelity_calculator_instance, mock_ai_model, sample_test_data):
        mock_extract_logits.return_value = np.random.rand(10)
        ground_truth = fidelity_calculator_instance._extract_ground_truth(mock_ai_model, sample_test_data, 'output')
        mock_extract_logits.assert_called_once_with(mock_ai_model, sample_test_data)
        assert ground_truth.shape == (10,)

    def test_extract_ground_truth_invalid_target(self, fidelity_calculator_instance, mock_ai_model, sample_test_data):
        with pytest.raises(ValueError, match="Unknown target behavior: invalid_target"):
            fidelity_calculator_instance._extract_ground_truth(mock_ai_model, sample_test_data, 'invalid_target')

    # --- Tests for _create_variable_substitutions ---
    def test_create_variable_substitutions(self, fidelity_calculator_instance, sample_test_data):
        var_pos_diff = MagicMock(); var_pos_diff.name = 'pos_diff'
        var_pos_ratio = MagicMock(); var_pos_ratio.name = 'pos_ratio'
        var_token_type_i = MagicMock(); var_token_type_i.name = 'token_type_i'
        var_rel_pos = MagicMock(); var_rel_pos.name = 'relative_pos'
        var_direct = MagicMock(); var_direct.name = 'x' # From sample_test_data
        var_missing = MagicMock(); var_missing.name = 'z' # Not in sample_test_data

        variables = [var_pos_diff, var_pos_ratio, var_token_type_i, var_rel_pos, var_direct, var_missing]
        subs = fidelity_calculator_instance._create_variable_substitutions(sample_test_data, variables)

        seq_len = sample_test_data['sequence_length']
        assert sp.Symbol('pos_diff') in subs and subs[sp.Symbol('pos_diff')].shape == (seq_len * seq_len,)
        assert sp.Symbol('pos_ratio') in subs and subs[sp.Symbol('pos_ratio')].shape == (seq_len * seq_len,)
        assert sp.Symbol('token_type_i') in subs # Will use sample_test_data['token_types']
        assert sp.Symbol('relative_pos') in subs and subs[sp.Symbol('relative_pos')].shape == (seq_len * seq_len,)
        assert sp.Symbol('x') in subs and np.array_equal(subs[sp.Symbol('x')], sample_test_data['x'])
        assert sp.Symbol('z') in subs # Should default to random values

    # --- Tests for _evaluate_expression ---
    def test_evaluate_expression_symbolic_obj(self, fidelity_calculator_instance, mock_expression, sample_test_data, mock_variables):
        # This test will likely go into _evaluate_attention_expression due to 'attention' in name
        # or _evaluate_standard_expression. Let's make the mock_expression simple.
        mock_expression.symbolic = sp.Symbol('x') + 2
        # Ensure 'x' is in var_subs from _create_variable_substitutions
        sample_test_data['x'] = np.array([1, 2, 3])
        mock_variables[0].name = 'x'

        # Mock _create_variable_substitutions to return controlled subs
        with patch.object(fidelity_calculator_instance, '_create_variable_substitutions') as mock_create_subs:
            mock_create_subs.return_value = {sp.Symbol('x'): np.array([1,2,3])}

            # Mock _evaluate_standard_expression as it's simpler to test directly here
            with patch.object(fidelity_calculator_instance, '_evaluate_standard_expression') as mock_eval_standard:
                mock_eval_standard.return_value = np.array([3, 4, 5]) # Expected: x+2

                result = fidelity_calculator_instance._evaluate_expression(mock_expression, sample_test_data, [mock_variables[0]])
                mock_eval_standard.assert_called_once()
                assert np.array_equal(result, np.array([3, 4, 5]))


    def test_evaluate_expression_sympy_expr(self, fidelity_calculator_instance, sample_test_data, mock_variables):
        expr = sp.Symbol('y') * 3
        sample_test_data['y'] = np.array([1,1,1])
        mock_variables[0].name = 'y'

        with patch.object(fidelity_calculator_instance, '_create_variable_substitutions') as mock_create_subs:
            mock_create_subs.return_value = {sp.Symbol('y'): np.array([1,1,1])}
            with patch.object(fidelity_calculator_instance, '_evaluate_standard_expression') as mock_eval_standard:
                mock_eval_standard.return_value = np.array([3,3,3])

                result = fidelity_calculator_instance._evaluate_expression(expr, sample_test_data, [mock_variables[0]])
                assert np.array_equal(result, np.array([3,3,3]))


    def test_evaluate_expression_string_expr(self, fidelity_calculator_instance, sample_test_data, mock_variables):
        expr_str = "x / 2"
        sample_test_data['x'] = np.array([2,4,6])
        mock_variables[0].name = 'x'

        with patch.object(fidelity_calculator_instance, '_create_variable_substitutions') as mock_create_subs:
            mock_create_subs.return_value = {sp.Symbol('x'): np.array([2,4,6])}
            with patch.object(fidelity_calculator_instance, '_evaluate_standard_expression') as mock_eval_standard:
                mock_eval_standard.return_value = np.array([1,2,3])

                result = fidelity_calculator_instance._evaluate_expression(expr_str, sample_test_data, [mock_variables[0]])
                assert np.array_equal(result, np.array([1,2,3]))

    def test_evaluate_expression_attention_path(self, fidelity_calculator_instance, sample_test_data, mock_variables):
        # Expression contains 'attention' to trigger the specific path
        expr_str = "attention_var * 0.5"
        # We need a variable that will be part of var_subs for the attention path
        mock_variables[0].name = 'pos_diff' # pos_diff is used in _evaluate_attention_expression indirectly

        # Mock _create_variable_substitutions
        with patch.object(fidelity_calculator_instance, '_create_variable_substitutions') as mock_create_subs:
            # pos_diff will be (seq_len*seq_len,) = 16 elements for seq_len=4
            mock_pos_diff_values = np.arange(16).astype(float)
            mock_create_subs.return_value = {sp.Symbol('pos_diff'): mock_pos_diff_values}

            # Mock _evaluate_attention_expression
            with patch.object(fidelity_calculator_instance, '_evaluate_attention_expression') as mock_eval_attention:
                expected_result = mock_pos_diff_values * 0.5 # Simulate the expression's effect
                mock_eval_attention.return_value = expected_result

                # Use a sympy expression that would be created from expr_str
                symbolic_expr = sp.sympify(expr_str.replace("attention_var", "pos_diff")) # Simulate sympify

                result = fidelity_calculator_instance._evaluate_expression(symbolic_expr, sample_test_data, mock_variables)
                mock_eval_attention.assert_called_once()
                assert np.array_equal(result, expected_result)


    def test_evaluate_expression_none(self, fidelity_calculator_instance, sample_test_data, mock_variables):
        result = fidelity_calculator_instance._evaluate_expression(None, sample_test_data, mock_variables)
        assert np.array_equal(result, np.zeros(1))

    def test_evaluate_expression_eval_fails(self, fidelity_calculator_instance, sample_test_data, mock_variables):
        expr_str = "x / 0" # This would cause error if not handled by lambdify/sympy
        mock_variables[0].name = 'x'
        sample_test_data['x'] = np.array([1])

        # Patch _create_variable_substitutions to control inputs to the evaluation
        with patch.object(fidelity_calculator_instance, '_create_variable_substitutions') as mock_create_subs:
            mock_create_subs.return_value = {sp.Symbol('x'): np.array([1])}

            # Depending on which path (standard/attention) is taken, mock that one to raise error
            # Assuming it goes to standard for "x/0"
            with patch.object(fidelity_calculator_instance, '_evaluate_standard_expression', side_effect=Exception("Eval error")):
                result = fidelity_calculator_instance._evaluate_expression(expr_str, sample_test_data, mock_variables)
                assert np.array_equal(result, np.zeros(1)) # Should return zeros on failure


    # --- Tests for _evaluate_attention_expression ---
    def test_evaluate_attention_expression(self, fidelity_calculator_instance):
        expr = sp.Symbol('a') + sp.Symbol('b')
        var_subs = {sp.Symbol('a'): np.array([1,2,3]), sp.Symbol('b'): np.array([4,5,6])}
        result = fidelity_calculator_instance._evaluate_attention_expression(expr, {}, var_subs)
        assert np.array_equal(result, np.array([5,7,9]))

    def test_evaluate_attention_expression_scalar_result(self, fidelity_calculator_instance):
        expr = sp.Integer(5) # A constant expression
        var_subs = {sp.Symbol('a'): np.array([1,2,3])} # Need some var_subs to determine length
        result = fidelity_calculator_instance._evaluate_attention_expression(expr, {}, var_subs)
        assert np.array_equal(result, np.array([5,5,5]))

    def test_evaluate_attention_expression_empty_values(self, fidelity_calculator_instance):
        expr = sp.Symbol('a')
        var_subs = {sp.Symbol('a'): np.array([])}
        # This will cause min_length to be 0 in the original code, which might lead to issues.
        # The current code would try to create np.zeros(100) if min_length is not determined.
        # Let's assume it should return an empty array or handle it gracefully.
        # The `func(*values)` might fail if values are empty.
        # sp.lambdify with empty symbols list might also be an issue.
        # For now, let's expect it to return a default zero array if things go wrong.
        # The code has: return np.zeros(min_length if 'min_length' in locals() else 100)
        # If var_subs is empty, symbols will be empty, values will be empty.
        # min_length might not be defined if values is empty.

        # Case 1: var_subs has a variable with an empty array
        result1 = fidelity_calculator_instance._evaluate_attention_expression(expr, {}, var_subs)
        assert np.array_equal(result1, np.zeros(100)) # Fallback size

        # Case 2: var_subs is empty (no variables for the expression)
        expr_const = sp.Integer(7)
        result2 = fidelity_calculator_instance._evaluate_attention_expression(expr_const, {}, {})
        # Will try to lambdify with no symbols. Result of func() will be 7.
        # Then it hits `np.isscalar(result)`. min_length won't be in locals.
        # So, np.full(100, 7)
        assert np.array_equal(result2, np.full(100, 7.0))


    # --- Tests for _compute_correlation_fidelity ---
    @pytest.mark.parametrize("gt, pred, expected_fidelity_range", [
        (np.array([1,2,3,4]), np.array([1,2,3,4]), (0.99, 1.0)),  # Perfect
        (np.array([1,2,3,4]), np.array([4,3,2,1]), (0.99, 1.0)),  # Perfect anti-correlation (abs used)
        (np.array([1,2,3,4]), np.array([1,1,1,1]), (0.0, 0.1)),  # One constant
        (np.array([1,2,3,4]), np.array([2,3,4,5]), (0.99, 1.0)),  # Shifted, perfect linear
        (np.array([1,2,3,4]), np.array([2,1,4,3]), (0.0, 0.5)),  # Low correlation
        (np.array([1,2,3,4,5]), np.array([1,4,9,16,25]), (0.9, 1.0)) # Perfect spearman, good pearson/r2
    ])
    def test_compute_correlation_fidelity_various_cases(self, fidelity_calculator_instance, gt, pred, expected_fidelity_range):
        fidelity = fidelity_calculator_instance._compute_correlation_fidelity(gt, pred)
        assert expected_fidelity_range[0] <= fidelity <= expected_fidelity_range[1]

    def test_compute_correlation_fidelity_nans_infs(self, fidelity_calculator_instance):
        gt = np.array([1,2,np.nan,4, np.inf])
        pred = np.array([1,2,3,4,5])
        fidelity = fidelity_calculator_instance._compute_correlation_fidelity(gt, pred)
        # After cleaning, gt=[1,2,4], pred=[1,2,4]. Should be perfect.
        assert 0.99 <= fidelity <= 1.0

    def test_compute_correlation_fidelity_all_nans_or_empty(self, fidelity_calculator_instance):
        assert fidelity_calculator_instance._compute_correlation_fidelity(np.array([np.nan]), np.array([np.nan])) == 0.0
        assert fidelity_calculator_instance._compute_correlation_fidelity(np.array([]), np.array([])) == 0.0
        assert fidelity_calculator_instance._compute_correlation_fidelity(np.array([1,2]), np.array([])) == 0.0


    def test_compute_correlation_fidelity_constant_arrays(self, fidelity_calculator_instance):
        # Both constant, same
        assert fidelity_calculator_instance._compute_correlation_fidelity(np.array([1,1,1]), np.array([1,1,1])) == 1.0
        # Both constant, different
        assert fidelity_calculator_instance._compute_correlation_fidelity(np.array([1,1,1]), np.array([2,2,2])) == 0.0
        # One constant, one not (already covered by parametrize, but good to be explicit)
        fidelity = fidelity_calculator_instance._compute_correlation_fidelity(np.array([1,2,3]), np.array([1,1,1]))
        assert 0.0 <= fidelity < 0.1 # Should be low as std(pred) is 0


    # --- Tests for main calculate_fidelity method ---
    def test_calculate_fidelity_successful(self, fidelity_calculator_instance, mock_ai_model, sample_test_data, mock_expression, mock_variables):
        # Mock internal methods to return controlled values
        with patch.object(fidelity_calculator_instance, '_extract_ground_truth') as mock_extract, \
             patch.object(fidelity_calculator_instance, '_evaluate_expression') as mock_evaluate, \
             patch.object(fidelity_calculator_instance, '_compute_correlation_fidelity') as mock_compute:

            mock_extract.return_value = np.array([0.1, 0.2, 0.3])
            mock_evaluate.return_value = np.array([0.1, 0.2, 0.3])
            mock_compute.return_value = 0.95 # Simulate high correlation

            fidelity = fidelity_calculator_instance.calculate_fidelity(mock_expression, mock_ai_model, sample_test_data, mock_variables)

            mock_extract.assert_called_once_with(mock_ai_model, sample_test_data, 'attention') # Default target
            mock_evaluate.assert_called_once_with(mock_expression, sample_test_data, mock_variables)
            mock_compute.assert_called_once_with(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
            assert fidelity == 0.95

    def test_calculate_fidelity_extraction_fails(self, fidelity_calculator_instance, mock_ai_model, sample_test_data, mock_expression, mock_variables):
        with patch.object(fidelity_calculator_instance, '_extract_ground_truth', side_effect=Exception("Extraction failed")):
            fidelity = fidelity_calculator_instance.calculate_fidelity(mock_expression, mock_ai_model, sample_test_data, mock_variables)
            assert fidelity == 0.0

    def test_calculate_fidelity_evaluation_fails(self, fidelity_calculator_instance, mock_ai_model, sample_test_data, mock_expression, mock_variables):
        with patch.object(fidelity_calculator_instance, '_extract_ground_truth', return_value=np.array([1])), \
             patch.object(fidelity_calculator_instance, '_evaluate_expression', side_effect=Exception("Evaluation failed")):
            fidelity = fidelity_calculator_instance.calculate_fidelity(mock_expression, mock_ai_model, sample_test_data, mock_variables)
            assert fidelity == 0.0

    def test_calculate_fidelity_correlation_fails(self, fidelity_calculator_instance, mock_ai_model, sample_test_data, mock_expression, mock_variables):
        with patch.object(fidelity_calculator_instance, '_extract_ground_truth', return_value=np.array([1])), \
             patch.object(fidelity_calculator_instance, '_evaluate_expression', return_value=np.array([1])), \
             patch.object(fidelity_calculator_instance, '_compute_correlation_fidelity', side_effect=Exception("Correlation failed")):
            fidelity = fidelity_calculator_instance.calculate_fidelity(mock_expression, mock_ai_model, sample_test_data, mock_variables)
            assert fidelity == 0.0

    def test_calculate_fidelity_clamps_score(self, fidelity_calculator_instance, mock_ai_model, sample_test_data, mock_expression, mock_variables):
        with patch.object(fidelity_calculator_instance, '_extract_ground_truth', return_value=np.array([1])), \
             patch.object(fidelity_calculator_instance, '_evaluate_expression', return_value=np.array([1])), \
             patch.object(fidelity_calculator_instance, '_compute_correlation_fidelity') as mock_compute:

            mock_compute.return_value = 1.5 # Score > 1
            assert fidelity_calculator_instance.calculate_fidelity(mock_expression, mock_ai_model, sample_test_data, mock_variables) == 1.0

            mock_compute.return_value = -0.5 # Score < 0
            assert fidelity_calculator_instance.calculate_fidelity(mock_expression, mock_ai_model, sample_test_data, mock_variables) == 0.0

    # Test for _extract_output_logits (private method, but important for 'output' target_behavior)
    def test_extract_output_logits(self, fidelity_calculator_instance, mock_ai_model, sample_test_data):
        # Ensure the mock_ai_model's forward method is set up for logits
        mock_logits_output = torch.rand(1, sample_test_data['sequence_length'], 10) # B, S, NumClasses

        def model_forward_side_effect(*args, **kwargs):
            # This version of forward is for when output_attentions is False or not present
            outputs = MagicMock()
            outputs.logits = mock_logits_output
            return outputs

        mock_ai_model.forward.side_effect = model_forward_side_effect
        # Reset mocks as they might have been called by other tests or fixtures
        mock_ai_model.eval.reset_mock()
        mock_ai_model.forward.reset_mock()

        logits = fidelity_calculator_instance._extract_output_logits(mock_ai_model, sample_test_data)

        mock_ai_model.eval.assert_called_once()
        mock_ai_model.forward.assert_called_once()
        # Check called with input_ids and attention_mask, but NOT output_attentions=True
        call_args = mock_ai_model.forward.call_args
        assert 'input_ids' in call_args[1]
        assert 'attention_mask' in call_args[1]
        assert not call_args[1].get('output_attentions', False) # Should be False or absent

        # Expected shape: (batch_size * sequence_length * num_classes) flattened
        expected_shape_flat = mock_logits_output.numel()
        assert logits.shape == (expected_shape_flat,)
        assert np.array_equal(logits, mock_logits_output.detach().cpu().numpy().flatten())

    # Test for _evaluate_standard_expression (private, but core logic)
    def test_evaluate_standard_expression(self, fidelity_calculator_instance):
        expr = sp.Symbol('a') * 2 + sp.Symbol('b')
        # For standard evaluation, it expects scalar results after substitution
        # The var_subs should contain scalar values for this path usually
        var_subs_scalar = {sp.Symbol('a'): 3, sp.Symbol('b'): 4} # a=3, b=4 -> 3*2 + 4 = 10
        result = fidelity_calculator_instance._evaluate_standard_expression(expr, var_subs_scalar)
        assert np.array_equal(result, np.array([10.0]))

    def test_evaluate_standard_expression_evalf_failure(self, fidelity_calculator_instance):
        # Create a mock expression that doesn't have evalf or whose evalf might fail
        # or results in a non-float type if not handled.
        # For example, if substitution results in a symbolic form that cannot be evalf'd to float.
        expr = sp.Symbol('nosuchvar')
        var_subs = {} # No substitution for 'nosuchvar'
        # sympy.subs will leave it as Symbol('nosuchvar'). This doesn't have .evalf() directly for float.
        # The code has `float(result.evalf())` or `float(result)`.
        # A raw Symbol does not directly convert to float.

        # If result.subs(...) is still a Symbol, it won't have evalf() that returns a float directly.
        # float(sp.Symbol('x')) raises TypeError.
        # Let's mock `expr.subs` to return something that causes an issue.
        mock_expr = MagicMock(spec=sp.Expr)
        mock_expr.subs.return_value = sp.Symbol('unhandled') # This will cause float() to fail.

        result = fidelity_calculator_instance._evaluate_standard_expression(mock_expr, var_subs)
        assert np.array_equal(result, np.zeros(1))

        # Test with an expression that does have evalf but might return non-float compatible
        # This is harder to mock without deeper sympy interaction.
        # The current fallback to np.zeros(1) is the main thing to test.

        # Test when result is already a number (e.g. Python float, not Sympy type)
        mock_expr_returns_float = MagicMock(spec=sp.Expr)
        mock_expr_returns_float.subs.return_value = 5.0 # A Python float

        result_float = fidelity_calculator_instance._evaluate_standard_expression(mock_expr_returns_float, var_subs)
        assert np.array_equal(result_float, np.array([5.0]))

    # Test for _compute_position_differences, _compute_position_ratios, _compute_relative_positions
    def test_positional_computations(self, fidelity_calculator_instance, sample_test_data):
        seq_len = sample_test_data['sequence_length'] # Should be 4

        pos_diff = fidelity_calculator_instance._compute_position_differences(sample_test_data)
        assert pos_diff.shape == (seq_len*seq_len,)
        # Expected for seq_len=4:
        # pos_i = [[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]]
        # pos_j = [[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]]
        # diff  = [[0,-1,-2,-3],[1,0,-1,-2],[2,1,0,-1],[3,2,1,0]]
        expected_diff_flat = np.array([0,-1,-2,-3,1,0,-1,-2,2,1,0,-1,3,2,1,0], dtype=float).flatten()
        assert np.array_equal(pos_diff, expected_diff_flat)

        pos_ratio = fidelity_calculator_instance._compute_position_ratios(sample_test_data)
        assert pos_ratio.shape == (seq_len*seq_len,)
        # Positions are 1 to seq_len. For seq_len=4, positions are [1,2,3,4]
        # pos_i_ratio = [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
        # pos_j_ratio = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
        # ratio = [[1,1/2,1/3,1/4], [2,1,2/3,1/2], [3,3/2,1,3/4], [4,2,4/3,1]]
        expected_ratio_flat = np.array([
            1, 0.5, 1/3, 0.25,
            2, 1, 2/3, 0.5,
            3, 1.5, 1, 0.75,
            4, 2, 4/3, 1
        ], dtype=float).flatten()
        assert np.allclose(pos_ratio, expected_ratio_flat)


        rel_pos = fidelity_calculator_instance._compute_relative_positions(sample_test_data)
        assert rel_pos.shape == (seq_len*seq_len,)
        # rel_pos = abs(pos_i - pos_j) / seq_len. positions = [0,1,2,3]
        # abs_diff = [[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]]
        # expected_rel_pos_flat = abs_diff.flatten() / 4
        expected_rel_pos_flat = np.array([
            0,1,2,3, 1,0,1,2, 2,1,0,1, 3,2,1,0
        ], dtype=float).flatten() / seq_len
        assert np.allclose(rel_pos, expected_rel_pos_flat)
