"""
Tests for environments/ai_interpretability/model_hooks.py
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch

from environments.ai_interpretability.model_hooks import ModelHookManager, register_hooks_for_layers

# --- Helper Simple Model ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)
        self.identity = nn.Identity() # For testing tuple outputs if needed

    def forward(self, x):
        x1 = self.fc1(x)
        x_relu = self.relu(x1)
        x2 = self.fc2(x_relu)
        # To test tuple output from a layer if necessary for data_processor
        # For now, layers return single tensors.
        return x2

@pytest.fixture
def simple_model():
    return SimpleNet()

@pytest.fixture
def model_hook_manager(simple_model):
    return ModelHookManager(simple_model)

# --- Tests for ModelHookManager ---
class TestModelHookManager:

    def test_init(self, simple_model):
        manager = ModelHookManager(simple_model)
        assert manager.model == simple_model
        assert manager.active_hooks == []
        assert manager.captured_data == {}

    def test_create_hook_fn(self, model_hook_manager):
        # Test without data_processor
        hook_fn_no_proc = model_hook_manager._create_hook_fn("layer1", "data_key1")
        assert callable(hook_fn_no_proc)

        dummy_module = MagicMock()
        dummy_inputs = (torch.randn(1,1),)
        dummy_outputs = torch.randn(1,1)

        hook_fn_no_proc(dummy_module, dummy_inputs, dummy_outputs)
        assert "data_key1" in model_hook_manager.captured_data
        assert model_hook_manager.captured_data["data_key1"] is dummy_outputs # Raw output

        model_hook_manager.clear_captured_data()

        # Test with data_processor
        mock_processor = MagicMock(return_value="processed_data")
        hook_fn_with_proc = model_hook_manager._create_hook_fn("layer2", "data_key2", mock_processor)
        assert callable(hook_fn_with_proc)

        hook_fn_with_proc(dummy_module, dummy_inputs, dummy_outputs)
        mock_processor.assert_called_once_with(dummy_outputs)
        assert "data_key2" in model_hook_manager.captured_data
        assert model_hook_manager.captured_data["data_key2"] == "processed_data"

    def test_add_hook_successful(self, simple_model, model_hook_manager):
        # Mock named_modules to control which module is found
        mock_fc1_module = simple_model.fc1 # Get the actual module
        mock_fc1_module.register_forward_hook = MagicMock() # Mock its hook method

        # Ensure named_modules returns the module we want to hook
        with patch.object(simple_model, 'named_modules', return_value=[('fc1', mock_fc1_module)]):
            success = model_hook_manager.add_hook('fc1', 'fc1_out')

        assert success is True
        mock_fc1_module.register_forward_hook.assert_called_once()
        assert len(model_hook_manager.active_hooks) == 1
        # Check that the handle returned by register_forward_hook is stored (MagicMock returns MagicMock)
        assert model_hook_manager.active_hooks[0] == mock_fc1_module.register_forward_hook.return_value

    def test_add_hook_module_not_found(self, simple_model, model_hook_manager, capsys):
        with patch.object(simple_model, 'named_modules', return_value=[('some_other_layer', MagicMock())]):
            success = model_hook_manager.add_hook('non_existent_layer', 'data_key')

        assert success is False
        assert len(model_hook_manager.active_hooks) == 0
        captured = capsys.readouterr()
        assert "Warning: Module 'non_existent_layer' not found" in captured.out

    def test_clear_captured_data(self, model_hook_manager):
        model_hook_manager.captured_data = {'some_key': 'some_value'}
        model_hook_manager.clear_captured_data()
        assert model_hook_manager.captured_data == {}

    def test_remove_all_hooks(self, model_hook_manager):
        mock_handle1 = MagicMock()
        mock_handle2 = MagicMock()
        model_hook_manager.active_hooks = [mock_handle1, mock_handle2]

        model_hook_manager.remove_all_hooks()

        mock_handle1.remove.assert_called_once()
        mock_handle2.remove.assert_called_once()
        assert model_hook_manager.active_hooks == []

    def test_run_with_hooks(self, simple_model, model_hook_manager):
        # Add hooks for fc1 and relu
        # We need to use the actual modules from simple_model for hooks to register correctly
        model_hook_manager.add_hook('fc1', 'fc1_data', ModelHookManager.default_data_processor)
        model_hook_manager.add_hook('relu', 'relu_data', ModelHookManager.default_data_processor)

        input_tensor = torch.randn(3, 10) # Batch=3, Features=10

        # Patch model's eval to check it's called
        with patch.object(simple_model, 'eval') as mock_eval:
            # Patch model's forward to check it's called and to get its actual output
            # The actual forward pass will trigger the hooks.
            actual_model_output_tensor = simple_model(input_tensor) # Run manually to get expected output

            # Reset model state if necessary for a clean run_with_hooks call
            # (not strictly needed here as hooks are fresh and simple_model is stateless)
            simple_model.zero_grad()
            model_hook_manager.clear_captured_data() # Ensure it's clean before run_with_hooks

            # Re-add hooks as they might have been cleared if tests run in certain orders
            # or if simple_model was reused and hooks removed.
            # For this test, let's assume model_hook_manager is fresh or hooks are re-added.
            # The fixture provides a fresh manager, so hooks added above are fine.

            returned_model_output, captured = model_hook_manager.run_with_hooks(input_tensor)

            mock_eval.assert_called_once()
            assert torch.equal(returned_model_output, actual_model_output_tensor)

            assert 'fc1_data' in captured
            assert 'relu_data' in captured
            assert isinstance(captured['fc1_data'], np.ndarray)
            assert isinstance(captured['relu_data'], np.ndarray)

            # Check shapes (output of fc1 is (batch,5), relu is also (batch,5))
            assert captured['fc1_data'].shape == (3, 5)
            assert captured['relu_data'].shape == (3, 5)

            # Check that captured_data in the manager is also populated (it's a copy that's returned)
            assert 'fc1_data' in model_hook_manager.captured_data
            assert 'relu_data' in model_hook_manager.captured_data


    def test_default_data_processor(self):
        # Test with a tensor
        tensor_in = torch.randn(2,2)
        processed_tensor = ModelHookManager.default_data_processor(tensor_in)
        assert isinstance(processed_tensor, np.ndarray)
        assert np.array_equal(processed_tensor, tensor_in.cpu().numpy())

        # Test with a tuple of tensors
        tensor_tuple = (torch.randn(1,1), torch.randn(1,1))
        processed_tuple = ModelHookManager.default_data_processor(tensor_tuple)
        assert isinstance(processed_tuple, tuple)
        assert len(processed_tuple) == 2
        assert isinstance(processed_tuple[0], np.ndarray)
        assert isinstance(processed_tuple[1], np.ndarray)
        assert np.array_equal(processed_tuple[0], tensor_tuple[0].cpu().numpy())
        assert np.array_equal(processed_tuple[1], tensor_tuple[1].cpu().numpy())

        # Test with non-tensor data
        non_tensor_data = "string_data"
        processed_other = ModelHookManager.default_data_processor(non_tensor_data)
        assert processed_other == non_tensor_data

    def test_get_module_by_name(self, simple_model):
        # Test finding existing modules
        fc1_mod = ModelHookManager.get_module_by_name(simple_model, 'fc1')
        assert fc1_mod is simple_model.fc1

        relu_mod = ModelHookManager.get_module_by_name(simple_model, 'relu')
        assert relu_mod is simple_model.relu

        # Test finding a nested module if we had one (e.g., 'block.0.conv')
        # SimpleNet doesn't have deeply nested ones by default.

        # Test non-existent module
        non_existent = ModelHookManager.get_module_by_name(simple_model, 'does_not_exist')
        assert non_existent is None


# --- Tests for register_hooks_for_layers function ---

@patch('environments.ai_interpretability.model_hooks.ModelHookManager.get_module_by_name')
def test_register_hooks_for_layers_successful(mock_get_module, simple_model):
    captured_dict = {}
    layer_names_to_hook = ['fc1', 'relu']

    # Mock modules that get_module_by_name will return
    mock_fc1 = simple_model.fc1
    mock_relu = simple_model.relu
    mock_fc1.register_forward_hook = MagicMock(return_value="handle_fc1")
    mock_relu.register_forward_hook = MagicMock(return_value="handle_relu")

    def get_module_side_effect(model, name):
        if name == 'fc1': return mock_fc1
        if name == 'relu': return mock_relu
        return None
    mock_get_module.side_effect = get_module_side_effect

    handles = register_hooks_for_layers(simple_model, layer_names_to_hook, captured_dict)

    assert len(handles) == 2
    assert "handle_fc1" in handles
    assert "handle_relu" in handles

    mock_get_module.assert_any_call(simple_model, 'fc1')
    mock_get_module.assert_any_call(simple_model, 'relu')
    mock_fc1.register_forward_hook.assert_called_once()
    mock_relu.register_forward_hook.assert_called_once()

    # Test that the hook function itself works (captures data)
    # Get the hook function passed to register_forward_hook for fc1
    fc1_hook_fn = mock_fc1.register_forward_hook.call_args[0][0]
    dummy_fc1_out = torch.randn(1,5)
    fc1_hook_fn(mock_fc1, (torch.randn(1,10),), dummy_fc1_out, lname='fc1') # lname must match

    assert 'fc1' in captured_dict
    assert isinstance(captured_dict['fc1'], np.ndarray)
    assert np.array_equal(captured_dict['fc1'], dummy_fc1_out.cpu().numpy())


@patch('environments.ai_interpretability.model_hooks.ModelHookManager.get_module_by_name')
def test_register_hooks_for_layers_module_not_found(mock_get_module, simple_model, capsys):
    captured_dict = {}
    mock_get_module.return_value = None # Simulate module not found

    handles = register_hooks_for_layers(simple_model, ['non_existent'], captured_dict)

    assert len(handles) == 0
    mock_get_module.assert_called_once_with(simple_model, 'non_existent')
    captured = capsys.readouterr()
    assert "Warning (register_hooks_for_layers): Layer 'non_existent' not found" in captured.out
    assert captured_dict == {}
