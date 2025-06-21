"""
Tests for ai_interpretability/interpreters/base_interpreter.py
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, call

# Import the class to be tested
from ai_interpretability.interpreters.base_interpreter import AILawDiscovery

# --- Mock Dependencies ---
# These will be patched during test execution.

# Mock for JanusAI.core.grammar.base_grammar.Expression
MockExpression = MagicMock(name="MockExpression")
MockExpression.__instancecheck__ = lambda _, instance: isinstance(instance, MagicMock)


# Mock for JanusAI.ai_interpretability.grammars.neural_grammar.NeuralGrammar
MockNeuralGrammar = MagicMock(name="MockNeuralGrammar")

# Mock for JanusAI.environments.ai_interpretability.neural_net_env.AIBehaviorData
MockAIBehaviorData = MagicMock(name="MockAIBehaviorData")

# Mock for JanusAI.environments.ai_interpretability.neural_net_env.AIInterpretabilityEnv
MockAIInterpretabilityEnv = MagicMock(name="MockAIInterpretabilityEnv")

# Mock for JanusAI.environments.ai_interpretability.neural_net_env.LocalInterpretabilityEnv
MockLocalInterpretabilityEnv = MagicMock(name="MockLocalInterpretabilityEnv")

# Mock for JanusAI.ml.networks.hypothesis_net.HypothesisNet
MockHypothesisNet = MagicMock(name="MockHypothesisNet")

# Mock for JanusAI.ml.networks.hypothesis_net.PPOTrainer
MockPPOTrainer = MagicMock(name="MockPPOTrainer")


@pytest.fixture
def mock_ai_model_nn():
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock()
    model.named_modules = MagicMock(return_value=[("layer1", MagicMock(spec=nn.Linear))]) # Example layer

    # Define a side effect for forward pass
    def forward_side_effect(input_tensor):
        # Simulate output based on input shape, e.g., (batch_size, output_features)
        return torch.rand(input_tensor.shape[0], 5)
    model.forward = MagicMock(side_effect=forward_side_effect)
    model.__call__ = model.forward # Make it callable
    return model

# Patch all dependencies for the duration of the tests in this class
@patch('ai_interpretability.interpreters.base_interpreter.Expression', MockExpression)
@patch('ai_interpretability.interpreters.base_interpreter.NeuralGrammar', MockNeuralGrammar)
@patch('ai_interpretability.interpreters.base_interpreter.AIBehaviorData', MockAIBehaviorData)
@patch('ai_interpretability.interpreters.base_interpreter.AIInterpretabilityEnv', MockAIInterpretabilityEnv)
@patch('ai_interpretability.interpreters.base_interpreter.LocalInterpretabilityEnv', MockLocalInterpretabilityEnv)
@patch('ai_interpretability.interpreters.base_interpreter.HypothesisNet', MockHypothesisNet)
@patch('ai_interpretability.interpreters.base_interpreter.PPOTrainer', MockPPOTrainer)
class TestAILawDiscovery:

    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        # Reset mocks before each test to clear call counts, etc.
        MockExpression.reset_mock()
        MockNeuralGrammar.reset_mock()
        MockAIBehaviorData.reset_mock()
        MockAIInterpretabilityEnv.reset_mock()
        MockLocalInterpretabilityEnv.reset_mock()
        MockHypothesisNet.reset_mock()
        MockPPOTrainer.reset_mock()


    def test_init(self, mock_ai_model_nn):
        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn, model_type='test_type')
        assert law_discovery.ai_model == mock_ai_model_nn
        assert law_discovery.model_type == 'test_type'
        MockNeuralGrammar.assert_called_once() # Default grammar initialization
        assert law_discovery.grammar == MockNeuralGrammar.return_value
        assert law_discovery.discovered_laws == []

    def test_collect_behavior_data_basic(self, mock_ai_model_nn):
        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn)
        input_data = np.random.rand(10, 3) # 10 samples, 3 features

        # Expected output from the mocked model's forward pass
        expected_model_output = mock_ai_model_nn(torch.FloatTensor(input_data)).cpu().numpy()

        behavior_data_instance = MockAIBehaviorData.return_value

        result = law_discovery._collect_behavior_data(input_data)

        mock_ai_model_nn.eval.assert_called_once()
        mock_ai_model_nn.forward.assert_called_once() # or __call__

        # Check arguments to AIBehaviorData constructor
        MockAIBehaviorData.assert_called_once()
        args, kwargs = MockAIBehaviorData.call_args
        assert np.array_equal(kwargs['inputs'], input_data)
        assert np.array_equal(kwargs['outputs'], expected_model_output)
        assert kwargs['intermediate_activations'] is None
        assert kwargs['attention_weights'] is None
        assert result == behavior_data_instance


    def test_collect_behavior_data_with_activations(self, mock_ai_model_nn):
        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn)
        input_data = np.random.rand(5, 2)
        capture_layers = ['layer1']

        # Mock the layer and its hook registration
        mock_layer_module = MagicMock(spec=nn.Module)
        mock_hook = MagicMock()
        mock_layer_module.register_forward_hook = MagicMock(return_value=mock_hook)
        mock_ai_model_nn.named_modules = MagicMock(return_value=[('layer1', mock_layer_module), ('layer2', MagicMock())])

        # Simulate hook behavior: capture output
        captured_act_data = {}
        def mock_hook_fn(mod, inp, outp, lname='layer1'): # Need to match lambda signature
            captured_act_data.update({lname: outp.detach().cpu().numpy()})

        # Simulate the output that the hook will capture
        simulated_layer1_output = torch.rand(input_data.shape[0], 4) # Batch size, layer1_features

        # Side effect for register_forward_hook: call our mock_hook_fn
        # This is tricky because the hook is called internally by PyTorch.
        # We'll manually populate captured_act_data as if the hook ran.

        # Instead of complex hook simulation, let's assume the hook correctly populates
        # intermediate_acts_data inside _collect_behavior_data.
        # We can verify register_forward_hook was called and then check AIBehaviorData args.

        # To make the test simpler, let's patch the lambda function inside _collect_behavior_data,
        # or ensure the named_modules and hook mechanism works as expected by the code.
        # The current structure uses a lambda, which is harder to patch directly by name.
        # We can assert that register_forward_hook was called.
        # For the actual data, we can check what was passed to AIBehaviorData.

        # Let's refine the mock_ai_model_nn.forward to also simulate activations being available
        # if a hook were "really" registered. This is indirect.
        # A better way is to ensure the hook is called and then check the outcome.

        # For this test, we will assume the hook mechanism is called.
        # The actual data captured by the hook is an internal detail of the PyTorch hook mechanism.
        # We can verify that `register_forward_hook` was called and `hook.remove()` was called.

        # The lambda directly updates `intermediate_acts_data`.
        # We'll check the content of `intermediate_acts_data` passed to `AIBehaviorData`.

        # To make the test work, we need the lambda to be executed.
        # This happens when the layer's forward is called, which happens when model.forward is called.
        # We need `mock_layer_module` to be the one whose `register_forward_hook` is called.

        # Let's assume the hook is correctly registered and fires.
        # The output of the layer 'layer1' needs to be defined.
        # We can achieve this by making the main model's forward pass also return
        # something that we can say is the activation (this is a bit of a hack for testing).
        # A more robust way is to have the hook actually execute and populate the dict.

        # Let's simplify: assume the hook lambda works.
        # We'll check if `register_forward_hook` was called with a callable.
        # And then we'll provide the data that *would* have been captured.

        # Patch the lambda? No, let's make the hook store data in a place we can inspect.
        # The lambda updates `intermediate_acts_data` in `_collect_behavior_data`'s scope.
        # We can't easily access that dict from here.
        # So, we check the arguments to AIBehaviorData.

        # Let's mock the actual output of the layer that the hook would capture.
        # The hook function in the code is:
        # lambda mod, inp, outp, lname=layer_name: intermediate_acts_data.update({lname: outp.detach().cpu().numpy()})
        # So, `outp` is key.
        # We need `mock_layer_module` to produce a specific `outp` when it's "forwarded".
        # This is getting too complex. The easiest is to check `AIBehaviorData`'s args.

        # Modify the model's forward pass to "simulate" that an activation was captured by a hook.
        # This is indirect. A better approach is to make the test setup such that the actual
        # hook mechanism can be tested.
        # For now, let's assume the hook captures *something* and check AIBehaviorData.

        # We'll have to rely on the internal `intermediate_acts_data` dictionary being populated by the hook.
        # The test will then check the arguments to `AIBehaviorData`.
        # To ensure the hook has data to capture, the layer `layer1` must produce output.
        # This output is `outp` in the hook lambda.
        # This `outp` comes from `module.forward()`.

        # Let the test pass if AIBehaviorData is called with non-empty activations
        # when capture_activations is set.

        # We need to ensure the hook is called. The hook is attached to `mock_layer_module`.
        # `mock_layer_module.forward()` must be called. This happens during `self.ai_model(inputs_tensor)`.
        # This implies `mock_ai_model_nn` should internally call `mock_layer_module.forward()`.
        # This is usually true for `nn.Sequential` or if `mock_layer_module` is a child.

        # Let's assume the hook registration and firing works if `named_modules` finds the layer.
        # The critical part is that `intermediate_acts_data` gets populated.

        # For the test, we will assume the hook is set up and works.
        # The test will verify `register_forward_hook` is called, and `AIBehaviorData` gets some activations.
        # To make `intermediate_acts_data` populated, the hook lambda needs `outp`.
        # We can make `mock_layer_module.forward` (if it were called) return something.
        # However, `nn.Module.register_forward_hook` calls the hook with the actual output.

        # Simplification: Assume the hook system works.
        # We need to provide the data that the hook *would* capture.
        # This means when `AIBehaviorData` is called, its `intermediate_activations` arg should be populated.
        # This requires the `intermediate_acts_data` dict in `_collect_behavior_data` to be filled.
        # This dict is filled by the lambda hook. The lambda needs `outp`.
        # `outp` is the output of the hooked module.

        # Let's ensure the hook is registered and removed.
        # And that AIBehaviorData is called with some activation data.
        # To make this happen, we need `outp` in the lambda to be defined.
        # `outp` is the output of `module` (which is `mock_layer_module`).
        # So, when `self.ai_model(inputs_tensor)` is called, and if `mock_layer_module` is part of it,
        # its forward pass result will be `outp`.

        # Let's mock the `outp` that the hook receives.
        # This means we need to control what the hook lambda receives as `outp`.
        # This is hard because the lambda is called by PyTorch internals.

        # Alternative: Patch `intermediate_acts_data.update` if we could get a handle to the dict.
        # Or, more simply, just check that `register_forward_hook` was called, and then
        # check that `AIBehaviorData` was called with `intermediate_activations` not None.
        # The actual values in `intermediate_activations` are hard to control precisely here
        # without making `mock_ai_model_nn` a very complex mock that simulates internal layer calls.

        # Let's assume the hook does its job.
        # We verify the hook was set and removed, and AIBehaviorData got *some* activation data.
        # The test will make `mock_layer_module.register_forward_hook` store the passed hook function.
        # Then, we can manually call this hook function with a dummy output to populate
        # the `intermediate_acts_data` dictionary if we could access it.
        # This is too complex.

        # Simplest path:
        # 1. Check `register_forward_hook` is called on the correct module.
        # 2. Check `hook.remove()` is called.
        # 3. Check `AIBehaviorData` is called with `intermediate_activations` being a dict (possibly empty if layer not found).
        # If layer is found, it should contain data.

        # To ensure `intermediate_acts_data` is populated, the hook lambda must execute.
        # The lambda is `lambda mod, inp, outp, lname=layer_name: intermediate_acts_data.update({lname: outp.detach().cpu().numpy()})`
        # We need to make `outp` have a value. `outp` is the output of `mock_layer_module`.
        # So, if `mock_layer_module` is "forwarded" during `mock_ai_model_nn()`, it will work.
        # Let's assume `mock_ai_model_nn` is like `nn.Sequential(mock_layer_module)`.

        # Redefine mock_ai_model_nn to include the layer we want to hook
        mock_layer1 = MagicMock(spec=nn.Linear)
        mock_hook_instance = MagicMock()
        mock_layer1.register_forward_hook = MagicMock(return_value=mock_hook_instance)

        # Simulate layer1's forward pass result (this will be `outp` for the hook)
        simulated_layer1_tensor_output = torch.rand(input_data.shape[0], 7)
        mock_layer1.forward = MagicMock(return_value=simulated_layer1_tensor_output)
        mock_layer1.__call__ = mock_layer1.forward

        # Make the main model contain this layer and call it
        main_model_with_layer = MagicMock(spec=nn.Module)
        main_model_with_layer.eval = MagicMock()
        main_model_with_layer.named_modules = MagicMock(return_value=[('layer1', mock_layer1)])

        final_output_tensor = torch.rand(input_data.shape[0], 3) # Final output of main model
        def main_model_forward_with_layer_call(input_tensor):
            _ = mock_layer1(input_tensor) # Simulate layer1 being called
            return final_output_tensor
        main_model_with_layer.forward = MagicMock(side_effect=main_model_forward_with_layer_call)
        main_model_with_layer.__call__ = main_model_with_layer.forward

        law_discovery_acts = AILawDiscovery(ai_model=main_model_with_layer)
        law_discovery_acts._collect_behavior_data(input_data, capture_activations=capture_layers)

        mock_layer1.register_forward_hook.assert_called_once()
        # The hook function is a lambda, so we can check it's a callable
        assert callable(mock_layer1.register_forward_hook.call_args[0][0])

        MockAIBehaviorData.assert_called_once()
        _, kwargs_behavior = MockAIBehaviorData.call_args
        assert 'intermediate_activations' in kwargs_behavior
        assert kwargs_behavior['intermediate_activations'] is not None
        assert 'layer1' in kwargs_behavior['intermediate_activations']
        assert np.array_equal(kwargs_behavior['intermediate_activations']['layer1'], simulated_layer1_tensor_output.detach().cpu().numpy())

        mock_hook_instance.remove.assert_called_once()


    def test_collect_behavior_data_activation_layer_not_found(self, mock_ai_model_nn, capsys):
        # Make named_modules return something that doesn't include 'non_existent_layer'
        mock_ai_model_nn.named_modules = MagicMock(return_value=[("layer1", MagicMock())])
        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn)
        input_data = np.random.rand(2,2)

        law_discovery._collect_behavior_data(input_data, capture_activations=['non_existent_layer'])

        captured = capsys.readouterr()
        assert "Warning: Layer non_existent_layer not found" in captured.out

        MockAIBehaviorData.assert_called_once()
        _, kwargs_behavior = MockAIBehaviorData.call_args
        # Activations should be what was collected (empty dict if layer not found and it was the only one)
        # or None if the dict remains empty. The code does: intermediate_acts_data if intermediate_acts_data else None
        assert kwargs_behavior['intermediate_activations'] is None


    def test_collect_behavior_data_model_output_tuple(self, mock_ai_model_nn):
        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn)
        input_data = np.random.rand(3,3)

        # Make model forward return a tuple
        main_output = torch.rand(input_data.shape[0], 2)
        extra_output = torch.rand(input_data.shape[0], 1)
        mock_ai_model_nn.forward = MagicMock(return_value=(main_output, extra_output))
        mock_ai_model_nn.__call__ = mock_ai_model_nn.forward # Ensure callable uses this new forward

        law_discovery._collect_behavior_data(input_data)

        MockAIBehaviorData.assert_called_once()
        _, kwargs = MockAIBehaviorData.call_args
        assert np.array_equal(kwargs['outputs'], main_output.cpu().numpy())


    @patch.object(AILawDiscovery, '_collect_behavior_data')
    @patch.object(AILawDiscovery, '_extract_laws_from_env')
    def test_discover_global_laws(self, mock_extract_laws, mock_collect_data, mock_ai_model_nn):
        # Setup mocks for this specific test
        mock_collect_data.return_value = MockAIBehaviorData() # Instance

        mock_env_instance = MagicMock()
        mock_env_instance.observation_space = MagicMock()
        mock_env_instance.observation_space.shape = [10] # Example obs dim
        mock_env_instance.action_space = MagicMock()
        mock_env_instance.action_space.n = 5 # Example act dim
        MockAIInterpretabilityEnv.return_value = mock_env_instance

        mock_trainer_instance = MagicMock()
        MockPPOTrainer.return_value = mock_trainer_instance

        mock_discovered_law = MockExpression()
        mock_extract_laws.return_value = [mock_discovered_law]

        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn)
        input_data = np.random.rand(10,3)

        # Test with default env_kwargs
        returned_laws = law_discovery.discover_global_laws(input_data, max_complexity=15, n_epochs=50)

        mock_collect_data.assert_called_once_with(input_data, capture_activations=None)
        MockAIInterpretabilityEnv.assert_called_once_with(
            ai_model=mock_ai_model_nn,
            grammar=law_discovery.grammar, # This is MockNeuralGrammar.return_value
            behavior_data=MockAIBehaviorData.return_value,
            interpretation_mode='global',
            max_complexity=15
        )
        MockHypothesisNet.assert_called_once_with(
            obs_dim=10, act_dim=5, grammar=law_discovery.grammar
        )
        MockPPOTrainer.assert_called_once_with(
            MockHypothesisNet.return_value, mock_env_instance
        )
        mock_trainer_instance.train.assert_called_once_with(total_timesteps=50 * 1000) # Default steps_per_epoch
        mock_extract_laws.assert_called_once_with(mock_env_instance, mock_trainer_instance)

        assert returned_laws == [mock_discovered_law]
        assert law_discovery.discovered_laws == [mock_discovered_law]

        # Test with custom env_kwargs
        mock_collect_data.reset_mock()
        MockAIInterpretabilityEnv.reset_mock()
        MockHypothesisNet.reset_mock()
        MockPPOTrainer.reset_mock()
        mock_trainer_instance.reset_mock()
        mock_extract_laws.reset_mock()
        law_discovery.discovered_laws = []

        custom_env_kwargs = {
            "capture_activation_layers": ["l1"],
            "trainer_config": {"lr": 0.01},
            "steps_per_epoch": 200,
            "some_other_env_param": True
        }
        law_discovery.discover_global_laws(input_data, env_kwargs=custom_env_kwargs)

        mock_collect_data.assert_called_once_with(input_data, capture_activations=["l1"])
        MockAIInterpretabilityEnv.assert_called_once_with(
            ai_model=mock_ai_model_nn,
            grammar=law_discovery.grammar,
            behavior_data=MockAIBehaviorData.return_value,
            interpretation_mode='global',
            max_complexity=10, # Default if not specified in discover_global_laws call
            capture_activation_layers=["l1"], # Passed through
            trainer_config={"lr": 0.01},      # Passed through
            steps_per_epoch=200,              # Passed through
            some_other_env_param=True         # Passed through
        )
        MockPPOTrainer.assert_called_once_with(
            MockHypothesisNet.return_value, mock_env_instance, lr=0.01 # trainer_config is unrolled
        )
        mock_trainer_instance.train.assert_called_once_with(total_timesteps=100 * 200) # n_epochs * steps_per_epoch


    def test_discover_global_laws_env_space_missing(self, mock_ai_model_nn, capsys):
        # Simulate environment not having observation_space
        mock_env_no_space = MagicMock()
        del mock_env_no_space.observation_space # Remove attribute
        MockAIInterpretabilityEnv.return_value = mock_env_no_space

        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn)
        with patch.object(AILawDiscovery, '_collect_behavior_data', return_value=MockAIBehaviorData()):
            laws = law_discovery.discover_global_laws(np.random.rand(1,1))

        assert laws == []
        captured = capsys.readouterr()
        assert "Environment does not have observation_space or action_space defined" in captured.out


    @patch.object(AILawDiscovery, '_collect_behavior_data')
    @patch.object(AILawDiscovery, '_quick_discovery')
    def test_discover_neuron_roles(self, mock_quick_discovery, mock_collect_data, mock_ai_model_nn):
        # Setup: behavior_data with activations for 'layer1' which has 2 neurons
        mock_activations = np.random.rand(10, 2) # 10 samples, 2 neurons
        behavior_with_acts_instance = MockAIBehaviorData()
        behavior_with_acts_instance.intermediate_activations = {'layer1': mock_activations}
        mock_collect_data.return_value = behavior_with_acts_instance

        # _quick_discovery will be called per neuron, make it return different expressions
        mock_expr_neuron0 = MockExpression(name="neuron0_expr")
        mock_expr_neuron1 = MockExpression(name="neuron1_expr")
        mock_quick_discovery.side_effect = [[mock_expr_neuron0], [mock_expr_neuron1]]

        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn)
        input_data = np.random.rand(10,3)
        layer_name = 'layer1'

        roles = law_discovery.discover_neuron_roles(layer_name, input_data, max_complexity_neuron=7, n_steps_neuron=500)

        mock_collect_data.assert_called_once_with(input_data, capture_activations=[layer_name])

        # Check AIInterpretabilityEnv calls (2 times, one for each neuron)
        assert MockAIInterpretabilityEnv.call_count == 2
        env_call_args_list = MockAIInterpretabilityEnv.call_args_list

        # Neuron 0
        args_n0, kwargs_n0 = env_call_args_list[0]
        assert kwargs_n0['ai_model'] == mock_ai_model_nn
        assert kwargs_n0['grammar'] == law_discovery.grammar
        assert kwargs_n0['max_complexity'] == 7
        # Check behavior_data for neuron 0
        bd_n0 = kwargs_n0['behavior_data']
        MockAIBehaviorData.assert_any_call(inputs=input_data, outputs=mock_activations[:, 0:1])
        assert bd_n0.inputs is input_data # This check is tricky due to how MockAIBehaviorData is used
        assert bd_n0.outputs is mock_activations[:, 0:1] # Check that the instance was created with this

        # Neuron 1
        args_n1, kwargs_n1 = env_call_args_list[1]
        assert kwargs_n1['max_complexity'] == 7
        bd_n1 = kwargs_n1['behavior_data']
        MockAIBehaviorData.assert_any_call(inputs=input_data, outputs=mock_activations[:, 1:2])
        assert bd_n1.outputs is mock_activations[:, 1:2]

        # Check _quick_discovery calls
        assert mock_quick_discovery.call_count == 2
        quick_discovery_calls = mock_quick_discovery.call_args_list
        assert quick_discovery_calls[0] == call(MockAIInterpretabilityEnv.return_value, n_steps=500)
        assert quick_discovery_calls[1] == call(MockAIInterpretabilityEnv.return_value, n_steps=500)

        assert roles == {0: mock_expr_neuron0, 1: mock_expr_neuron1}

    @patch.object(AILawDiscovery, '_collect_behavior_data')
    def test_discover_neuron_roles_no_activations(self, mock_collect_data, mock_ai_model_nn, capsys):
        # Simulate _collect_behavior_data returning no activations for the layer
        behavior_no_acts_instance = MockAIBehaviorData()
        behavior_no_acts_instance.intermediate_activations = {} # Empty dict
        mock_collect_data.return_value = behavior_no_acts_instance

        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn)
        roles = law_discovery.discover_neuron_roles("missing_layer", np.random.rand(1,1))

        assert roles == {}
        captured = capsys.readouterr()
        assert "Could not retrieve activations for layer missing_layer" in captured.out


    @patch.object(AILawDiscovery, '_quick_discovery')
    def test_explain_decision(self, mock_quick_discovery, mock_ai_model_nn):
        mock_expr_local = MockExpression(name="local_expr")
        mock_quick_discovery.return_value = [mock_expr_local] # Assume it returns a list

        law_discovery = AILawDiscovery(ai_model=mock_ai_model_nn)
        sample_1d = np.random.rand(5)
        sample_2d = sample_1d.reshape(1, -1)

        explanation = law_discovery.explain_decision(sample_1d, neighborhood_size=0.05, n_steps_local=300)

        MockLocalInterpretabilityEnv.assert_called_once()
        args, kwargs = MockLocalInterpretabilityEnv.call_args
        assert kwargs['ai_model'] == mock_ai_model_nn
        assert kwargs['grammar'] == law_discovery.grammar
        assert kwargs['behavior_data'] is None
        assert np.array_equal(kwargs['anchor_input'], sample_2d) # Should be reshaped
        assert kwargs['neighborhood_size'] == 0.05

        mock_quick_discovery.assert_called_once_with(MockLocalInterpretabilityEnv.return_value, n_steps=300)
        assert explanation == mock_expr_local

        # Test with already 2D input
        MockLocalInterpretabilityEnv.reset_mock()
        mock_quick_discovery.reset_mock()
        law_discovery.explain_decision(sample_2d)
        assert np.array_equal(MockLocalInterpretabilityEnv.call_args[1]['anchor_input'], sample_2d)


    def test_quick_discovery(self, mock_ai_model_nn): # mock_ai_model_nn is not directly used but sets up patches
        mock_env = MagicMock(spec=MockAIInterpretabilityEnv) # Use the patched mock type
        mock_env.action_space.sample = MagicMock(return_value=0) # Dummy action

        # Simulate env.reset() and env.step() behavior
        # Episode 1: expr1, reward 10
        # Episode 2: expr2, reward 5
        # Episode 3: expr3, reward 12
        mock_expr1 = MockExpression(name="expr1")
        mock_expr2 = MockExpression(name="expr2")
        mock_expr3 = MockExpression(name="expr3")

        # Define side effects for env.reset() and env.step()
        # reset -> (obs, info_reset)
        # step -> (obs, reward, terminated, truncated, info_step)

        # Side effect for reset:
        reset_infos = [{}, {}, {}] # Dummy infos from reset
        reset_side_effect = [(np.array([0.1]), reset_infos[i]) for i in range(3)]

        # Side effect for step (simplified: one step per episode for this test)
        # (obs, reward, terminated, truncated, info_with_expr_and_reward)
        step_side_effect = [
            (np.array([0.2]), 10, True, False, {'expression_obj': mock_expr1, 'reward': 10}),
            (np.array([0.3]), 5, True, False, {'expression_obj': mock_expr2, 'reward': 5}),
            (np.array([0.4]), 12, True, False, {'expression_obj': mock_expr3, 'reward': 12}),
        ]

        mock_env.reset = MagicMock(side_effect=reset_side_effect)
        mock_env.step = MagicMock(side_effect=step_side_effect)

        law_discovery = AILawDiscovery(ai_model=MagicMock()) # Dummy model for init

        # Test with n_steps = 3, so 3 episodes
        best_laws = law_discovery._quick_discovery(mock_env, n_steps=3)

        assert mock_env.reset.call_count == 3
        assert mock_env.step.call_count == 3
        assert mock_env.action_space.sample.call_count == 3

        # Expected order: expr3 (12), expr1 (10), expr2 (5)
        assert best_laws == [mock_expr3, mock_expr1, mock_expr2]

        # Test keeping top-k (e.g. if many expressions found, only top 10 are kept)
        # This is implicitly tested by the sorting and popping logic.

    def test_extract_laws_from_env_with_trainer_method(self, mock_ai_model_nn):
        mock_env = MockAIInterpretabilityEnv()
        mock_trainer = MockPPOTrainer()

        expected_laws = [MockExpression(name="law_from_trainer")]
        mock_trainer.get_best_expressions = MagicMock(return_value=expected_laws)

        law_discovery = AILawDiscovery(ai_model=MagicMock())
        laws = law_discovery._extract_laws_from_env(mock_env, mock_trainer)

        mock_trainer.get_best_expressions.assert_called_once()
        assert laws == expected_laws

    @patch.object(AILawDiscovery, '_quick_discovery')
    def test_extract_laws_from_env_fallback_to_quick_discovery(self, mock_qd, mock_ai_model_nn, capsys):
        mock_env = MockAIInterpretabilityEnv()
        mock_trainer_no_method = MagicMock()
        del mock_trainer_no_method.get_best_expressions # Ensure method is missing

        expected_laws_qd = [MockExpression(name="law_from_qd")]
        mock_qd.return_value = expected_laws_qd

        # Setup action_space for the fallback _quick_discovery call
        mock_env.action_space = MagicMock()
        mock_env.action_space.n = 5


        law_discovery = AILawDiscovery(ai_model=MagicMock())
        laws = law_discovery._extract_laws_from_env(mock_env, mock_trainer_no_method)

        captured = capsys.readouterr()
        assert "Warning: PPOTrainer does not have get_best_expressions" in captured.out
        mock_qd.assert_called_once_with(mock_env, n_steps=max(100, 5 * 5)) # n * 5
        assert laws == expected_laws_qd

    def test_extract_laws_from_env_no_trainer(self, mock_ai_model_nn):
        # This case should also fall back to _quick_discovery
        mock_env = MockAIInterpretabilityEnv()
        mock_env.action_space = MagicMock() # Needed for _quick_discovery
        mock_env.action_space.n = 3

        law_discovery = AILawDiscovery(ai_model=MagicMock())

        expected_laws_qd = [MockExpression(name="law_from_qd_no_trainer")]
        with patch.object(AILawDiscovery, '_quick_discovery', return_value=expected_laws_qd) as mock_qd_no_trainer:
            laws = law_discovery._extract_laws_from_env(mock_env, None) # No trainer passed

            mock_qd_no_trainer.assert_called_once_with(mock_env, n_steps=max(100, 3*5))
            assert laws == expected_laws_qd
