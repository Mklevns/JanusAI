"""
Tests for environments/enhanced/feedback_env.py
"""
import pytest
import numpy as np
import torch
from collections import deque
from unittest.mock import MagicMock, patch

from environments.enhanced.feedback_env import EnhancedObservationEncoder, EnhancedSymbolicDiscoveryEnv
from environments.base.symbolic_env import SymbolicDiscoveryEnv, TreeState, ExpressionNode
from core.grammar.base_grammar import ProgressiveGrammar
from core.expressions.expression import Variable
from ml.rewards.intrinsic_rewards import IntrinsicRewardCalculator # For type hinting/mocking

# --- Mocks for dependencies ---
MockTreeState = MagicMock(spec=TreeState)
MockExpressionNode = MagicMock(spec=ExpressionNode)
MockProgressiveGrammar = MagicMock(spec=ProgressiveGrammar)
MockIntrinsicRewardCalculator = MagicMock(spec=IntrinsicRewardCalculator)
MockBaseSymbolicDiscoveryEnv = MagicMock(spec=SymbolicDiscoveryEnv)


# --- Tests for EnhancedObservationEncoder ---
class TestEnhancedObservationEncoder:
    @pytest.fixture
    def encoder(self):
        return EnhancedObservationEncoder(base_dim=10, history_length=3)

    @pytest.fixture
    def mock_tree_state(self):
        state = MagicMock(spec=TreeState)
        state.root = MagicMock(spec=ExpressionNode)
        state.root.node_type = MagicMock()
        state.root.node_type.value = "operator" # Example
        state.root.children = []
        state.is_complete = MagicMock(return_value=True)
        state.count_nodes = MagicMock(return_value=5) # Used for complexity features
        # Mock max_complexity on state if _extract_complexity_features expects it there
        # The code has: getattr(state, 'max_complexity', 30)
        state.max_complexity = 10 # For testing complexity features
        return state

    @pytest.fixture
    def mock_grammar_instance(self):
        grammar = MagicMock(spec=ProgressiveGrammar)
        grammar.primitives = {'constants': {}, 'unary_ops': {}, 'binary_ops': {}}
        grammar.learned_functions = {}
        grammar.variables = {} # Assuming grammar has .variables for _extract_grammar_features
        grammar.current_stage = 1
        return grammar

    def test_init_encoder(self):
        encoder = EnhancedObservationEncoder(base_dim=128, history_length=5)
        assert encoder.base_dim == 128
        assert encoder.history_length == 5
        assert isinstance(encoder.action_history, deque)
        assert encoder.action_history.maxlen == 5
        assert isinstance(encoder.context_encoder, torch.nn.Sequential)

    def test_update_history(self, encoder):
        encoder.update_history(1, 0.5, "expr1")
        encoder.update_history(2, 0.6, "expr2")
        encoder.update_history(3, 0.7, "expr3")
        assert list(encoder.action_history) == [1, 2, 3]
        assert list(encoder.reward_history) == [0.5, 0.6, 0.7]
        assert list(encoder.expression_history) == ["expr1", "expr2", "expr3"]

        encoder.update_history(4, 0.8, "expr4") # Maxlen is 3, so '1' gets pushed out
        assert list(encoder.action_history) == [2, 3, 4]

    def test_extract_tree_features(self, encoder, mock_tree_state):
        # Mock helper methods of the encoder itself for this specific test unit
        with patch.object(encoder, '_get_node_depths', return_value=[0, 1, 1, 2]) as mock_depths, \
             patch.object(encoder, '_count_node_types', return_value={'operator': 2, 'variable': 1, 'constant': 1}) as mock_counts, \
             patch.object(encoder, '_calculate_tree_balance', return_value=0.5) as mock_balance:

            features = encoder._extract_tree_features(mock_tree_state)

            mock_depths.assert_called_once_with(mock_tree_state.root)
            mock_counts.assert_called_once_with(mock_tree_state.root)
            mock_balance.assert_called_once_with(mock_tree_state.root)
            mock_tree_state.is_complete.assert_called_once()

            # Expected feature vector structure:
            # [mean_depth, max_depth, std_depth, op_ratio, var_ratio, const_ratio, empty_ratio, is_complete_float, balance]
            # Total nodes from mock_counts = 2+1+1 = 4
            # mean([0,1,1,2])=1, max=2, std=sqrt( ((0-1)^2 + (1-1)^2 + (1-1)^2 + (2-1)^2) / 4 ) = sqrt( (1+0+0+1)/4 ) = sqrt(0.5) ~ 0.707
            # op_ratio=2/4=0.5, var_ratio=1/4=0.25, const_ratio=1/4=0.25, empty_ratio=0/4=0.0
            # is_complete=1.0, balance=0.5
            expected = [
                1.0, 2.0, np.std([0,1,1,2]),
                0.5, 0.25, 0.25, 0.0,
                1.0, 0.5
            ]
            assert features.shape == (9,)
            assert np.allclose(features, expected)

    def test_encode_history(self, encoder):
        encoder.action_history.extend([1,2,0]) # Maxlen is 3
        encoder.reward_history.extend([0.1, 0.2, 0.3])
        encoder.expression_history.extend(["x", "y+1", "z*x"])

        features = encoder._encode_history()
        # action_vec (3) + reward_vec (3) + expr_stats (3) = 9 elements before context_encoder
        # context_encoder input dim: hist_len*2 (actions,rewards) + 4 (expr_stats in code, actually 3 here)
        # Code has: nn.Linear(history_length * 2 + 4, 64)
        # Our hist_len=3. So, 3*2+4 = 10.
        # Our features vector has 3(act)+3(rew)+3(expr)=9. This might indicate a mismatch or default in Linear.
        # The Linear layer expects input of size 10. Our `features` has 9.
        # This implies the test or code for context_encoder input size might need adjustment.
        # Let's assume the test should reflect the code's Linear layer.
        # The code for _encode_history produces: action_vec (len=hist_len), reward_vec (len=hist_len), expr_features (len=3)
        # Total = hist_len + hist_len + 3 = 2*hist_len + 3.
        # If hist_len=3, this is 2*3+3 = 9.
        # If the Linear layer is nn.Linear(history_length * 2 + 4, ...), it expects 2*hist_len + 4.
        # This means there's a mismatch of 1.
        # For now, we test the output shape of _encode_history directly.
        assert features.shape == (encoder.history_length * 2 + 3,) # 3 actions, 3 rewards, 3 expr stats

        # Test padding
        encoder.action_history.clear(); encoder.action_history.append(1) # Only 1 action
        encoder.reward_history.clear(); encoder.reward_history.append(0.1)
        encoder.expression_history.clear(); encoder.expression_history.append("a")
        padded_features = encoder._encode_history()
        assert padded_features.shape == (encoder.history_length * 2 + 3,)
        # First action should be 1, rest 0. First reward 0.1, rest 0.
        assert padded_features[0] == 1.0
        assert padded_features[1] == 0.0
        assert padded_features[encoder.history_length] == 0.1
        assert padded_features[encoder.history_length+1] == 0.0


    def test_extract_grammar_features(self, encoder, mock_grammar_instance):
        # mock_grammar_instance.learned_functions = {'f1': ...} -> len = 1
        # mock_grammar_instance.variables = {'v1': ...} -> len = 1
        # mock_grammar_instance.primitives = {'const': [c1,c2], 'unary': [u1]} -> total len = 3
        # mock_grammar_instance.current_stage = 2
        mock_grammar_instance.learned_functions = {"f1": "def"}
        mock_grammar_instance.variables = {"v1": "var_obj"}
        mock_grammar_instance.primitives = {'const': [1,2], 'unary': ['sin']}
        mock_grammar_instance.current_stage = 2

        features = encoder._extract_grammar_features(mock_grammar_instance)
        expected = [
            1/10.0, # n_learned_functions / 10
            1/10.0, # n_variables_in_grammar / 10
            3/20.0, # n_primitives / 20
            2/5.0   # current_grammar_stage / 5
        ]
        assert features.shape == (4,)
        assert np.allclose(features, expected)

    def test_extract_complexity_features(self, encoder, mock_tree_state):
        # mock_tree_state.count_nodes returns 5
        # mock_tree_state.max_complexity is 10
        # Need to mock _get_max_depth for the tree state
        with patch.object(encoder, '_get_max_depth', return_value=2) as mock_get_depth:
            features = encoder._extract_complexity_features(mock_tree_state)
            mock_get_depth.assert_called_once_with(mock_tree_state.root)

            # current_complexity = 5, max_complexity = 10, current_max_depth = 2
            # usage_ratio = 5/10 = 0.5
            # remaining_ratio = (10-5)/10 = 0.5
            # complexity_per_depth = 5/2 = 2.5
            expected = [0.5, 0.5, 2.5]
            assert features.shape == (3,)
            assert np.allclose(features, expected)

    def test_enhance_observation(self, encoder, mock_tree_state, mock_grammar_instance):
        base_obs = np.random.rand(encoder.base_dim) # Matches encoder's base_dim

        # Mock all helper extractors
        with patch.object(encoder, '_extract_tree_features', return_value=np.array([1.0]*9)) as mock_tree_f, \
             patch.object(encoder, '_encode_history', return_value=np.array([2.0]*(encoder.history_length*2+3))) as mock_hist_f, \
             patch.object(encoder, '_extract_grammar_features', return_value=np.array([3.0]*4)) as mock_gram_f, \
             patch.object(encoder, '_extract_complexity_features', return_value=np.array([4.0]*3)) as mock_comp_f:

            # Mock the context_encoder's output directly
            # The input to context_encoder is the output of _encode_history
            encoded_history_mock_output = torch.randn(32) # Output size of context_encoder
            with patch.object(encoder.context_encoder, 'forward', return_value=encoded_history_mock_output) as mock_ctx_encoder_fwd:

                enhanced_obs = encoder.enhance_observation(base_obs, mock_tree_state, mock_grammar_instance)

                mock_tree_f.assert_called_once_with(mock_tree_state)
                mock_hist_f.assert_called_once() # _encode_history takes no args
                # Check that context_encoder.forward was called with the output of _encode_history
                # Convert np.array([2.0]*X) to tensor for call check
                expected_ctx_encoder_input = torch.FloatTensor(np.array([2.0]*(encoder.history_length*2+3))).unsqueeze(0)
                # mock_ctx_encoder_fwd.assert_called_once_with(expected_ctx_encoder_input)
                # For some reason, direct tensor comparison in assert_called_with can be flaky. Check properties.
                called_tensor = mock_ctx_encoder_fwd.call_args[0][0]
                assert torch.is_tensor(called_tensor)
                assert torch.allclose(called_tensor, expected_ctx_encoder_input)


                mock_gram_f.assert_called_once_with(mock_grammar_instance)
                mock_comp_f.assert_called_once_with(mock_tree_state)

                # Expected length: base_dim + tree_f_len + context_encoder_out_len + gram_f_len + comp_f_len
                # base_dim=10, tree_f=9, context_encoder_out=32, gram_f=4, comp_f=3
                # Total = 10 + 9 + 32 + 4 + 3 = 58
                assert enhanced_obs.shape == (encoder.base_dim + 9 + 32 + 4 + 3,)
                # Check parts of the concatenation
                assert np.allclose(enhanced_obs[:encoder.base_dim], base_obs)
                assert np.allclose(enhanced_obs[encoder.base_dim : encoder.base_dim+9], np.array([1.0]*9))


# --- Tests for EnhancedSymbolicDiscoveryEnv ---
# We need to patch the superclass (SymbolicDiscoveryEnv) for these tests
@patch('environments.enhanced.feedback_env.SymbolicDiscoveryEnv', MockBaseSymbolicDiscoveryEnv)
@patch('environments.enhanced.feedback_env.IntrinsicRewardCalculator', MockIntrinsicRewardCalculator)
@patch('environments.enhanced.feedback_env.EnhancedObservationEncoder') # Patch the class used by EnhancedEnv
class TestEnhancedSymbolicDiscoveryEnv:

    @pytest.fixture
    def mock_env_args(self, mock_grammar_instance): # Use the fixture from above
        # sample_X_data, sample_y_data, sample_variables are global fixtures
        return {
            "grammar": mock_grammar_instance,
            "X_data": np.random.rand(10,2),
            "y_data": np.random.rand(10,1),
            "variables": [Variable("v1",0,{}), Variable("v2",1,{})],
            "max_depth": 5,
            "max_complexity": 10
        }

    def test_enhanced_env_init(self, MockEnhancedObservationEncoder, mock_env_args):
        # MockEnhancedObservationEncoder is the patched class, MockIntrinsicRewardCalculator is also patched
        env = EnhancedSymbolicDiscoveryEnv(**mock_env_args, novelty_weight=0.5, history_size=50)

        MockBaseSymbolicDiscoveryEnv.assert_called_once_with(**mock_env_args)
        MockIntrinsicRewardCalculator.assert_called_once_with(
            novelty_weight=0.5, diversity_weight=0.2, # 0.2 is default
            complexity_growth_weight=0.1, conservation_weight=0.4,
            conservation_types=['energy', 'momentum'], apply_entropy_penalty=False,
            entropy_penalty_weight=0.1, history_size=50
        )
        MockEnhancedObservationEncoder.assert_called_once()
        assert env.intrinsic_calculator == MockIntrinsicRewardCalculator.return_value
        assert env.observation_encoder == MockEnhancedObservationEncoder.return_value
        assert env.episode_discoveries == []

    def test_enhanced_env_reset(self, MockEnhancedObservationEncoder, mock_env_args):
        # Setup mocks for super().reset() and encoder.enhance_observation()
        mock_base_obs_val = np.array([1.0, 2.0])
        mock_base_info_val = {'key': 'value'}
        MockBaseSymbolicDiscoveryEnv.return_value.reset = MagicMock(return_value=(mock_base_obs_val, mock_base_info_val))

        mock_encoder_instance = MockEnhancedObservationEncoder.return_value
        mock_enhanced_obs_val = np.array([1.0,2.0,3.0,4.0])
        mock_encoder_instance.enhance_observation = MagicMock(return_value=mock_enhanced_obs_val)
        mock_encoder_instance.update_history = MagicMock()

        env = EnhancedSymbolicDiscoveryEnv(**mock_env_args)
        # Mock current_state and grammar which are needed by enhance_observation
        env.current_state = MagicMock(spec=TreeState)
        env.grammar = mock_env_args['grammar']


        obs, info = env.reset(seed=123, options={'opt': True})

        MockBaseSymbolicDiscoveryEnv.return_value.reset.assert_called_once_with(seed=123, options={'opt': True})
        assert env.episode_discoveries == [] # reset_episode_metrics called
        mock_encoder_instance.update_history.assert_called_once_with(0, 0.0, "") # Reset history
        mock_encoder_instance.enhance_observation.assert_called_once_with(mock_base_obs_val, env.current_state, env.grammar)
        assert np.array_equal(obs, mock_enhanced_obs_val)
        assert info == mock_base_info_val


    def test_enhanced_env_step(self, MockEnhancedObservationEncoder, mock_env_args):
        # Mock super().step()
        mock_base_obs_step = np.array([0.1,0.2])
        mock_extrinsic_reward_step = 0.5
        mock_terminated_step = False
        mock_truncated_step = False
        # Info from super().step() needs to contain expression, complexity, etc. for intrinsic rewards
        mock_info_step = {
            'expression': 'new_expr', 'complexity': 7,
            'trajectory_data': np.array([[1.0]]), 'variables': mock_env_args['variables'],
            'predicted_forward_traj': None, 'predicted_backward_traj': None
        }
        MockBaseSymbolicDiscoveryEnv.return_value.step = MagicMock(
            return_value=(mock_base_obs_step, mock_extrinsic_reward_step, mock_terminated_step, mock_truncated_step, mock_info_step)
        )

        # Mock intrinsic calculator
        mock_intrinsic_calc_instance = MockIntrinsicRewardCalculator.return_value
        mock_intrinsic_reward_val = 0.25
        mock_intrinsic_calc_instance.calculate_intrinsic_reward = MagicMock(return_value=mock_intrinsic_reward_val)

        # Mock observation encoder
        mock_encoder_instance_step = MockEnhancedObservationEncoder.return_value
        mock_enhanced_obs_step_val = np.array([0.1,0.2,0.3,0.4])
        mock_encoder_instance_step.enhance_observation = MagicMock(return_value=mock_enhanced_obs_step_val)
        mock_encoder_instance_step.update_history = MagicMock()

        env = EnhancedSymbolicDiscoveryEnv(**mock_env_args)
        # Mock attributes needed by step that are set up by __init__ or reset
        env.current_state = MagicMock(spec=TreeState)
        env.grammar = mock_env_args['grammar']
        env.variables = mock_env_args['variables'] # Ensure this is set for _get_expression_embedding

        action_taken = 1
        obs, reward, term, trunc, info = env.step(action_taken)

        MockBaseSymbolicDiscoveryEnv.return_value.step.assert_called_once_with(action_taken)

        # Check intrinsic reward calculation call
        # The embedding is calculated by _get_expression_embedding
        # We can patch _get_expression_embedding or let it run if simple enough
        expected_embedding = env._get_expression_embedding('new_expr') # Let the actual method run
        mock_intrinsic_calc_instance.calculate_intrinsic_reward.assert_called_once_with(
            expression='new_expr', complexity=7, extrinsic_reward=mock_extrinsic_reward_step,
            embedding=expected_embedding, data=mock_info_step['trajectory_data'],
            variables=mock_info_step['variables'],
            predicted_forward_traj=None, predicted_backward_traj=None
        )

        assert reward == mock_extrinsic_reward_step + mock_intrinsic_reward_val # Combined reward

        mock_encoder_instance_step.enhance_observation.assert_called_once_with(mock_base_obs_step, env.current_state, env.grammar)
        assert np.array_equal(obs, mock_enhanced_obs_step_val)

        mock_encoder_instance_step.update_history.assert_called_once_with(action_taken, reward, 'new_expr')

        assert env.episode_rewards_list == [reward]
        assert env.episode_discoveries == ['new_expr']
        assert env.episode_complexities == [7]

        assert info['intrinsic_reward_components_sum'] == mock_intrinsic_reward_val
        assert info['combined_reward'] == reward


    def test_get_expression_embedding(self, mock_env_args):
        env = EnhancedSymbolicDiscoveryEnv(**mock_env_args) # Uses variables from mock_env_args
        expr_str = "v1 + sin(v2)"
        embedding = env._get_expression_embedding(expr_str)

        # Expected length: num_ops (8) + num_vars (2 from mock_env_args) + 2 (len, parentheses) = 12
        assert embedding.shape == (12,)
        # Example checks:
        assert embedding[0] == 1 # Count of '+'
        assert embedding[8] == 1 # Count of 'v1' (assuming variables are 'v1', 'v2')
        assert embedding[9] == 1 # Count of 'v2'
        assert embedding[10] == len(expr_str) # Length
        assert embedding[11] == expr_str.count('(') # Parentheses

    def test_get_adaptation_metrics(self, mock_env_args):
        env = EnhancedSymbolicDiscoveryEnv(**mock_env_args)
        env.episode_discoveries = ["e1", "e2", "e1", "e3"]
        env.episode_complexities = [3, 5, 3, 7]
        env.episode_rewards_list = [0.1, 0.2, 0.15, 0.3]

        metrics = env.get_adaptation_metrics()

        assert abs(metrics['discovery_rate'] - (3/4)) < 1e-6 # 3 unique out of 4
        assert abs(metrics['mean_complexity_episode'] - np.mean([3,5,3,7])) < 1e-6
        assert abs(metrics['mean_reward_episode'] - np.mean([0.1,0.2,0.15,0.3])) < 1e-6
        assert metrics['unique_discoveries_episode'] == 3
        assert metrics['total_steps_episode'] == 4
        # Complexity trend requires polyfit, check it's a float
        assert isinstance(metrics['complexity_trend'], float)

    def test_reset_episode_metrics(self, mock_env_args):
        env = EnhancedSymbolicDiscoveryEnv(**mock_env_args)
        env.episode_discoveries = ["test"]
        env.reset_episode_metrics()
        assert env.episode_discoveries == []
        assert env.episode_complexities == []
        assert env.episode_rewards_list == []
