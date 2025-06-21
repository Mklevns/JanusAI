"""
Tests for environments/base/symbolic_env.py
"""
import pytest
import numpy as np
import gym

from JanusAI.environments.base.symbolic_env import (
    SymbolicDiscoveryEnv, ExpressionNode, TreeState, NodeType
)
from JanusAI.core.grammar.base_grammar import ProgressiveGrammar
from JanusAI.core.expressions.expression import Expression, Variable
from unittest.mock import MagicMock, patch


# --- Helper Fixtures ---
@pytest.fixture
def mock_grammar():
    grammar = MagicMock(spec=ProgressiveGrammar)
    grammar.primitives = {
        'constants': {'const1': 1.0, 'const2': 2.0},
        'unary_ops': {'sin': np.sin, 'cos': np.cos},
        'binary_ops': {'+': np.add, '*': np.multiply},
        'calculus_ops': {} # None for simplicity here
    }
    return grammar

@pytest.fixture
def sample_variables():
    return [Variable('x', 0, {}), Variable('y', 1, {})]

@pytest.fixture
def sample_X_data():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # 3 samples, 2 features

@pytest.fixture
def sample_y_data():
    return np.array([[3.0], [7.0], [11.0]]) # Corresponds to x + y

# --- Tests for ExpressionNode ---
class TestExpressionNode:
    def test_node_creation_defaults(self):
        node = ExpressionNode(NodeType.CONSTANT, 1.0)
        assert node.node_type == NodeType.CONSTANT
        assert node.value == 1.0
        assert node.children == []
        assert node.depth == 0

    def test_is_complete(self):
        empty_node = ExpressionNode(NodeType.EMPTY)
        assert not empty_node.is_complete()

        const_node = ExpressionNode(NodeType.CONSTANT, 1.0)
        assert const_node.is_complete()

        var_node = ExpressionNode(NodeType.VARIABLE, Variable('x',0,{}))
        assert var_node.is_complete()

        op_incomplete = ExpressionNode(NodeType.OPERATOR, '+') # Needs 2 children
        assert not op_incomplete.is_complete()

        op_incomplete.children.append(const_node)
        assert not op_incomplete.is_complete() # Still needs 1 more

        op_complete_children_not = ExpressionNode(NodeType.OPERATOR, '+', children=[const_node, empty_node])
        assert not op_complete_children_not.is_complete() # One child is not complete

        op_complete = ExpressionNode(NodeType.OPERATOR, '+', children=[const_node, var_node])
        assert op_complete.is_complete()

        unary_op_incomplete = ExpressionNode(NodeType.OPERATOR, 'sin') # Needs 1 child
        assert not unary_op_incomplete.is_complete()

        unary_op_complete = ExpressionNode(NodeType.OPERATOR, 'sin', children=[const_node])
        assert unary_op_complete.is_complete()

    def test_get_required_children(self):
        assert ExpressionNode(NodeType.OPERATOR, '+')._get_required_children() == 2
        assert ExpressionNode(NodeType.OPERATOR, 'sin')._get_required_children() == 1
        assert ExpressionNode(NodeType.OPERATOR, 'unknown_op')._get_required_children() == 2 # Default


# --- Tests for TreeState ---
class TestTreeState:
    def test_tree_state_defaults(self):
        state = TreeState()
        assert state.root is None
        assert state.current_node is None
        assert state.incomplete_nodes == []
        assert state.depth == 0
        assert state.node_count == 0

    def test_is_complete(self):
        state = TreeState()
        assert not state.is_complete() # No root

        state.root = ExpressionNode(NodeType.EMPTY)
        assert not state.is_complete() # Root is not complete

        const_node = ExpressionNode(NodeType.CONSTANT, 1.0)
        state.root = const_node
        assert state.is_complete() # Single constant node is complete

        # Incomplete operator
        op_node = ExpressionNode(NodeType.OPERATOR, '+')
        state.root = op_node
        state.incomplete_nodes = [op_node] # Manually set for this test
        assert not state.is_complete()

        # Complete operator, but still in incomplete_nodes (should not happen in practice if logic is correct)
        var_x = Variable('x',0,{})
        op_node_full_children = ExpressionNode(NodeType.OPERATOR, '+', children=[const_node, ExpressionNode(NodeType.VARIABLE, var_x)])
        state.root = op_node_full_children
        state.incomplete_nodes = [op_node_full_children] # If this list is not empty, it's not complete by TreeState's logic
        assert not state.is_complete()

        state.incomplete_nodes = [] # Now it should be complete
        assert state.is_complete()


    def test_to_expression(self):
        state = TreeState()
        assert state.to_expression() is None # Incomplete

        # Build a simple tree: x + 1.0
        var_x_obj = Variable('x',0,{})
        node_x = ExpressionNode(NodeType.VARIABLE, var_x_obj)
        node_const1 = ExpressionNode(NodeType.CONSTANT, 1.0)
        node_plus = ExpressionNode(NodeType.OPERATOR, '+', children=[node_x, node_const1])
        state.root = node_plus
        # state.incomplete_nodes should be empty for a complete tree

        expr = state.to_expression()
        assert isinstance(expr, Expression)
        assert expr.operator == '+'
        assert isinstance(expr.operands[0], Variable)
        assert expr.operands[0].name == 'x'
        assert expr.operands[1].operator == 'const'
        assert expr.operands[1].operands[0] == 1.0

        # Test incomplete tree to expression
        state.root = ExpressionNode(NodeType.OPERATOR, '+', children=[node_x]) # Missing one child
        state.incomplete_nodes = [state.root]
        assert state.to_expression() is None
        state.incomplete_nodes = [] # Even if list is empty, node itself is incomplete
        assert state.to_expression() is None


# --- Tests for SymbolicDiscoveryEnv ---

@patch('JanusAI.environments.base.symbolic_env.evaluate_expression_on_data', MagicMock(return_value=np.array([1.0, 1.0, 1.0])))
@patch('JanusAI.environments.base.symbolic_env.calculate_expression_complexity', MagicMock(return_value=5))
class TestSymbolicDiscoveryEnv:

    @pytest.fixture
    def env(self, mock_grammar, sample_X_data, sample_y_data, sample_variables):
        return SymbolicDiscoveryEnv(
            grammar=mock_grammar,
            X_data=sample_X_data,
            y_data=sample_y_data,
            variables=sample_variables,
            max_depth=5,
            max_complexity=10, # Max nodes in tree
            provide_tree_structure=False # Simpler observation for some tests
        )

    def test_init_successful(self, env, mock_grammar, sample_X_data, sample_y_data, sample_variables):
        assert env.grammar == mock_grammar
        assert np.array_equal(env.X_data, sample_X_data)
        assert np.array_equal(env.y_data, sample_y_data) # Should be reshaped to 2D
        assert env.y_data.ndim == 2
        assert env.variables == sample_variables
        assert env.max_depth == 5
        assert env.max_complexity == 10
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert 'mse_weight' in env.reward_config # Check default config

    def test_init_input_validation(self, mock_grammar, sample_X_data, sample_y_data, sample_variables):
        # X_data wrong ndim
        with pytest.raises(ValueError, match="X_data must be 2D array"):
            SymbolicDiscoveryEnv(mock_grammar, sample_X_data.flatten(), sample_y_data, sample_variables)

        # y_data wrong ndim (e.g. 3D)
        with pytest.raises(ValueError, match="y_data must be 1D or 2D array"):
            SymbolicDiscoveryEnv(mock_grammar, sample_X_data, np.random.rand(3,1,1), sample_variables)

        # Mismatched samples X and y
        with pytest.raises(ValueError, match="must have same number of samples"):
            SymbolicDiscoveryEnv(mock_grammar, sample_X_data, sample_y_data[:-1], sample_variables)

        # Mismatched variables and X_data features
        with pytest.raises(ValueError, match="Number of variables .* must match"):
            SymbolicDiscoveryEnv(mock_grammar, sample_X_data, sample_y_data, sample_variables[:-1])

    def test_calculate_action_space_size(self, env, mock_grammar, sample_variables):
        # constants: 2, variables: 2, unary_ops: 2, binary_ops: 2, calculus_ops: 0
        # Total grammar elements = 2+2+2+2+0 = 8. Plus 1 for no-op (not in current code).
        # Current code: 2 (const) + 2 (var) + (2 unary + 2 binary + 0 calculus) = 8
        # The mock_grammar has 2 const, 2 unary, 2 binary. sample_variables has 2.
        # Expected size = 2 + 2 + (2+2+0) = 8. The code adds +1 for no-op.
        # Let's recheck: n_const + n_var + n_ops + 1 (no-op)
        # For this mock: 2 + 2 + (2+2) + 1 = 9.
        # However, _calculate_action_space_size in current code does not add +1 for no-op.
        # It sums counts of constants, variables, and all operator types.
        # So, for the mock: 2 (const) + 2 (vars) + 2 (unary) + 2 (binary) = 8
        # Let's make the test reflect the code's logic.
        expected_size = (
            len(mock_grammar.primitives['constants']) +
            len(sample_variables) +
            len(mock_grammar.primitives['unary_ops']) +
            len(mock_grammar.primitives['binary_ops']) +
            len(mock_grammar.primitives['calculus_ops'])
        )
        assert env._calculate_action_space_size() == expected_size
        assert env.action_space.n == expected_size # If action_space_size not passed to init

    def test_reset(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert env.episode_steps == 0
        assert isinstance(env.current_state, TreeState)
        assert env.current_state.root is None # Starts empty

        assert 'episode_steps' in info
        assert 'tree_depth' in info
        assert 'node_count' in info
        assert 'is_complete' in info

    def test_action_to_element(self, env, mock_grammar, sample_variables):
        # Constants (0, 1)
        assert env._action_to_element(0) == ('constant', ('const1', 1.0))
        assert env._action_to_element(1) == ('constant', ('const2', 2.0))
        # Variables (2, 3)
        assert env._action_to_element(2) == ('variable', sample_variables[0]) # x
        assert env._action_to_element(3) == ('variable', sample_variables[1]) # y
        # Unary Ops (4, 5)
        assert env._action_to_element(4) == ('operator', 'sin')
        assert env._action_to_element(5) == ('operator', 'cos')
        # Binary Ops (6, 7)
        assert env._action_to_element(6) == ('operator', '+')
        assert env._action_to_element(7) == ('operator', '*')
        # Invalid
        assert env._action_to_element(8) is None

    def test_apply_element_to_tree(self, env):
        env.reset()
        state = env.current_state

        # 1. Add operator as root
        applied = env._apply_element_to_tree(('operator', '+'))
        assert applied
        assert state.root is not None and state.root.value == '+'
        assert state.root.node_type == NodeType.OPERATOR
        assert state.node_count == 1
        assert state.incomplete_nodes == [state.root]

        # 2. Add variable as first child
        applied = env._apply_element_to_tree(('variable', Variable('x',0,{})))
        assert applied
        assert len(state.root.children) == 1
        assert state.root.children[0].value.name == 'x'
        assert state.node_count == 2
        assert state.incomplete_nodes == [state.root] # Root still incomplete

        # 3. Add constant as second child
        applied = env._apply_element_to_tree(('constant', ('const1',1.0))) # ('const1',1.0) is element[1]
        assert applied
        assert len(state.root.children) == 2
        assert state.root.children[1].value == 1.0 # const_value is element[1][1]
        assert state.node_count == 3
        assert state.incomplete_nodes == [] # Root is now complete
        assert state.is_complete()

        # 4. Try to add to a complete tree (should fail)
        applied = env._apply_element_to_tree(('operator', '*'))
        assert not applied


    def test_step_flow(self, env):
        env.reset()
        # Action 6 corresponds to '+' operator
        obs, reward, terminated, truncated, info = env.step(6)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        assert env.current_state.root is not None
        assert env.current_state.root.value == '+' # Action 6 is '+'
        assert not terminated # Tree is not complete yet

        # Action 2 corresponds to variable 'x'
        obs, reward, terminated, truncated, info = env.step(2)
        assert len(env.current_state.root.children) == 1
        assert not terminated

        # Action 0 corresponds to 'const1' (value 1.0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert len(env.current_state.root.children) == 2
        assert env.current_state.is_complete()
        assert 'expression' in info
        assert info['expression'] == "x + 1.0" # Mock Variable str is its name
        assert terminated # Episode terminates as expression is complete

    def test_reward_calculation(self, env):
        env.reset()
        # Build expression "x + y"
        # Corresponds to actions: + (idx 6), x (idx 2), y (idx 3)
        env.step(6) # +
        env.step(2) # x
        _, reward_final, _, _, _ = env.step(3) # y -> completes "x + y"

        # evaluate_expression_on_data is mocked to return [1,1,1]
        # y_true is [[3],[7],[11]]
        # mse = mean( ([1,1,1] - [3,7,11])^2 ) = mean( [-2,-6,-10]^2 ) = mean( [4,36,100] ) = 140/3 = 46.66
        # complexity is mocked to 5
        # depth is 1 for "x+y" (root at depth 0, children at 1)
        # mse_weight = -1.0, complexity_penalty = -0.01, depth_penalty = -0.001, completion_bonus = 1.0
        # mse_scale_factor = 1.0
        # scaled_mse = 46.66 / 1.0 = 46.66
        # Expected reward = -1.0 * 46.66 + -0.01 * 5 + -0.001 * 1 + 1.0
        #                 = -46.66 - 0.05 - 0.001 + 1.0 = -45.711
        # Parsimony bonus does not apply as mse is high.

        # Let's mock evaluate_expression_on_data for this specific test for better control
        # X_data = [[1,2],[3,4],[5,6]], y_data = [[3],[7],[11]]
        # If expression is "x+y", predictions = [1+2, 3+4, 5+6] = [3,7,11]
        # Then MSE = 0
        with patch('JanusAI.environments.base.symbolic_env.evaluate_expression_on_data', return_value=np.array([3.0,7.0,11.0])) as mock_eval, \
             patch('JanusAI.environments.base.symbolic_env.calculate_expression_complexity', return_value=3) as mock_complexity: # "x+y" complexity

            env.reset()
            env.step(6) # +
            env.step(2) # x
            _, reward_perfect, _, _, _ = env.step(3) # y

            mock_eval.assert_called_once()
            # Check str(expression) passed was "x + y" (depends on Variable's __str__)
            # args, kwargs = mock_eval.call_args
            # assert args[0] == "x + y" # This is tricky due to Variable object string form

            # MSE = 0, complexity = 3, depth = 1
            # Expected reward = -1.0 * 0 + -0.01 * 3 + -0.001 * 1 + 1.0 (completion) + 0.1 (parsimony, mse<0.1, C<10)
            #                 = 0 - 0.03 - 0.001 + 1.0 + 0.1 = 1.069
            assert abs(reward_perfect - 1.069) < 1e-6


    def test_termination_truncation(self, env):
        env.reset()
        # Max depth termination
        env.current_state.depth = env.max_depth
        assert env._is_terminated()

        env.current_state.depth = 1 # Reset depth
        # Max complexity (node count) termination
        env.current_state.node_count = env.max_complexity
        assert env._is_terminated()

        env.current_state.node_count = 1 # Reset count
        # Max episode steps truncation
        env.episode_steps = env.max_episode_steps
        assert env._is_truncated()

    def test_observation_structure_with_tree_tensor(self, mock_grammar, sample_X_data, sample_y_data, sample_variables):
        env_with_tree = SymbolicDiscoveryEnv(
            mock_grammar, sample_X_data, sample_y_data, sample_variables,
            provide_tree_structure=True, max_nodes=3 # Small max_nodes for test
        )
        env_with_tree.reset()
        # Build a small tree: x (action 2)
        env_with_tree.step(2)
        obs = env_with_tree._get_observation()

        base_dim = 5
        tree_dim = 3 * 4 # max_nodes * 4
        assert obs.shape == (base_dim + tree_dim,)

        # Check tree tensor part (first node should be 'x')
        tree_tensor_part = obs[base_dim:].reshape(3,4)
        assert tree_tensor_part[0,0] == float(NodeType.VARIABLE.value) # Node type
        # Value hash, depth, parent_idx would also be set.
        # hash(str(Variable('x',0,{})))
        expected_val_hash = float(hash(str(sample_variables[0])) % 1000) / 1000
        assert abs(tree_tensor_part[0,1] - expected_val_hash) < 1e-9
        assert tree_tensor_part[0,2] == 0.0 # Depth of root
        assert tree_tensor_part[0,3] == 0.0 # Parent index (0 for root's "parent")

        # Other nodes in tensor should be zero
        assert np.all(tree_tensor_part[1:] == 0)
