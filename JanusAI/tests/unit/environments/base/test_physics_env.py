"""
Tests for environments/base/physics_env.py
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, ANY

# Import the class to be tested
from JanusAI.environments.base.physics_env import PhysicsEnvironment

# Mock dependencies that are imported by physics_env
MockSymbolicDiscoveryEnv = MagicMock(name="MockSymbolicDiscoveryEnv")
MockExpression = MagicMock(name="MockExpression")
MockVariable = MagicMock(name="MockVariable")
MockPhysicsTask = MagicMock(name="MockPhysicsTask")
MockConservationDetector = MagicMock(name="MockConservationDetector")
MockProgressiveGrammar = MagicMock(name="MockProgressiveGrammar") # Assuming this is the grammar type

# Helper to create Variable instances for testing
def create_mock_variable(name, index):
    var = MagicMock(spec=MockVariable) # Use spec of the patched class
    var.name = name
    var.index = index
    var.properties = {}
    return var


@patch('JanusAI.environments.base.physics_env.SymbolicDiscoveryEnv', MockSymbolicDiscoveryEnv)
@patch('JanusAI.environments.base.physics_env.Expression', MockExpression)
@patch('JanusAI.environments.base.physics_env.Variable', MockVariable)
@patch('JanusAI.physics.data.task_distribution.PhysicsTask', MockPhysicsTask) # Assuming this is the path
@patch('JanusAI.physics.laws.conservation.ConservationDetector', MockConservationDetector)
class TestPhysicsEnvironment:

    @pytest.fixture(autouse=True)
    def reset_global_mocks(self):
        MockSymbolicDiscoveryEnv.reset_mock()
        MockExpression.reset_mock()
        MockVariable.reset_mock()
        MockPhysicsTask.reset_mock() # Reset the class mock itself
        MockConservationDetector.reset_mock()
        MockProgressiveGrammar.reset_mock() # Also reset grammar if it's used globally or as a class

    @pytest.fixture
    def mock_physics_task_instance(self):
        task = MagicMock(spec=MockPhysicsTask) # Instance of the mocked class
        task.variables = ['x', 'y', 'target_var']
        task.physical_parameters = {'g': 9.8}
        task.true_law = "x + y"
        task.symmetries = ["time_translation"]
        task.conserved_quantities = ["energy"]
        task.name = "TestTask"
        task.difficulty = 0.5
        task.noise_level = 0.0
        # Mock generate_data method for the instance
        task.generate_data = MagicMock(return_value=np.random.rand(100, 3)) # 100 samples, 3 vars
        return task

    @pytest.fixture
    def mock_grammar_instance(self):
        return MagicMock(spec=MockProgressiveGrammar)


    def test_init(self, mock_grammar_instance, mock_physics_task_instance):
        # Mock Variable class to return specific instances when called
        # This simulates Variable(name=var_name, index=idx, properties={})
        mock_vars_created = [create_mock_variable(name, i) for i, name in enumerate(mock_physics_task_instance.variables)]
        MockVariable.side_effect = lambda name, index, properties: next(
            (v for v in mock_vars_created if v.name == name and v.index == index), None
        )

        env = PhysicsEnvironment(
            grammar=mock_grammar_instance,
            physics_task=mock_physics_task_instance,
            max_depth=5,
            max_complexity=10,
            reward_config={'mse': -1.0},
            action_space_size=10, # Example
            provide_tree_structure=True
        )

        MockSymbolicDiscoveryEnv.assert_called_once_with(
            grammar=mock_grammar_instance,
            target_data=ANY, # Initial dummy data (np.array([]))
            variables=mock_physics_task_instance.variables, # Passed as list of strings initially
            max_depth=5,
            max_complexity=10,
            reward_config={'mse': -1.0},
            action_space_size=10,
            provide_tree_structure=True
        )
        # Check that target_data was an empty array
        assert np.array_equal(MockSymbolicDiscoveryEnv.call_args[1]['target_data'], np.array([]))


        assert env.physics_task == mock_physics_task_instance
        MockConservationDetector.assert_called_once()
        assert env.conservation_detector == MockConservationDetector.return_value

        # Verify self.variables are Variable objects
        assert len(env.variables) == len(mock_physics_task_instance.variables)
        for i, var_name in enumerate(mock_physics_task_instance.variables):
            # This check depends on how MockVariable instances are created and returned by the class mock
            # For this to pass, MockVariable needs to be set up to return specific instances
            # or we check properties of the created variables.
            assert env.variables[i].name == var_name
            assert env.variables[i].index == i
            assert isinstance(env.variables[i].properties, dict)

        # Verify MockVariable was called for each variable name
        assert MockVariable.call_count == len(mock_physics_task_instance.variables)
        for i, var_name in enumerate(mock_physics_task_instance.variables):
            MockVariable.assert_any_call(name=var_name, index=i, properties={})


    def test_reset(self, mock_grammar_instance, mock_physics_task_instance):
        # Prepare mock for super().reset()
        mock_super_reset_obs = np.random.rand(5) # Dummy observation
        mock_super_reset_info = {'tree_structure': {}} # Dummy info from super
        MockSymbolicDiscoveryEnv.return_value.reset = MagicMock(
            return_value=(mock_super_reset_obs, mock_super_reset_info)
        )

        # Mock Variable class calls as in test_init
        mock_vars_created = [create_mock_variable(name, i) for i, name in enumerate(mock_physics_task_instance.variables)]
        MockVariable.side_effect = lambda name, index, properties: next(
            (v for v in mock_vars_created if v.name == name and v.index == index), None
        )

        env = PhysicsEnvironment(grammar=mock_grammar_instance, physics_task=mock_physics_task_instance, max_depth=3, max_complexity=5)

        # Call reset
        obs, info = env.reset(seed=42, options={'some_option': True})

        # Check physics_task.generate_data was called
        mock_physics_task_instance.generate_data.assert_called_once_with(n_samples=1000, add_noise=False)

        # Check self.target_data was updated (used by super().reset)
        # env.target_data should be the return value of generate_data
        assert np.array_equal(env.target_data, mock_physics_task_instance.generate_data.return_value)

        # Check super().reset() was called
        MockSymbolicDiscoveryEnv.return_value.reset.assert_called_once_with(seed=42, options={'some_option': True})

        assert np.array_equal(obs, mock_super_reset_obs)

        # Check info dictionary
        expected_info = {
            'tree_structure': {}, # From super_info
            'physical_constants': mock_physics_task_instance.physical_parameters,
            'true_law': mock_physics_task_instance.true_law,
            'symmetries': mock_physics_task_instance.symmetries,
            'conserved_quantities': mock_physics_task_instance.conserved_quantities,
            'task_name': mock_physics_task_instance.name,
            'task_difficulty': mock_physics_task_instance.difficulty,
            'trajectory_data': {} # Will be populated
        }
        # Populate trajectory_data for expected_info
        generated_data = mock_physics_task_instance.generate_data.return_value
        for idx, var_name in enumerate(mock_physics_task_instance.variables):
            if idx < generated_data.shape[1]:
                expected_info['trajectory_data'][var_name] = generated_data[:, idx]

        assert info == expected_info


    def test_step(self, mock_grammar_instance, mock_physics_task_instance):
        # Prepare mock for super().step()
        mock_super_step_obs = np.random.rand(5)
        mock_super_step_reward = 0.5
        mock_super_step_terminated = False
        mock_super_step_truncated = False
        mock_super_step_info = {'expression': 'x*2', 'complexity': 3} # Dummy info from super
        MockSymbolicDiscoveryEnv.return_value.step = MagicMock(
            return_value=(mock_super_step_obs, mock_super_step_reward, mock_super_step_terminated, mock_super_step_truncated, mock_super_step_info)
        )

        # Mock Variable class calls as in test_init
        mock_vars_created = [create_mock_variable(name, i) for i, name in enumerate(mock_physics_task_instance.variables)]
        MockVariable.side_effect = lambda name, index, properties: next(
            (v for v in mock_vars_created if v.name == name and v.index == index), None
        )

        env = PhysicsEnvironment(grammar=mock_grammar_instance, physics_task=mock_physics_task_instance, max_depth=3, max_complexity=5)

        # Call reset first to set up internal state like self.target_data
        # Mock generate_data for the reset call
        reset_data = np.array([[1,2,3],[4,5,6]])
        mock_physics_task_instance.generate_data.return_value = reset_data
        env.reset()
        # Reset the mock for generate_data if it's called again in step's info population
        mock_physics_task_instance.generate_data.reset_mock()


        # Call step
        action_taken = 1
        obs, reward, terminated, truncated, info = env.step(action_taken)

        # Check super().step() was called
        MockSymbolicDiscoveryEnv.return_value.step.assert_called_once_with(action_taken)

        assert np.array_equal(obs, mock_super_step_obs)
        assert reward == mock_super_step_reward
        assert terminated == mock_super_step_terminated
        assert truncated == mock_super_step_truncated

        # Check info dictionary from step
        # The current code for step *always* calls generate_data for info['trajectory_data']
        mock_physics_task_instance.generate_data.assert_called_once_with(n_samples=100)
        step_generated_data = mock_physics_task_instance.generate_data.return_value

        expected_info_from_step = {
            'expression': 'x*2', # From super_info
            'complexity': 3,     # From super_info
            'physical_constants': mock_physics_task_instance.physical_parameters,
            'true_law': mock_physics_task_instance.true_law,
            'symmetries': mock_physics_task_instance.symmetries,
            'conserved_quantities': mock_physics_task_instance.conserved_quantities,
            'task_name': mock_physics_task_instance.name,
            'task_difficulty': mock_physics_task_instance.difficulty,
            'trajectory_data': {} # Will be populated by the new generate_data call
        }
        for idx, var_name in enumerate(mock_physics_task_instance.variables):
             if idx < step_generated_data.shape[1]:
                expected_info_from_step['trajectory_data'][var_name] = step_generated_data[:, idx]

        assert info == expected_info_from_step
