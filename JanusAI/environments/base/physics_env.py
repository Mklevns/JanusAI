"""
PhysicsEnvironment
==================

Defines a base class for physics-specific symbolic discovery environments.
These environments focus on discovering physical laws and relationships
from simulated or real physics data.

The environment now consumes a `PhysicsTask` object, delegating data generation
to the `PhysicsTask`'s associated data generator.
"""

import numpy as np
import sympy as sp
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

# Import base symbolic environment
from JanusAI.environments.base.symbolic_env import SymbolicDiscoveryEnv, TreeState, ExpressionNode

# Import core expression and symbolic math utilities
from JanusAI.core.expressions.expression import Expression, Variable
from JanusAI.core.expressions.symbolic_math import get_variables_from_expression, evaluate_expression_on_data

# Import physics-specific utilities
from JanusAI.physics.laws.conservation import ConservationDetector # For potential use in reward
# Import PhysicsTask from the new task_distribution module
from JanusAI.physics.data.task_distribution import PhysicsTask


class PhysicsEnvironment(SymbolicDiscoveryEnv):
    """
    Base class for environments where the goal is to discover physical laws.

    Extends SymbolicDiscoveryEnv by integrating with a `PhysicsTask` object
    to handle data generation and task-specific metadata.
    """

    def __init__(self,
                 grammar: Any, # ProgressiveGrammar
                 physics_task: PhysicsTask, # The specific physics task for this environment instance
                 max_depth: int,
                 max_complexity: int,
                 reward_config: Optional[Dict[str, float]] = None,
                 action_space_size: Optional[int] = None,
                 provide_tree_structure: bool = False # Pass to super
                 ):
        """
        Initializes the PhysicsEnvironment with a specific physics task.

        Args:
            grammar: The grammar object (e.g., ProgressiveGrammar).
            physics_task: An instance of `PhysicsTask` defining the current physics problem.
            max_depth: Maximum depth of the expression tree allowed.
            max_complexity: Maximum complexity of the expression tree.
            reward_config: Configuration for the reward function.
            action_space_size: Explicit size of the action space if different from grammar.
            provide_tree_structure: Whether to include 'tree_structure' in info dict.
        """
        # The `target_data` and `variables` are now derived from the `physics_task`
        # and will be updated in `reset` as data is generated.
        # Initialize `SymbolicDiscoveryEnv` with dummy data/variables, which `reset` will override.
        super().__init__(
            grammar=grammar,
            target_data=np.array([]), # Dummy initial data
            variables=physics_task.variables, # Use variable names from task (will convert to Variable objects later)
            max_depth=max_depth,
            max_complexity=max_complexity,
            reward_config=reward_config,
            action_space_size=action_space_size,
            provide_tree_structure=provide_tree_structure
        )
        self.physics_task = physics_task
        
        # Convert variable names from PhysicsTask into Variable objects for the environment
        # This mapping is crucial for `SymbolicDiscoveryEnv` and expression evaluation
        self.variables: List[Variable] = []
        for idx, var_name in enumerate(physics_task.variables):
            # For now, properties are empty, but could be derived from task metadata if available
            self.variables.append(Variable(name=var_name, index=idx, properties={}))

        # Update `self.variables` in the superclass (SymbolicDiscoveryEnv) as well
        # if SymbolicDiscoveryEnv maintains its own list.
        # Assumes SymbolicDiscoveryEnv's __init__ or a method allows updating.
        # For a dataclass-based approach, it might be immutable, so careful.
        # If SymbolicDiscoveryEnv takes `variables` in __init__ and copies them, this is fine.

        # The following attributes are now part of `self.physics_task`
        # self.physical_constants = physics_task.physical_parameters
        # self.true_law_sympy = sp.sympify(physics_task.true_law)
        # self.true_law_str = physics_task.true_law
        
        # Initialize ConservationDetector if rewards use it
        self.conservation_detector = ConservationDetector()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment for a new episode.
        Generates new physics data using the associated `PhysicsTask`.
        """
        # Generate new trajectory data for the episode using the PhysicsTask's generator
        # This assumes `self.physics_task.generate_data` handles random parameters internally.
        # `n_samples` should be based on `self.num_steps` if that's still relevant, or a default.
        # Let's generate 1000 samples for the episode's data.
        episode_data_matrix = self.physics_task.generate_data(n_samples=1000, add_noise=self.physics_task.noise_level > 0)
        
        # Update the target_data in the SymbolicDiscoveryEnv base class.
        # Assumes the last column of the generated data is the target (dependent variable).
        # This convention needs to be consistent with how `PhysicsTask` lists `variables`.
        self.target_data = episode_data_matrix # Update for SymbolicDiscoveryEnv's internal use

        # The initial observation for the agent (e.g., encoded blank tree, or initial state)
        # The base `SymbolicDiscoveryEnv.reset` will handle the initial observation.
        
        # Pass relevant physics-specific info to the info dict
        info = {
            'trajectory_data': {}, # Will be populated from episode_data_matrix
            'physical_constants': self.physics_task.physical_parameters,
            'true_law': self.physics_task.true_law,
            'symmetries': self.physics_task.symmetries,
            'conserved_quantities': self.physics_task.conserved_quantities,
            'task_name': self.physics_task.name,
            'task_difficulty': self.physics_task.difficulty
        }

        # Populate 'trajectory_data' in info by mapping variable names to columns in episode_data_matrix
        for idx, var_name in enumerate(self.physics_task.variables):
            if idx < episode_data_matrix.shape[1]:
                info['trajectory_data'][var_name] = episode_data_matrix[:, idx]

        # Call the parent's reset method to initialize the expression tree state
        # It will use `self.target_data` and `self.variables` from this instance.
        obs, super_info = super().reset(seed=seed, options=options)
        super_info.update(info) # Merge physics-specific info

        # The `info['variables']` will now correctly contain the `Variable` objects defined in this `PhysicsEnvironment` instance.
        return obs, super_info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Performs one step in the environment.

        Args:
            action: The action taken by the agent.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Call the parent's step method, which will update current_state, etc.
        obs, reward, terminated, truncated, info = super().step(action)

        # Ensure essential physics-specific details are in the info dict for rewards
        info['trajectory_data'] = info.get('trajectory_data', self.physics_task.generate_data(n_samples=100)) # Provide fresh data if not already there
        info['physical_constants'] = self.physics_task.physical_parameters
        info['true_law'] = self.physics_task.true_law
        info['symmetries'] = self.physics_task.symmetries
        info['conserved_quantities'] = self.physics_task.conserved_quantities
        info['task_name'] = self.physics_task.name
        info['task_difficulty'] = self.physics_task.difficulty
        # The `info['variables']` should already be present from reset

        return obs, reward, terminated, truncated, info


# No longer need a concrete `HarmonicOscillatorEnv` here, as its data generation
# is handled by `HarmonicOscillatorData` in `generators.py`.
# The `PhysicsEnvironment` is now instantiated directly with a `PhysicsTask`.

# Example Usage (Illustrative in __main__)
if __name__ == "__main__":
    # Mock Grammar and SymbolicDiscoveryEnv if not fully available for standalone run
    class MockGrammar:
        def __init__(self):
            self.primitives = {'+': [None], '-': [None]} # Minimal
        def get_valid_actions(self, current_expression_node): return [0, 1, 2] # Dummy actions

    # Mock SymbolicDiscoveryEnv for testing PhysicsEnvironment in isolation
    class MockSymbolicDiscoveryEnv:
        def __init__(self, grammar, target_data, variables, max_depth, max_complexity, reward_config, action_space_size, provide_tree_structure):
            self.grammar = grammar
            self.target_data = target_data
            self.variables = variables
            self.max_depth = max_depth
            self.max_complexity = max_complexity
            self.reward_config = reward_config
            self.action_space = type('ActionSpace', (object,), {'n': action_space_size or 5})()
            self.observation_space = type('ObsSpace', (object,), {'shape': (10,)})() # Dummy obs space
            self.provide_tree_structure = provide_tree_structure
            # Mock current_state for observation_encoder if used
            self.current_state = type('TreeState', (object,), {
                'root': type('Node', (object,), {'node_type': type('NT', (object,), {'value': 'root'})(), 'children': []})(),
                'count_nodes': lambda self_node: 1,
                'is_complete': lambda : False
            })() # Minimal mock

        def reset(self, seed=None, options=None):
            # Simulate initial observation and empty info
            initial_obs = np.random.rand(self.observation_space.shape[0])
            info = {}
            if self.provide_tree_structure:
                info['tree_structure'] = {0: []} # Dummy structure
            return initial_obs, info

        def step(self, action):
            # Simulate a step: dummy next_obs, reward, done, truncated
            next_obs = np.random.rand(self.observation_space.shape[0])
            reward = np.random.rand() * 0.1 # Small random reward
            done = False
            truncated = False
            info = {}
            # Mock expression in info
            info['expression'] = "x_0 + 1" 
            info['complexity'] = 5
            return next_obs, reward, done, truncated, info

        def get_action_mask(self): # Mock get_action_mask
            return np.ones(self.action_space.n, dtype=bool)

    # Temporarily override SymbolicDiscoveryEnv for testing PhysicsEnvironment directly
    import sys
    sys.modules['janus.environments.base.symbolic_env'].SymbolicDiscoveryEnv = MockSymbolicDiscoveryEnv

    # --- Test Case Setup ---
    print("--- Testing PhysicsEnvironment with HarmonicOscillatorTask ---")
    grammar = MockGrammar()

    # Create a PhysicsTask for Harmonic Oscillator using the data generator
    physics_task_dist = PhysicsTaskDistribution()
    ho_task = physics_task_dist.get_task_by_name("harmonic_oscillator_energy")

    # Instantiate PhysicsEnvironment with the specific PhysicsTask
    env = PhysicsEnvironment(
        grammar=grammar,
        physics_task=ho_task, # Pass the task object directly
        max_depth=7,
        max_complexity=15,
        reward_config={'mse_weight': -1.0},
        action_space_size=ho_task.data_generator(1).shape[1] # Action space based on observable variables
    )

    # --- Simulate interaction ---
    print("\nEnvironment initialized with task:", env.physics_task.name)
    print("True Law:", env.physics_task.true_law)
    print("Variables:", [v.name for v in env.variables])

    obs, info = env.reset()
    print("\nEnv Reset. Initial Observation Shape:", obs.shape)
    print("Info after reset (contains trajectory_data, etc.):")
    # print(info.keys()) # Commented to avoid too much output
    print(f"  Sampled trajectory data for 'x' (first 5 points): {info['trajectory_data']['x'][:5].round(4)}")
    print(f"  Sampled true energy (should be constant): {info['trajectory_data']['energy_true'][:5].round(4)}")


    print("\nSimulating 3 steps:")
    for i in range(3):
        action = np.random.randint(env.action_space.n)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        print(f"  Step {i+1}: Action={action}, Reward={reward:.4f}, Done={terminated|truncated}")
        print(f"    Hypothesis in info: {step_info.get('expression')}")
        print(f"    Physics constants in info: {step_info.get('physical_constants')}")
        print(f"    True law in info: {step_info.get('true_law')}")

    print("\nPhysicsEnvironment test completed.")

    # Clean up mock
    del sys.modules['janus.environments.base.symbolic_env'].SymbolicDiscoveryEnv

